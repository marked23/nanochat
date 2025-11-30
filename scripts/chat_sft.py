"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid , which checkpoint to load the model from (base model or midtrained model)
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
# compute/precision
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 4 # max to avoid OOM
use_fsdp = False # enable FSDP for parameter-efficient training
# optimization
num_epochs = 1
num_iterations = -1 # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 100
eval_steps = 100
eval_metrics_every = 2000
eval_metrics_max_problems = 100 #1024
wandb_log_start_step = 10  # Skip logging first N steps to avoid distorting charts with initialization artifacts
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # original, uncompiled model (keep reference for checkpoint saving)

# Wrap model with FSDP if enabled
if use_fsdp:
    from nanochat.fsdp_utils import wrap_model_with_fsdp, get_fsdp_config
    print0("Wrapping model with FSDP for parameter-efficient training...")
    fsdp_config = get_fsdp_config(
        sharding_strategy="FULL_SHARD",  # ZeRO-3 equivalent
        mixed_precision_dtype=ptdtype,
        use_orig_params=True,  # Required for standard optimizers
        activation_checkpointing=True,  # Trade compute for memory
    )
    model = wrap_model_with_fsdp(orig_model, fsdp_config)
    print0(f"Model wrapped with FSDP (sharding_strategy=FULL_SHARD, activation_checkpointing=True)")
    # For FSDP, we'll use the wrapped model directly
    # orig_model still points to the original unwrapped model for reference
else:
    print0("Using standard DDP (FSDP disabled)")

# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(orig_model, tokenizer) # will be used for inline model evaluation only (use unwrapped model)

# -----------------------------------------------------------------------------
# Task data mixture we'll train on
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
    CustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
    SimpleSpelling(size=300, split="train"), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=300, split="train"), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
val_ds = SmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        
        # Debug: Check if all targets are masked
        num_valid_targets = (targets != -1).sum().item()
        if num_valid_targets == 0:
            print0(f"WARNING: Generated batch with all targets masked! Batch size: {nrows}, ncols: {ncols}")
            print0(f"  Conversation lengths: {[len(ids) for ids, mask in batch]}")
            print0(f"  Mask sums: {[sum(mask) for ids, mask in batch]}")
        
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []
        # Yield any remaining partial batch at epoch end
        if len(batch) > 0:
            yield collate_and_yield(batch)
            batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    # derive num_iterations from num_epochs and the size of the dataset
    assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

if use_fsdp:
    # With FSDP, use standard (non-distributed) optimizers
    # FSDP handles parameter/gradient sharding automatically
    print0("Setting up standard optimizers for FSDP...")

    # IMPORTANT: FSDP flattens all parameters to 1D for efficient sharding
    # We must use parameter NAMES (not shapes) to determine which optimizer to use
    # Note: Muon optimizer requires 2D tensors for Newton-Schulz orthogonalization,
    # but FSDP flattens everything to 1D. Solution: Use AdamW for all parameters.
    
    # Group parameters by name patterns (matching original 3-group setup)
    matrix_params = []        # transformer.h.* weights -> matrix_lr
    embedding_params = []     # transformer.wte.* -> embedding_lr
    lm_head_params = []       # lm_head.* -> unembedding_lr

    # Collect parameters by their name pattern
    for name, param in model.named_parameters():
        # FSDP wraps parameter names with _fsdp_wrapped_module prefix
        if 'transformer.wte' in name and 'weight' in name:
            # Token embedding
            embedding_params.append(param)
        elif 'lm_head' in name:
            # Output projection (unembedding)
            lm_head_params.append(param)
        elif 'transformer.h.' in name and 'weight' in name:
            # Transformer block weights: c_q, c_k, c_v, c_proj (attention), c_fc (MLP)
            # FSDP flattens to 1D, so we can't filter by ndim
            matrix_params.append(param)

    # Debug: Print parameter distribution summary
    print0(f"Parameter distribution:")
    print0(f"  Matrix params (AdamW @ matrix_lr): {len(matrix_params)}")
    print0(f"  Embedding params (AdamW @ embedding_lr): {len(embedding_params)}")
    print0(f"  LM head params (AdamW @ unembedding_lr): {len(lm_head_params)}")

    # Sanity check: ensure we have parameters
    total_params = len(matrix_params) + len(embedding_params) + len(lm_head_params)
    if total_params == 0:
        raise ValueError("No parameters found! Check FSDP wrapping and parameter name matching.")

    # Create single AdamW optimizer with 3 parameter groups (different learning rates)
    # Note: Can't use Muon because it requires 2D tensors, FSDP flattens to 1D
    print0("NOTE: Using AdamW for all parameters (Muon requires 2D tensors, FSDP flattens to 1D)")
    param_groups = []
    if len(lm_head_params) > 0:
        param_groups.append({'params': lm_head_params, 'lr': unembedding_lr})
    if len(embedding_params) > 0:
        param_groups.append({'params': embedding_params, 'lr': embedding_lr})
    if len(matrix_params) > 0:
        param_groups.append({'params': matrix_params, 'lr': matrix_lr})
    
    optimizers = [
        torch.optim.AdamW(param_groups, weight_decay=weight_decay),
    ]
    print0(f"Created AdamW optimizer with {len(param_groups)} parameter groups ({total_params} total params)")
else:
    # With DDP, use distributed optimizers
    print0("Setting up distributed optimizers (DistMuon, DistAdamW)...")
    optimizers = orig_model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )

# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
step = 0
train_iter = iter(train_loader)
start_time = time.time()
step_times = []  # Track recent step times for throughput calculation
cumulative_tokens = 0  # Track total tokens seen across all steps

for step in range(num_iterations):
    last_step = step == num_iterations - 1
    step_start_time = time.time()

    # evaluate the validation loss (before the first step and after each eval_every steps)
    if step > 0 and step % eval_every == 0:
        print0(f"[EVAL DEBUG] Step {step:05d}: Starting validation loss evaluation")
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        nan_count = 0
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            # Only include non-NaN losses (NaN occurs when all targets are masked)
            if not torch.isnan(loss):
                losses.append(loss)
            else:
                nan_count += 1
        if len(losses) > 0:
            val_loss = torch.stack(losses).mean() # average over eval_steps
            if ddp:
                print0(f"[NCCL DEBUG] Rank {ddp_rank}: 1 Starting all_reduce for validation loss")
                print0(f"[NCCL DEBUG] Rank {ddp_rank}: val_loss={val_loss.item():.6f}")
                if use_fsdp:
                    # FSDP: manually average loss across ranks
                    # (FSDP handles gradients but not arbitrary tensors)
                    print0(f"[NCCL DEBUG] Rank {ddp_rank}: About to call all_reduce (SUM) on val_loss")
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    print0(f"[NCCL DEBUG] Rank {ddp_rank}: Completed all_reduce on val_loss")
                    val_loss = val_loss / ddp_world_size
                else:
                    # DDP: average across ranks
                    print0(f"[NCCL DEBUG] Rank {ddp_rank}: About to call all_reduce (AVG) on val_loss")
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    print0(f"[NCCL DEBUG] Rank {ddp_rank}: Completed all_reduce on val_loss")
                print0(f"[NCCL DEBUG] Rank {ddp_rank}: Final averaged val_loss={val_loss.item():.6f}")
            val_loss = val_loss.item()
        else:
            val_loss = float('nan')  # All batches had NaN loss
        if nan_count > 0:
            print0(f"Step {step:05d} | Warning: {nan_count}/{eval_steps} validation batches had all targets masked (NaN loss)")
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        print0(f"[EVAL DEBUG] Step {step:05d}: Completed validation loss evaluation")
        if step >= wandb_log_start_step:
            wandb_run.log({
                "step": step,
                "val_loss": val_loss,
            })
        model.train()

    # evaluate accuracy of the multiple choice tasks (which are quick to run)
    if step > 0 and step % eval_metrics_every == 0:
        print0(f"[EVAL DEBUG] Step {step:05d}: Starting metrics evaluation (MMLU, ARC-Easy)")
        model.eval()
        metrics = {}
        # IMPORTANT: When using FSDP, pass the unwrapped original model to evaluation.
        # FSDP's all_gather operations during forward pass can timeout in eval scenarios
        # where ranks process different amounts of data (variable batch sizes per rank).
        # The unwrapped model avoids these collective operation complexities.
        eval_model = orig_model if use_fsdp else model
        if use_fsdp:
            orig_model.eval()  # Ensure unwrapped model is also in eval mode
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval("MMLU", eval_model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", eval_model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        print0(f"[EVAL DEBUG] Step {step:05d}: Completed metrics evaluation")
        if step >= wandb_log_start_step:
            wandb_run.log({
                "step": step,
                **metrics,
            })
        model.train()
        if use_fsdp:
            orig_model.train()  # Keep unwrapped model in sync

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach() # for logging
        # Skip backward if loss is NaN (happens when all targets are masked)
        if not torch.isnan(loss):
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # Calculate gradient norm (before clipping/stepping)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    cumulative_tokens += num_tokens_item
    
    # Calculate performance metrics
    step_end_time = time.time()
    step_duration = step_end_time - step_start_time
    step_times.append(step_duration)
    
    # Keep last 10 steps for smoothed throughput calculation
    if len(step_times) > 10:
        step_times.pop(0)
    
    # Calculate throughput metrics
    avg_step_time = sum(step_times) / len(step_times)
    tokens_per_sec = num_tokens_item / step_duration if step_duration > 0 else 0
    
    # Calculate percent per second (progress rate independent of batch size)
    # This shows what % of total training completes per second
    percent_per_sec = (100.0 / num_iterations / step_duration) if step > 3 else 0
    
    # Calculate ETA
    remaining_steps = num_iterations - step - 1
    eta_seconds = remaining_steps * avg_step_time
    eta_minutes = eta_seconds / 60.0
    
    # Critical metrics for WandB
    # 1. Perplexity - more interpretable than loss
    train_perplexity = math.exp(min(train_loss_item, 10))  # Cap at 10 for stability
    
    # 2. Per-group learning rates (3 groups for FSDP: lm_head, embedding, matrix)
    lr_dict = {}
    if use_fsdp and len(optimizers[0].param_groups) >= 3:
        lr_dict = {
            "lr/lm_head": optimizers[0].param_groups[0]['lr'],
            "lr/embedding": optimizers[0].param_groups[1]['lr'],
            "lr/matrix": optimizers[0].param_groups[2]['lr'],
        }
    
    # 3. Gradient norm (already calculated above)
    grad_norm_item = grad_norm.item()
    
    # 4. Train/val gap (only available when we have recent val_loss)
    train_val_gap = train_loss_item - val_loss if 'val_loss' in locals() else None
    
    # 5. Cumulative tokens seen
    
    # Format throughput string with aligned columns
    throughput_str = f"| {avg_step_time:5.2f} s/step | {tokens_per_sec:6,.0f} tok/s | {percent_per_sec:5.3f}%/s | ETA: {eta_minutes:5.1f}m"
    
    print0(f"Step {step:05d}/{num_iterations:05d} | Loss: {train_loss_item:8.6f} | lrm: {lrm:6.6f} | tokens: {num_tokens_item:6,} {throughput_str}")
    
    # Build wandb log dict with all metrics (skip first N steps to avoid distorting charts)
    if step >= wandb_log_start_step:
        wandb_log_dict = {
            "step": step,
            "lrm": lrm,
            "train_loss": train_loss_item,
            "train_perplexity": train_perplexity,
            "num_tokens": num_tokens_item,
            "tokens_total_seen": cumulative_tokens,
            "grad_norm": grad_norm_item,
            "sec_per_step": avg_step_time,
            "tokens_per_sec": tokens_per_sec,
            "percent_per_sec": percent_per_sec,
            "eta_minutes": eta_minutes,
        }
        # Add per-group learning rates if available
        wandb_log_dict.update(lr_dict)
        # Add train/val gap if available
        if train_val_gap is not None:
            wandb_log_dict["loss_train_val_gap"] = train_val_gap
        
        wandb_run.log(wandb_log_dict)
    step += 1

# Final evaluation after training completes
print0(f"[EVAL DEBUG] Final evaluation at step {step:05d}")
print0(f"[EVAL DEBUG] Step {step:05d}: Starting validation loss evaluation")
model.eval()
val_iter = iter(build_val_loader())
losses = []
nan_count = 0
for _ in range(eval_steps):
    val_inputs, val_targets = next(val_iter)
    with torch.no_grad(), autocast_ctx:
        loss = model(val_inputs, val_targets)
    # Only include non-NaN losses (NaN occurs when all targets are masked)
    if not torch.isnan(loss):
        losses.append(loss)
    else:
        nan_count += 1
if len(losses) > 0:
    val_loss = torch.stack(losses).mean() # average over eval_steps
    if ddp:
        print0(f"[NCCL DEBUG] Rank {ddp_rank}: 2 Starting all_reduce for validation loss")
        print0(f"[NCCL DEBUG] Rank {ddp_rank}: val_loss={val_loss.item():.6f}")
        if use_fsdp:
            print0(f"[NCCL DEBUG] Rank {ddp_rank}: About to call all_reduce (SUM) on val_loss")
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            print0(f"[NCCL DEBUG] Rank {ddp_rank}: Completed all_reduce on val_loss")
            val_loss = val_loss / ddp_world_size
        else:
            print0(f"[NCCL DEBUG] Rank {ddp_rank}: About to call all_reduce (AVG) on val_loss")
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(f"[NCCL DEBUG] Rank {ddp_rank}: Completed all_reduce on val_loss")
        print0(f"[NCCL DEBUG] Rank {ddp_rank}: Final averaged val_loss={val_loss.item():.6f}")
    val_loss = val_loss.item()
else:
    val_loss = float('nan')  # All batches had NaN loss
if nan_count > 0:
    print0(f"Step {step:05d} | Warning: {nan_count}/{eval_steps} validation batches had all targets masked (NaN loss)")
print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
print0(f"[EVAL DEBUG] Step {step:05d}: Completed validation loss evaluation")

# Final metrics evaluation
print0(f"[EVAL DEBUG] Step {step:05d}: Starting metrics evaluation (MMLU, ARC-Easy)")
model.eval()
metrics = {}
eval_model = orig_model if use_fsdp else model
if use_fsdp:
    orig_model.eval()  # Ensure unwrapped model is in eval mode
metrics["mmlu_acc"] = run_chat_eval("MMLU", eval_model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", eval_model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
print0(f"[EVAL DEBUG] Step {step:05d}: Completed metrics evaluation")
wandb_run.log({
    "step": step,
    "val_loss": val_loss,
    **metrics,
})
model.train()
if use_fsdp:
    orig_model.train()  # Keep unwrapped model in sync

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = orig_model.config.n_layer
    model_tag = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = orig_model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer

    # Get state dict (FSDP-aware)
    if use_fsdp:
        from nanochat.fsdp_utils import get_fsdp_state_dict
        print0("Consolidating FSDP state dict to rank 0...")
        state_dict = get_fsdp_state_dict(model, full_state_dict=True)
    else:
        state_dict = model.state_dict()

    save_checkpoint(
        checkpoint_dir,
        step,
        state_dict,
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
        }
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
