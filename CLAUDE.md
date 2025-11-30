# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanochat is a minimal, hackable implementation of a full LLM training pipeline (tokenization → pretraining → midtraining → SFT → RL → inference → web UI) designed to train ChatGPT-like models on budgets under $1000. The codebase is intentionally kept minimal (~8K lines, 45 files) and readable. It is designed to run end-to-end via single shell scripts.

**Design Philosophy**: This is not an "LLM framework" - avoid adding configuration complexity, model factories, or if-then-else monsters. Keep it a single, cohesive, minimal, maximally-forkable "strong baseline".

## Development Commands

### Environment Setup
```bash
# Activate the uv virtual environment
source .venv/bin/activate
```

### Training Pipeline

**Quick start** (4 hours, ~$100 on 8XH100):
```bash
bash speedrun.sh  # Trains d20 model (561M params)
```

**Custom AMD GPU launcher** (6x AMD 7900 XTX):
```bash
bash launch_6x_amd_xtx.sh  # Configured for 6x AMD GPUs
```

**Individual training stages** (use `torchrun` for distributed):
```bash
# Tokenizer training
python -m scripts.tok_train

# Base model pretraining
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Midtraining (continue pretraining with task-specific data)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# Supervised fine-tuning (SFT)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Reinforcement learning (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

**Single GPU** (omit `torchrun`, code auto-switches to gradient accumulation):
```bash
python -m scripts.base_train
```

### Inference

```bash
# CLI chat interface
python -m scripts.chat_cli

# Web UI (ChatGPT-like interface)
python -m scripts.chat_web  # Then visit http://localhost:8000
```

### Evaluation

```bash
# Base model CORE score
python -m scripts.base_eval

# Chat model evaluation (ARC, GSM8K, MMLU, etc.)
python -m scripts.chat_eval
```

### Testing

```bash
# Run tokenizer tests
python -m pytest tests/test_rustbpe.py -v -s

# Run engine tests
python -m pytest tests/test_engine.py -v -s
```

### Dataset Management

```bash
# Download N shards of FineWeb training data (~250MB each)
python -m nanochat.dataset -n 240
```

## Code Architecture

### Training Pipeline Flow

```
1. Tokenizer Training (tok_train.py)
   ↓
2. Base Model Pretraining (base_train.py) → "Base" checkpoint
   ↓
3. Midtraining (mid_train.py) → "Mid" checkpoint
   ↓
4. Supervised Fine-Tuning (chat_sft.py) → "SFT" checkpoint
   ↓
5. Reinforcement Learning (chat_rl.py) → "RL" checkpoint
```

Each stage loads the previous stage's checkpoint and continues training.

### Model Architecture (nanochat/gpt.py)

**GPT class** - The core transformer model with these features:
- Rotary embeddings (no learned positional embeddings)
- QK normalization
- Untied weights (separate token embedding and lm_head)
- ReLU² activation in MLP
- RMSNorm without learnable parameters
- No biases in linear layers
- Group-Query Attention (GQA) support
- Logits softcap of 15

**Model sizes** are denoted as `dN` where N = number of layers:
- d20: 561M parameters (default for speedrun)
- d26: ~900M parameters
- d32: 1.9B parameters

**Key method**: `setup_optimizers()` returns 3-group optimizer setup:
1. **Muon** optimizer for all 2D weight matrices (e.g., linear layers)
2. **AdamW** for token embeddings
3. **AdamW** for lm_head (output projection)

### Distributed Training (DDP)

**Current implementation**: Standard PyTorch DDP with NCCL backend

**Initialization** ([nanochat/common.py](nanochat/common.py)):
- `compute_init()`: Sets up DDP, initializes process group, returns device info
- `compute_cleanup()`: Destroys process group

**Custom distributed optimizers**:
- **DistMuon** ([nanochat/muon.py](nanochat/muon.py)): ZeRO-2 style sharding with manual reduce_scatter/all_gather
  - Gradients: averaged via reduce_scatter
  - Optimizer states: sharded (each rank owns a slice)
  - Parameters: replicated via all_gather

- **DistAdamW** ([nanochat/adamw.py](nanochat/adamw.py)): Similar ZeRO-2 pattern
  - Shards optimizer states across ranks
  - Manual reduce_scatter for gradients
  - all_gather to replicate updated parameters

Both optimizers use block-cyclic parameter assignment to determine ownership.

### Configuration System ([nanochat/configurator.py](nanochat/configurator.py))

nanochat uses a custom "Poor Man's Configurator" instead of argparse:

```python
# In training scripts, define config at module level:
device_batch_size = 32
learning_rate = 0.02
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('nanochat/configurator.py').read())
```

**Override from CLI**:
```bash
python -m scripts.base_train --device_batch_size=16 --learning_rate=0.01
```

**Override from file**:
```bash
python -m scripts.base_train config/my_config.py
```

This allows all hyperparameters to be simple module-level variables instead of `config.param` everywhere.

### Data Loading

**Pretraining** ([nanochat/dataloader.py](nanochat/dataloader.py)):
- `tokenizing_distributed_data_loader_with_state()`: Streams from Parquet files
- Each rank processes different row groups (DDP-aware)
- Supports approximate resume via state_dict
- Tokenizes on-the-fly with multi-threading
- Infinite iterator (multi-epoch)

**SFT/Chat** ([scripts/chat_sft.py](scripts/chat_sft.py)):
- `sft_data_generator()`: Generator-based, not DataLoader
- Manually pads variable-length conversations to max_seq_len
- Uses `<|assistant_end|>` as pad token
- Each rank samples different examples via `range(ddp_rank, len(dataset), ddp_world_size)`

### Checkpoint Management ([nanochat/checkpoint_manager.py](nanochat/checkpoint_manager.py))

**Checkpoint organization**:
```
$NANOCHAT_BASE_DIR/
├── base_checkpoints/d20/
│   ├── model_NNNNNN.pt
│   ├── optim_NNNNNN_rank0.pt
│   └── meta_NNNNNN.json
├── mid_checkpoints/d20/
├── chatsft_checkpoints/d20/
└── chatrl_checkpoints/d20/
```

**Functions**:
- `save_checkpoint(dir, step, model, optimizer, meta, rank)`: Rank 0 saves model, all ranks save optimizer
- `load_model(source, device, phase, model_tag, step)`: Load from "base", "mid", "sft", or "rl"

### Engine ([nanochat/engine.py](nanochat/engine.py))

Efficient inference with KV cache:
```python
engine = Engine(model, tokenizer)
response = engine.generate(prompt, max_new_tokens=256, temperature=1.0, top_k=50)
```

Implements:
- KV cache management
- Top-k sampling
- Streaming generation
- Special token handling (assistant_start, assistant_end, etc.)

### Task System ([tasks/](tasks/))

All tasks inherit from a common pattern:
- `TaskMixture`: Combines multiple tasks, samples uniformly
- `TaskSequence`: Concatenates tasks sequentially

Each task provides:
- `__len__()`: Number of examples
- `__getitem__(idx)`: Returns conversation format
- Conversation format: `[{"role": "user/assistant", "content": "..."}]`

**Available tasks**:
- ARC (science questions)
- GSM8K (math word problems)
- MMLU (multiple choice, broad topics)
- HumanEval (Python coding)
- SmolTalk (conversational data)
- SpellingBee (letter counting/spelling)

## Hardware-Specific Notes

### AMD GPU Support ([launch_6x_amd_xtx.sh](launch_6x_amd_xtx.sh))

The codebase includes AMD-specific optimizations for 6x AMD 7900 XTX (24GB VRAM):

**ROCm environment variables**:
```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_ENABLE_INTERRUPT=1
export HSA_XNACK=0
```

**NCCL tuning**:
```bash
export NCCL_P2P_LEVEL=PHB
export NCCL_P2P_DISABLE=0
export NCCL_BUFFSIZE=33554432
export NCCL_IB_DISABLE=1
export NCCL_DMABUF_ENABLE=1
```

**Batch size auto-detection**: Script detects VRAM and sets appropriate batch sizes (e.g., 7 for 24GB GPUs).

### Memory Management

**For larger models** (to avoid OOM):
1. Reduce `--device_batch_size` (32 → 16 → 8 → 4 → 2)
2. Code automatically increases gradient accumulation to compensate
3. Turns parallel compute into sequential compute (same results, longer runtime)

**Memory allocator**:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
Note: Not supported on AMD 7900 XTX, disabled in [launch_6x_amd_xtx.sh](launch_6x_amd_xtx.sh)

## Important Development Patterns

### Rank-0 Logging

Always use `print0()` to avoid duplicate logs from all ranks:
```python
from nanochat.common import print0
print0(f"Training started")  # Only prints on rank 0
```

### Distributed Synchronization

Manual synchronization in training scripts:
```python
import torch.distributed as dist

# Average loss across ranks
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

# Sum tokens across ranks
dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
```

### Learning Rate Scaling

LR is scaled by model dimension:
```python
lr_scale = (model_dim / 768) ** -0.5
actual_lr = base_lr * lr_scale
```

This assumes the base LR was tuned for 768-dimensional models.

### Gradient Accumulation

Training scripts calculate grad_accum_steps automatically:
```python
examples_per_step = device_batch_size * ddp_world_size
grad_accum_steps = target_examples_per_step // examples_per_step
```

Loss is divided by grad_accum_steps before backward:
```python
loss = loss / grad_accum_steps
loss.backward()
```

## AMD-Specific Distributed Training Context

**Current Status**: The codebase uses standard DDP with custom ZeRO-2 style optimizers (DistMuon, DistAdamW).

**Future Direction**: User is evaluating conversion to FSDP/FSDP2/ZeRO for parameter-efficient training to enable larger models (1.9B+ params) on 24GB AMD GPUs. See https://github.com/marked23/fsdp2-minimal-rocm for working FSDP2 example on ROCm 6.4.1 + PyTorch 2.6.

## File Reference

**Core Model**:
- [nanochat/gpt.py](nanochat/gpt.py): GPT model architecture
- [nanochat/engine.py](nanochat/engine.py): Inference engine with KV cache
- [nanochat/tokenizer.py](nanochat/tokenizer.py): BPE tokenizer wrapper

**Training**:
- [scripts/base_train.py](scripts/base_train.py): Pretraining script
- [scripts/mid_train.py](scripts/mid_train.py): Midtraining script
- [scripts/chat_sft.py](scripts/chat_sft.py): Supervised fine-tuning
- [scripts/chat_rl.py](scripts/chat_rl.py): Reinforcement learning

**Distributed**:
- [nanochat/common.py](nanochat/common.py): DDP initialization, utilities
- [nanochat/muon.py](nanochat/muon.py): Distributed Muon optimizer
- [nanochat/adamw.py](nanochat/adamw.py): Distributed AdamW optimizer

**Data**:
- [nanochat/dataloader.py](nanochat/dataloader.py): Streaming parquet data loader
- [nanochat/dataset.py](nanochat/dataset.py): FineWeb download utilities
- [tasks/](tasks/): Task definitions (ARC, GSM8K, etc.)

**Infrastructure**:
- [nanochat/checkpoint_manager.py](nanochat/checkpoint_manager.py): Save/load checkpoints
- [nanochat/configurator.py](nanochat/configurator.py): CLI argument override system
- [rustbpe/](rustbpe/): Custom Rust BPE tokenizer trainer
