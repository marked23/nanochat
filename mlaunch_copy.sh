#!/bin/bash

# mlaunch.sh - Master Launch Script for 6x AMD GPU Setup
# This script runs the complete nanochat training pipeline optimized for 6x7900XTX GPUs
# It combines all activities from speedrun.sh with optimizations for AMD hardware
#
# Usage: bash mlaunch.sh
# Or with wandb: WANDB_RUN=my_run bash mlaunch.sh

# =============================================================================
# CONFIGURATION SECTION - Activity Enable/Disable
# =============================================================================

# Core Setup
ENABLE_GPU_CHECK=false
ENABLE_REPORT_RESET=true           # Reset and initialize report directory
ENABLE_RUST_INSTALL=false          # Install Rust/Cargo (set false if already installed)
ENABLE_TOKENIZER_BUILD=false       # Build rustbpe tokenizer (set false if already built)
ENABLE_TOKENIZER_TRAIN=false       # Train the tokenizer (set false if already trained)
ENABLE_TOKENIZER_EVAL=false        # Evaluate tokenizer compression ratio

# Dataset
ENABLE_DATASET_DOWNLOAD=false      # Download training data shards

# Base Model Training
ENABLE_BASE_TRAIN=false            # Pretrain the base d20 model
ENABLE_BASE_LOSS=false             # Evaluate model on train/val data
ENABLE_BASE_EVAL=false             # Evaluate model on CORE tasks

# Midtraining
ENABLE_IDENTITY_DOWNLOAD=false     # Download synthetic identity conversations
ENABLE_MID_TRAIN=false              # Run midtraining
ENABLE_MID_EVAL=false              # Evaluate after midtraining (SLOW - can skip)

# Supervised Fine-tuning
ENABLE_SFT_TRAIN=true              # Run supervised fine-tuning
ENABLE_SFT_EVAL=false              # Evaluate after SFT (SLOW - can skip)

# Reinforcement Learning (optional)
ENABLE_RL_TRAIN=false              # Run reinforcement learning
ENABLE_RL_EVAL=false               # Evaluate RL model on GSM8K

# Final Report
ENABLE_REPORT_GENERATE=true        # Generate final markdown report

# =============================================================================
# CONFIGURATION SECTION - Training Parameters
# =============================================================================

# Hardware Configuration
NUM_GPUS=6                         # Number of GPUs to use
MODEL_DEPTH=20                     # Model depth (d20 = 561M parameters)

# Base Training Parameters
DEVICE_BATCH_SIZE=7                # Per-GPU batch size (6 works with depth 20, BF16: 6=15GB, 7=17GB, 8=19GB VRAM)
MAX_SEQ_LEN=2048                   # Context length
NUM_ITERATIONS=-1                 # Number of training iterations (-1 = train to completion)
RESUME_FROM_STEP=-1               # Resume from checkpoint step (-1 = start fresh)
EVAL_EVERY=1000                      # Validation loss eval (CHEAPEST - quick val loss check, ~20M tokens)
SAMPLE_EVERY=4000                    # Text sampling (MEDIUM - generate example outputs, main GPU only)
CORE_METRIC_EVERY=9000              # CORE benchmark eval (EXPENSIVE - full academic benchmark suite)
SAVE_EVERY=5000                     # Checkpoint saving frequency (-1 = only save at end of run)
SPLIT_TOKENS=10407936              # Tokens for base_loss eval (must be divisible by batch×seq×gpus)

# Batch Size Calculation (target: 393216 tokens)
WORLD_TOKENS_PER_STEP=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NUM_GPUS))
TARGET_TOTAL_BATCH_SIZE=393216
GRAD_ACCUM_STEPS=$((TARGET_TOTAL_BATCH_SIZE / WORLD_TOKENS_PER_STEP))
if [ $GRAD_ACCUM_STEPS -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi
TOTAL_BATCH_SIZE=$((GRAD_ACCUM_STEPS * WORLD_TOKENS_PER_STEP))

# Learning Rates (tuned for 393216 batch size)
MATRIX_LR=0.02
EMBEDDING_LR=0.2
UNEMBEDDING_LR=0.004

# Dataset Configuration
DATASET_INITIAL_SHARDS=8           # Initial shards for tokenizer training (~2B chars)
DATASET_TOTAL_SHARDS=240           # Total shards for base training (~54B chars)

# Tokenizer Configuration
TOKENIZER_MAX_CHARS=2000000000     # Max characters for tokenizer training (2B)
TOKENIZER_VOCAB_SIZE=65538         # Vocab size (must be divisible by NUM_GPUS=6, default 65536 is not)

# Midtraining Configuration
MID_NUM_ITERATIONS=-1             # Number of midtraining iterations (-1 = train to completion)
MID_EVAL_EVERY=500                # Validation frequency for midtraining

# SFT Configuration
SFT_DEVICE_BATCH_SIZE=2            # Per-GPU batch size for SFT (default: 4)
SFT_EXAMPLES_PER_STEP=$((SFT_DEVICE_BATCH_SIZE * NUM_GPUS))  # Examples per step across all GPUs
SFT_TARGET_EXAMPLES=24             # Target examples per step (must be divisible by SFT_EXAMPLES_PER_STEP)
SFT_NUM_ITERATIONS=-1             # Number of SFT iterations (-1 = use num_epochs)

# RL Configuration  
RL_DEVICE_BATCH_SIZE=8             # Per-GPU batch size for RL (default: 8)
RL_NUM_ITERATIONS=-1              # Number of RL iterations (-1 = train to completion)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default intermediate artifacts directory
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy  # Special case: skips logging to wandb
fi

# Log file setup
LOG_FILE="$SCRIPT_DIR/training.log"

# =============================================================================
# PYTORCH/NCCL ENVIRONMENT VARIABLES
# =============================================================================

export OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# NCCL basic settings
export NCCL_DEBUG=OFF
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

# NCCL P2P and communication settings
export NCCL_P2P_LEVEL=PHB
export NCCL_P2P_DISABLE=0
export NCCL_P2P_NCHANNELS=5
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DMABUF_ENABLE=1

# NCCL buffer and channel settings
export NCCL_BUFFSIZE=33554432
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=128

# NCCL retry settings
export NCCL_CONNECTION_RETRY=5
export NCCL_CONNECTION_RETRY_TIMEOUT=60

# Disable alternative collective algorithms
export NCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export MSCCL_ENABLE=0

# =============================================================================
# ROCM-SPECIFIC OPTIMIZATIONS
# =============================================================================

export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5
export HSA_ENABLE_INTERRUPT=1
export ROCR_ENABLE_PRE_VEGA_FINALIZATION=0
export HSA_XNACK=0
export PYTORCH_NO_FLASH_ATTN_WARNINGS=1
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True # Not supported on XTX

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "=============================================================================" | tee -a "$LOG_FILE"
    log "$1"
    echo "=============================================================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

run_command() {
    local description="$1"
    shift
    log "Running: $description"
    log "Command: $*"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log "ERROR: Command failed with exit code $exit_code"
        log "FATAL: Stopping execution due to error in: $description"
        log "Check the log file for details: $LOG_FILE"
        exit $exit_code
    fi
    log "SUCCESS: $description completed"
    return 0
}

# =============================================================================
# STARTUP
# =============================================================================

log_section "Starting mlaunch.sh - Master Launch Script"

log "Configuration Summary:"
log "  GPUs: $NUM_GPUS"
log "  Device batch size: $DEVICE_BATCH_SIZE"
log "  Max sequence length: $MAX_SEQ_LEN"
log "  World tokens per step: $WORLD_TOKENS_PER_STEP"
log "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
log "  Total batch size: $TOTAL_BATCH_SIZE (target: $TARGET_TOTAL_BATCH_SIZE)"
log "  Matrix LR: $MATRIX_LR"
log "  Embedding LR: $EMBEDDING_LR"
log "  Unembedding LR: $UNEMBEDDING_LR"
log "  Num iterations: $NUM_ITERATIONS"
log "  Resume from step: $RESUME_FROM_STEP"
log "  SFT device batch size: $SFT_DEVICE_BATCH_SIZE"
log "  SFT examples per step: $SFT_EXAMPLES_PER_STEP"
log "  SFT target examples: $SFT_TARGET_EXAMPLES"
log "  RL device batch size: $RL_DEVICE_BATCH_SIZE"
log "  WANDB run: $WANDB_RUN"
log "  Log file: $LOG_FILE"

# =============================================================================
# VIRTUAL ENVIRONMENT ACTIVATION
# =============================================================================

log_section "Virtual Environment Activation"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    log "Activating virtual environment at $SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate" 2>&1 | tee -a "$LOG_FILE"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    log "Activating virtual environment at $SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate" 2>&1 | tee -a "$LOG_FILE"
else
    log "ERROR: No virtual environment found at $SCRIPT_DIR/.venv or $SCRIPT_DIR/venv"
    log "Please create one with: uv venv && uv sync --extra gpu"
    exit 1
fi

log "Virtual environment activated: $(which python)"

# =============================================================================
# GPU AVAILABILITY CHECK
# =============================================================================

if [ "$ENABLE_GPU_CHECK" = true ]; then

    log_section "GPU Availability Check"

    log "Checking PyTorch and GPU setup..."
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "ERROR")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "ERROR")
    DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

    log "PyTorch version: $PYTORCH_VERSION"
    log "GPUs available: $CUDA_AVAILABLE"
    log "GPU count: $DEVICE_COUNT"

    if [[ "$PYTORCH_VERSION" == *"rocm"* ]]; then
        log "✓ ROCm PyTorch detected"
    elif [[ "$PYTORCH_VERSION" == *"cu"* ]]; then
        log "✗ ERROR: CUDA PyTorch detected, but AMD GPUs require ROCm PyTorch"
        log "Please reinstall with: uv sync --reinstall-package torch"
        exit 1
    fi

    if [ "$CUDA_AVAILABLE" != "True" ] || [ "$DEVICE_COUNT" -lt "$NUM_GPUS" ]; then
        log "✗ ERROR: Expected $NUM_GPUS GPUs, but found $DEVICE_COUNT"
        log "GPU check failed. Please verify:"
        log "  1. ROCm drivers are installed"
        log "  2. ROCm PyTorch is installed (not CUDA version)"
        log "  3. Environment variables are set correctly"
        log "  4. GPUs are visible with: rocm-smi"
        exit 1
    fi

    log "✓ GPU check passed: $DEVICE_COUNT GPUs available"
else
    log_section "GPU Availability Check - SKIPPED"
fi

# =============================================================================
# REPORT INITIALIZATION
# =============================================================================

if [ "$ENABLE_REPORT_RESET" = true ]; then
    log_section "Report Initialization"
    run_command "Reset report directory" python -m nanochat.report reset
fi

# =============================================================================
# RUST/CARGO INSTALLATION
# =============================================================================

if [ "$ENABLE_RUST_INSTALL" = true ]; then
    log_section "Rust/Cargo Installation"
    if command -v cargo &> /dev/null; then
        log "Rust/Cargo already installed, skipping installation"
    else
        log "Installing Rust/Cargo..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tee -a "$LOG_FILE"
        source "$HOME/.cargo/env"
        log "Rust/Cargo installed successfully"
    fi
else
    log_section "Rust/Cargo Installation - SKIPPED"
    # Still need to source cargo env if it exists
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

# =============================================================================
# TOKENIZER BUILD
# =============================================================================

if [ "$ENABLE_TOKENIZER_BUILD" = true ]; then
    log_section "Tokenizer Build"
    run_command "Build rustbpe tokenizer" uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
else
    log_section "Tokenizer Build - SKIPPED"
fi

# =============================================================================
# DATASET DOWNLOAD
# =============================================================================

if [ "$ENABLE_DATASET_DOWNLOAD" = true ]; then
    log_section "Dataset Download"
    
    # Download initial shards for tokenizer training
    run_command "Download initial $DATASET_INITIAL_SHARDS data shards (~2B chars)" \
        python -m nanochat.dataset -n $DATASET_INITIAL_SHARDS
    
    # Start downloading all shards in background
    log "Starting background download of $DATASET_TOTAL_SHARDS total shards..."
    python -m nanochat.dataset -n $DATASET_TOTAL_SHARDS >> "$LOG_FILE" 2>&1 &
    DATASET_DOWNLOAD_PID=$!
    log "Background download PID: $DATASET_DOWNLOAD_PID"
else
    log_section "Dataset Download - SKIPPED"
    DATASET_DOWNLOAD_PID=""
fi

# =============================================================================
# TOKENIZER TRAINING
# =============================================================================

if [ "$ENABLE_TOKENIZER_TRAIN" = true ]; then
    log_section "Tokenizer Training"
    run_command "Train tokenizer on $TOKENIZER_MAX_CHARS characters" \
        python -m scripts.tok_train --max_chars=$TOKENIZER_MAX_CHARS --vocab_size=$TOKENIZER_VOCAB_SIZE
else
    log_section "Tokenizer Training - SKIPPED"
fi

# =============================================================================
# TOKENIZER EVALUATION
# =============================================================================

if [ "$ENABLE_TOKENIZER_EVAL" = true ]; then
    log_section "Tokenizer Evaluation"
    run_command "Evaluate tokenizer compression ratio" python -m scripts.tok_eval
else
    log_section "Tokenizer Evaluation - SKIPPED"
fi

# =============================================================================
# WAIT FOR DATASET DOWNLOAD
# =============================================================================

if [ "$ENABLE_DATASET_DOWNLOAD" = true ] && [ -n "$DATASET_DOWNLOAD_PID" ]; then
    log_section "Waiting for Dataset Download"
    log "Waiting for background dataset download (PID: $DATASET_DOWNLOAD_PID) to complete..."
    wait $DATASET_DOWNLOAD_PID
    log "Dataset download completed"
fi

# =============================================================================
# BASE MODEL TRAINING
# =============================================================================

if [ "$ENABLE_BASE_TRAIN" = true ]; then
    log_section "Base Model Training (Pretraining)"
    
    CMD=(
        torchrun --standalone
        --nproc_per_node=$NUM_GPUS
        -m scripts.base_train --
        --depth=$MODEL_DEPTH
        --num_iterations=$NUM_ITERATIONS
        --resume_from_step=$RESUME_FROM_STEP
        --device_batch_size=$DEVICE_BATCH_SIZE
        --total_batch_size=$TOTAL_BATCH_SIZE
        --max_seq_len=$MAX_SEQ_LEN
        --matrix_lr=$MATRIX_LR
        --embedding_lr=$EMBEDDING_LR
        --unembedding_lr=$UNEMBEDDING_LR
        --eval_every=$EVAL_EVERY
        --core_metric_every=$CORE_METRIC_EVERY
        --sample_every=$SAMPLE_EVERY
        --save_every=$SAVE_EVERY
        --run=$WANDB_RUN
    )
    
    run_command "Pretrain d$MODEL_DEPTH model" "${CMD[@]}"
else
    log_section "Base Model Training - SKIPPED"
fi

# =============================================================================
# BASE MODEL LOSS EVALUATION
# =============================================================================

if [ "$ENABLE_BASE_LOSS" = true ]; then
    log_section "Base Model Loss Evaluation"
    run_command "Evaluate model on train/val data" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_loss -- --split_tokens=$SPLIT_TOKENS --device_batch_size=$DEVICE_BATCH_SIZE
else
    log_section "Base Model Loss Evaluation - SKIPPED"
fi

# =============================================================================
# BASE MODEL CORE EVALUATION
# =============================================================================

if [ "$ENABLE_BASE_EVAL" = true ]; then
    log_section "Base Model CORE Evaluation"
    run_command "Evaluate model on CORE tasks" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_eval
else
    log_section "Base Model CORE Evaluation - SKIPPED"
fi

# =============================================================================
# IDENTITY CONVERSATIONS DOWNLOAD
# =============================================================================

if [ "$ENABLE_IDENTITY_DOWNLOAD" = true ]; then
    log_section "Identity Conversations Download"
    run_command "Download synthetic identity conversations" \
        curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
else
    log_section "Identity Conversations Download - SKIPPED"
fi

# =============================================================================
# MIDTRAINING
# =============================================================================

if [ "$ENABLE_MID_TRAIN" = true ]; then
    log_section "Midtraining"
    run_command "Run midtraining" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.mid_train -- \
        --device_batch_size=$DEVICE_BATCH_SIZE \
        --total_batch_size=$TOTAL_BATCH_SIZE \
        --max_seq_len=$MAX_SEQ_LEN \
        --num_iterations=$MID_NUM_ITERATIONS \
        --eval_every=$MID_EVAL_EVERY \
        --run=$WANDB_RUN
else
    log_section "Midtraining - SKIPPED"
fi

# =============================================================================
# MIDTRAINING EVALUATION
# =============================================================================

if [ "$ENABLE_MID_EVAL" = true ]; then
    log_section "Midtraining Evaluation"
    run_command "Evaluate after midtraining" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.chat_eval -- -i mid
else
    log_section "Midtraining Evaluation - SKIPPED"
fi

# =============================================================================
# SUPERVISED FINE-TUNING
# =============================================================================

if [ "$ENABLE_SFT_TRAIN" = true ]; then
    log_section "Supervised Fine-Tuning"
    run_command "Run supervised fine-tuning" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.chat_sft -- \
        --device_batch_size=$SFT_DEVICE_BATCH_SIZE \
        --target_examples_per_step=$SFT_TARGET_EXAMPLES \
        --num_iterations=$SFT_NUM_ITERATIONS \
        --run=$WANDB_RUN
else
    log_section "Supervised Fine-Tuning - SKIPPED"
fi

# =============================================================================
# SFT EVALUATION
# =============================================================================

if [ "$ENABLE_SFT_EVAL" = true ]; then
    log_section "SFT Evaluation"
    run_command "Evaluate after SFT" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.chat_eval -- -i sft
else
    log_section "SFT Evaluation - SKIPPED"
fi

# =============================================================================
# REINFORCEMENT LEARNING (Optional)
# =============================================================================

if [ "$ENABLE_RL_TRAIN" = true ]; then
    log_section "Reinforcement Learning"
    run_command "Run reinforcement learning" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.chat_rl -- \
        --device_batch_size=$RL_DEVICE_BATCH_SIZE \
        --num_iterations=$RL_NUM_ITERATIONS \
        --run=$WANDB_RUN
else
    log_section "Reinforcement Learning - SKIPPED"
fi

# =============================================================================
# RL EVALUATION
# =============================================================================

if [ "$ENABLE_RL_EVAL" = true ]; then
    log_section "RL Evaluation"
    run_command "Evaluate RL model on GSM8K" \
        torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.chat_eval -- -i rl -a GSM8K
else
    log_section "RL Evaluation - SKIPPED"
fi

# =============================================================================
# REPORT GENERATION
# =============================================================================

if [ "$ENABLE_REPORT_GENERATE" = true ]; then
    log_section "Report Generation"
    run_command "Generate final report" python -m nanochat.report generate
else
    log_section "Report Generation - SKIPPED"
fi

# =============================================================================
# COMPLETION
# =============================================================================

log_section "Training Pipeline Complete!"

log ""
log "Next steps:"
log "  - Chat with your model via CLI:"
log "    python -m scripts.chat_cli -p \"Why is the sky blue?\""
log ""
log "  - Chat with your model via WebUI:"
log "    python -m scripts.chat_web"
log ""
log "  - View the full report at:"
log "    $NANOCHAT_BASE_DIR/report/report.md"
log ""
log "Training log saved to: $LOG_FILE"
