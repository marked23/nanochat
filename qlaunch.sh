#!/bin/bash

# This script adapts speedrun.sh for AMD ROCm GPUs with 6x24GB configuration
# Designed to run the "Best ChatGPT clone that ~$150 can buy" on AMD hardware
#
# PERFORMANCE TUNING NOTES:
# - Model dtype: BF16 (50% memory reduction vs FP32)
# - Batch size 8 with LR scaling maintains convergence quality
# - Target: ~21,400 tok/sec with efficient loss convergence
# - Monitor 'eff' metric in logs to ensure speed + quality balance

# 1) Example launch (simplest):
# bash qlaunch.sh
# 2) Example launch in a screen session:
# screen -L -Logfile nanochat.log -S nanochat bash qlaunch.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=nanochat_amd screen -L -Logfile nanochat.log -S nanochat bash qlaunch.sh

# -----------------------------------------------------------------------------
# Virtual Environment Activation
# -----------------------------------------------------------------------------

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "Warning: No virtual environment found at $SCRIPT_DIR/.venv or $SCRIPT_DIR/venv"
    echo "Please create one with: uv venv && uv sync"
    exit 1
fi

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------

NUM_GPUS=6
MODEL_DEPTH=20           # d20 model (561M parameters)
DEVICE_BATCH_SIZE=7      # Per-GPU batch size (BF16: 7=17GB, 8=19GB, 9=22GB VRAM)
MAX_SEQ_LEN=2048         # Context length

EVAL_EVERY=10           # Validation eval frequency (default: 250)
CORE_METRIC_EVERY=100   # CORE metric eval frequency (default: 2000)
SAMPLE_EVERY=20        # Sample generation frequency (default: 2000)
NUM_ITERATIONS=100        # Set to -1 to train all the way
RESUME_FROM_STEP=-1      # Set to -1 to start fresh, or step number to resume


# Calculate appropriate total_batch_size
# Must be divisible by (device_batch_size × max_seq_len × num_gpus)
WORLD_TOKENS_PER_STEP=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NUM_GPUS))
TARGET_TOTAL_BATCH_SIZE=393216

# Use floor division to ensure we don't exceed target
GRAD_ACCUM_STEPS=$((TARGET_TOTAL_BATCH_SIZE / WORLD_TOKENS_PER_STEP))
if [ $GRAD_ACCUM_STEPS -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi
TOTAL_BATCH_SIZE=$((GRAD_ACCUM_STEPS * WORLD_TOKENS_PER_STEP))

# Learning rates (tuned for 393216 batch size)
# These stay constant since we maintain consistent total batch size
MATRIX_LR=0.02
EMBEDDING_LR=0.2
UNEMBEDDING_LR=0.004

echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Max sequence length: $MAX_SEQ_LEN"
echo "  World tokens per step: $WORLD_TOKENS_PER_STEP"
echo "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "  Total batch size: $TOTAL_BATCH_SIZE (target: $TARGET_TOTAL_BATCH_SIZE)"
echo "  Matrix LR: $MATRIX_LR"
echo "  Embedding LR: $EMBEDDING_LR"
echo "  Unembedding LR: $UNEMBEDDING_LR"
echo ""

# Default intermediate artifacts directory
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# PyTorch/NCCL Environment Setup
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# NCCL basic settings
export NCCL_DEBUG=OFF                         # Disable verbose logging for performance
export NCCL_TIMEOUT=1800                      # 30 minute timeout for large operations
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # Async error handling
export TORCH_NCCL_BLOCKING_WAIT=1             # Better error detection
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

# NCCL P2P and communication settings
export NCCL_P2P_LEVEL=PHB                     # P2P for GPUs on same PCIe root complex
export NCCL_P2P_DISABLE=0                     # Enable P2P
export NCCL_P2P_NCHANNELS=5                   # P2P channels (GPU count - 1)
export NCCL_SOCKET_IFNAME=lo                  # Use loopback interface
export NCCL_IB_DISABLE=1                      # Disable InfiniBand
export NCCL_DMABUF_ENABLE=1                   # Enable DMA-BUF

# NCCL buffer and channel settings
export NCCL_BUFFSIZE=33554432                 # 32MB buffer size
export NCCL_MIN_NCHANNELS=16                  # Minimum channels
export NCCL_MAX_NCHANNELS=32                  # Maximum channels
export NCCL_NTHREADS=128                      # Number of threads

# NCCL retry settings
export NCCL_CONNECTION_RETRY=5
export NCCL_CONNECTION_RETRY_TIMEOUT=60

# Disable alternative collective algorithms (use standard RCCL)
export NCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export MSCCL_ENABLE=0

# -----------------------------------------------------------------------------
# ROCm-specific Optimizations
# -----------------------------------------------------------------------------

export HSA_FORCE_FINE_GRAIN_PCIE=1            # Better PCIe performance
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5       # Explicit GPU visibility
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5        # HIP-specific visibility
export HSA_ENABLE_INTERRUPT=1
export ROCR_ENABLE_PRE_VEGA_FINALIZATION=0
export HSA_XNACK=0                            # Disable XNACK for better P2P

# -----------------------------------------------------------------------------
# Training Execution
# -----------------------------------------------------------------------------

# Set up log file
LOG_FILE="training.log"
echo "Logging to: $LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..." | tee -a "$LOG_FILE"

# Build the command as an array to ensure accurate logging
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
  --run=$WANDB_RUN
)

# Log the full command
echo "Command: ${CMD[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Execute the command
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Training complete! To chat with your model, run:" | tee -a "$LOG_FILE"
echo "  python -m scripts.chat_web" | tee -a "$LOG_FILE"
echo "or for CLI:" | tee -a "$LOG_FILE"
echo "  python -m scripts.chat_cli -p \"Why is the sky blue?\"" | tee -a "$LOG_FILE"
