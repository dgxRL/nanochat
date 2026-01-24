#!/bin/bash

# Run on DGX Spark (1xH100 GPU, 128GB shared RAM)
# This script is tuned for DGX Spark platform
# Last updated/tuned on Jan 19, 2026

# Run as:
# bash runs/run-dgx.sh

# DGX Spark: 1xH100 GPU, 128GB shared RAM
# Expect ~30-60 minutes total runtime for full pipeline

# Base unit for batch sizes - change this to scale everything
# b32  ->         , 500 sec
# b64  ->         , 1000 sec 
# b128 -> 27GB MEM, 2000 sec to finash
# b256 -> 42GB MEM, 3690 sec
# b512 -> 71GB MEM, 
# -------------
# b32_d12 -> 26.2GB, 35.7 minute

# b32_d20 -> 42.7GB, 131 minutes

# b32_d6_H128 -> 21.8GB, 9 minutes

# base: batch=32, seq_len = 512, depth = 6
BASE_BATCH_SIZE=32
MAX_SEQ_LENGTH=512 # default 512
DEPTH=6 # default 6
HEAD_DIM=64

GRAD_ACCUM_STEPS=2

# Calculate derived values as multiples of BASE_BATCH_SIZE
DEVICE_BATCH_SIZE=$BASE_BATCH_SIZE
TOTAL_BATCH_SIZE=$(($BASE_BATCH_SIZE * $MAX_SEQ_LENGTH * $GRAD_ACCUM_STEPS))
SPLIT_TOKENS=$TOTAL_BATCH_SIZE

# all setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Install GPU dependencies with timeout handling
uv sync --extra gpu

# train tokenizer on ~2B characters
# python -m nanochat.dataset -n 8
# python -m scripts.tok_train --max-chars=2000000000 # b64 -> 163M => max b8x64 = b512
# python -m scripts.tok_eval

# Detect device type
CUDA_AVAILABLE=$(python -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA_AVAILABLE" = "1" ]; then
    DEVICE_TYPE="cuda"
    echo "Training on GPU (CUDA)"
else
    DEVICE_TYPE="cpu"
    echo "Training on CPU (CUDA not available)"
fi

echo "Batch sizes (derived from BASE_BATCH_SIZE=$BASE_BATCH_SIZE):"
echo "  --device-batch-size = $DEVICE_BATCH_SIZE"
echo "  --total-batch-size = $TOTAL_BATCH_SIZE"
echo "  --split-tokens = $SPLIT_TOKENS"
echo "  --grad-accum-steps = $GRAD_ACCUM_STEPS"

# train base model on DGX Spark
# long goal 21400, loss: 2.727
# python -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
python -m scripts.base_train \
    --depth=$DEPTH \
    --head-dim=$HEAD_DIM \
    --window-pattern=L \
    --max-seq-len=$MAX_SEQ_LENGTH \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --eval-every=100 \
    --eval-tokens=$SPLIT_TOKENS \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=500 \
    --device-type=$DEVICE_TYPE \
    --run=$WANDB_RUN

