#!/bin/bash

# Run on DGX Spark (1xH100 GPU, 128GB shared RAM)
# This script is tuned for DGX Spark platform
# Last updated/tuned on Jan 19, 2026

# Run as:
# bash runs/run-dgx.sh

# DGX Spark: 1xH100 GPU, 128GB shared RAM
# Expect ~30-60 minutes total runtime for full pipeline

# Base unit for batch sizes - change this to scale everything
BASE_BATCH_SIZE=512

# Calculate derived values as multiples of BASE_BATCH_SIZE
DEVICE_BATCH_SIZE=$BASE_BATCH_SIZE
TOTAL_BATCH_SIZE=$(($BASE_BATCH_SIZE * 512))
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
python -m nanochat.dataset -n 32
python -m scripts.tok_train --max-chars=8_000_000_000
python -m scripts.tok_eval

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

# train base model on DGX Spark
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --eval-every=100 \
    --eval-tokens=$SPLIT_TOKENS \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --device-type=$DEVICE_TYPE \
    --run=$WANDB_RUN

# evaluate base model
python -m scripts.base_loss --device-batch-size=2 --split-tokens=$SPLIT_TOKENS --device-type=$DEVICE_TYPE
python -m scripts.base_eval --max-per-task=16

# midtraining with identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.mid_train \
    --max-seq-len=512 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --eval-every=200 \
    --eval-tokens=$SPLIT_TOKENS \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -i mid -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web -i mid
