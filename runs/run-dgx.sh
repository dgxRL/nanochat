#!/bin/bash

# Run on DGX Spark (1xH100 GPU, 128GB shared RAM)
# This script is tuned for DGX Spark platform
# Last updated/tuned on Jan 19, 2026

# Run as:
# bash runs/run-dgx.sh

# DGX Spark: 1xH100 GPU, 128GB shared RAM
# Expect ~30-60 minutes total runtime for full pipeline

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
echo "Installing GPU dependencies (may take a few minutes on first run)..."
timeout 6000 bash -c "uv sync --extra gpu" 2>&1 > /dev/null
if [ $? -eq 124 ]; then
    echo "GPU dependency installation timed out, falling back to CPU dependencies..."
    uv sync --extra cpu
elif [ $? -ne 0 ]; then
    echo "GPU dependency installation failed, falling back to CPU dependencies..."
    uv sync --extra cpu
else
    echo "GPU dependencies installed successfully"
fi

# train tokenizer on ~2B characters
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
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

# train base model on DGX Spark
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --device-type=$DEVICE_TYPE \
    --run=$WANDB_RUN

# evaluate base model
python -m scripts.base_loss --device-batch-size=1 --split-tokens=16384 --device-type=$DEVICE_TYPE
python -m scripts.base_eval --max-per-task=16

# midtraining with identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.mid_train \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -i mid -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web -i mid
