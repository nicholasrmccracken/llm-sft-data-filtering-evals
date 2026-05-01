#!/bin/bash
set -e

echo "Starting reproduction run..."

# Load environment
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "Missing .env file (use .env.example)"
  exit 1
fi

# Validate required vars
if [ -z "${PROJECT_ROOT}" ] || [ -z "${EVAL_MODEL_PATH}" ] || [ -z "${EVAL_OUTPUT_ROOT}" ]; then
  echo "Missing required env variables"
  exit 1
fi

cd "${PROJECT_ROOT}"

# Activate venv
source .venv/bin/activate

# Create folders
mkdir -p reproduction_logs
mkdir -p "${EVAL_OUTPUT_ROOT}/reproduction"

# OSC / cluster-safe settings
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/tmp/${USER}/triton_cache
export UV_CACHE_DIR=/tmp/${USER}/.cache/uv
export OPENAI_API_KEY="sk-dummy-not-used"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "Step 1: Running small SFT training..."
python train_sft.py --config configs/reproduce_sft.yaml \
  > reproduction_logs/train_sft.out \
  2> reproduction_logs/train_sft.err

echo "Step 2: Merging LoRA weights..."
python merge_models.py \
  > reproduction_logs/merge.out \
  2> reproduction_logs/merge.err

echo "Step 3: Installing safety eval dependencies..."
cd "${PROJECT_ROOT}/evals/olmes/oe_eval/dependencies/safety"
bash install.sh || true

cd "${PROJECT_ROOT}"

echo "Step 4: Running evaluation subset..."
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/reproduction" \
bash evals/eval_subset.sh \
  > reproduction_logs/eval_subset.out \
  2> reproduction_logs/eval_subset.err

echo "Reproduction complete."