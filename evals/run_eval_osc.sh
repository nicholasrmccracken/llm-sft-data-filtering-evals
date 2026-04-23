#!/bin/bash
#SBATCH --job-name=eval-5525
#SBATCH --account=PAS3272
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=eval_logs/eval_%j.out
#SBATCH --error=eval_logs/eval_%j.err

set -e

# Load .env from repo root
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "Missing .env file"
  exit 1
fi

# Validate required env vars
if [ -z "${PROJECT_ROOT}" ] || [ -z "${EVAL_MODEL_PATH}" ] || [ -z "${EVAL_OUTPUT_ROOT}" ]; then
  echo "Missing one of: PROJECT_ROOT, EVAL_MODEL_PATH, EVAL_OUTPUT_ROOT in .env"
  exit 1
fi

cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p "${EVAL_OUTPUT_ROOT}"

# OSC environment
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache
export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv
export UV_LINK_MODE=copy
export OPENAI_API_KEY="sk-dummy-not-used"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Install safety dependencies for safety-eval
cd "${PROJECT_ROOT}/evals/olmes/oe_eval/dependencies/safety"
bash install.sh || true

dataset_name=(
  "ifeval"
  "harmbench::default"
  "xstest::default"
)

cd "${PROJECT_ROOT}"
mkdir -p "${EVAL_OUTPUT_ROOT}"

# Set num-shots to 8 only for gsm8k, otherwise set to 0
for dataset in "${dataset_name[@]}"; do
  echo "Evaluating on ${dataset}..."
  if [ "$dataset" = "gsm8k" ]; then
    num_shots=8
  else
    num_shots=0
  fi

  safe_dataset_name=$(echo "${dataset}" | sed 's/::/_/g')

  uv run olmes \
    --model "${EVAL_MODEL_PATH}" \
    --task "${dataset}" \
    --output-dir "${EVAL_OUTPUT_ROOT}/${safe_dataset_name}" \
    --num-shots "${num_shots}" \
    --use-chat-format 1
done