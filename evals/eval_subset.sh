#!/bin/bash
set -e

dataset_name=(
  "gsm8k"
  "mbpp"
)

for dataset in "${dataset_name[@]}"; do
  echo "Evaluating on ${dataset}..."

  if [ "$dataset" = "gsm8k" ]; then
    num_shots=8
  else
    num_shots=0
  fi

  uv run olmes \
    --model "${EVAL_MODEL_PATH}" \
    --task "${dataset}" \
    --output-dir "${EVAL_OUTPUT_ROOT}/${dataset}" \
    --num-shots "${num_shots}" \
    --use-chat-format 1
done