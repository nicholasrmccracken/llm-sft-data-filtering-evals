#!/bin/bash

#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache


export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv  #control your uv caches

# Dummy key to prevent import error in safety-eval (WildGuard doesn't actually use it)
export OPENAI_API_KEY="sk-dummy-not-used"

# Disable vLLM V1 multiprocessing so EngineCore runs inline in the spawned subprocess
# rather than forking a grandchild process that loses CUDA visibility on SLURM
export VLLM_ENABLE_V1_MULTIPROCESSING=0


cd olmes/oe_eval/dependencies/safety
bash install.sh


dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"

)
# Point this to the output_path you used in your merge_lora.py script
PROJECT_ROOT="/users/PAS3272/chawla114/cse5525-final"

# 2. Point to the model inside that root
model_path="${PROJECT_ROOT}/trained_models/Llama-3.2-1B-SFT-Merged-Baseline2-11"
# A clean name for your results folder

for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."

    uv run olmes \
        --model ${model_path} \
        --task ${dataset} \
        --output-dir $model_path-eval-${dataset} 
done