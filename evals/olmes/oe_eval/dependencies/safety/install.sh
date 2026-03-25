#!/bin/bash
# set -ex
# rm -rf .venv
# # rm -rf safety-eval


# uv venv --python 3.11
git clone https://github.com/owos/safety-eval.git safety-eval

uv pip install -e safety-eval 
uv pip install -r safety-eval/requirements.txt
export CUDA_VISIBLE_DEVICES=0