#!/bin/bash

# Ensure the script exits if any command fails
set -e

echo "Starting SFT Pipeline..."
python train_sft.py --config configs/baseline_1500steps_128batch_.00049lr_samples_full_dsretrain.yaml