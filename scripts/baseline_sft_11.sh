#!/bin/bash

# Ensure the script exits if any command fails
set -e

echo "Starting SFT Pipeline..."
python train_sft.py --config configs/baseline_filtered_length.yaml