#!/bin/bash
# Training script for Anomaly-Transformer on Ballistic Dataset
# Usage: bash scripts/ballistic.sh

# Default: measurements mode
python train_ballistic.py \
    --data_folder ./dataset/ballistic \
    --feature_mode measurements \
    --win_size 100 \
    --num_epochs 10 \
    --batch_size 256 \
    --lr 1e-4 \
    --k 3

# Run inference
python infer_ballistic.py \
    --data_folder ./dataset/ballistic \
    --feature_mode measurements \
    --compute_threshold
