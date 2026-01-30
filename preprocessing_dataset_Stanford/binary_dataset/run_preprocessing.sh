#!/bin/bash

# Simple run script for binary balanced preprocessing
# All paths are hardcoded - just run: ./run_preprocessing.sh

# Define paths
VIDEOS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut"
LABELS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/labels"
OUTPUT_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class"
I3D_WEIGHTS="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/pretrained_models/i3d_rgb_kinetics.pt"

# Run preprocessing
./preprocess_stanford_binary_balanced.sh \
    "$VIDEOS_DIR" \
    "$LABELS_DIR" \
    "$OUTPUT_DIR" \
    --i3d_weights "$I3D_WEIGHTS" \
    --target_fps 25 \
    --train_ratio 0.8 \
    --n_splits 1 \
    --balance_seed 42
