#!/bin/bash

# Run script for multi-label preprocessing
# Usage: ./run_preprocessing_multilabel_fps.sh

python3 preprocess_stanford_multilabel_original_fps.py \
    --input_videos /data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut \
    --input_labels /data-8tb/nvaci/dataset/dataset_Stanford/raw/labels \
    --output_dir /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS \
    --i3d_weights ../rgb_imagenet.pt \
    --aggregation mean \
    --train_ratio 0.78125 \
    --n_splits 1 \
    --seed 42 \
    --cpu_threads 4
