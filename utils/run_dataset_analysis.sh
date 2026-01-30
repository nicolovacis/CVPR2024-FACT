#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class"
OUT_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method"   # or anywhere you want
DATASET_NAME="stanford_binary_2class"
FPS="1"

python3 analyze_dataset_distribution.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --fps "$FPS"
