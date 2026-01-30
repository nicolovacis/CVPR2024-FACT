#!/bin/bash
# Simple run script for binary balanced preprocessing with original FPS
# All paths are hardcoded - just run: ./run_preprocessing_original_fps.sh

# Define paths
VIDEOS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut"
LABELS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/labels"
OUTPUT_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS"
I3D_WEIGHTS="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/pretrained_models/i3d_rgb_kinetics.pt"

# Configuration
AGGREGATION="mean"  # Options: 'mean' or 'max'
TRAIN_RATIO=0.8
N_SPLITS=5
BALANCE_SEED=42
CPU_THREADS=4
USE_CPU=""  # Set to "--use_cpu" to force CPU usage

# Create log file
LOG_FILE="${OUTPUT_DIR}/preprocessing_original_fps_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "======================================================================"
echo "STANFORD BINARY DATASET PREPROCESSING (ORIGINAL FPS)"
echo "======================================================================"
echo "Started at: $(date)"
echo ""
echo "Configuration:"
echo "  Input videos: $VIDEOS_DIR"
echo "  Input labels: $LABELS_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  I3D weights:  $I3D_WEIGHTS"
echo "  Clip length:  AUTO (detected from video FPS)"
echo "  Aggregation:  $AGGREGATION"
echo "  Train ratio:  $TRAIN_RATIO"
echo "  Num splits:   $N_SPLITS"
echo "  Seed:         $BALANCE_SEED"
echo "  CPU threads:  $CPU_THREADS"
echo "  Use CPU:      $USE_CPU"
echo ""
echo "Log file: $LOG_FILE"
echo "======================================================================"
echo ""

# Build command
CMD="python preprocess_stanford_binary_original_fps.py \
    --input_videos \"$VIDEOS_DIR\" \
    --input_labels \"$LABELS_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --i3d_weights \"$I3D_WEIGHTS\" \
    --aggregation $AGGREGATION \
    --train_ratio $TRAIN_RATIO \
    --n_splits $N_SPLITS \
    --balance_seed $BALANCE_SEED \
    --cpu_threads $CPU_THREADS \
    $USE_CPU"

echo "Running command:"
echo "$CMD"
echo ""

# Run the preprocessing and save output to log
eval "$CMD" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Preprocessing completed successfully!"
else
    echo "Preprocessing failed with exit code: $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "======================================================================"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"

exit $EXIT_CODE
