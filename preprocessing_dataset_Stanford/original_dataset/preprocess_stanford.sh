#!/bin/bash

# Stanford Dataset to FACT Format Preprocessing Script
# This script extracts I3D features and converts labels to FACT-compatible format

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <videos_dir> <labels_dir> <output_dir> [options]"
    echo ""
    echo "Required arguments:"
    echo "  videos_dir   : Path to directory containing MP4 video files"
    echo "  labels_dir   : Path to directory containing CSV label files"
    echo "  output_dir   : Path to output directory for preprocessed dataset"
    echo ""
    echo "Optional arguments:"
    echo "  --target_fps RATE     : Target FPS for feature extraction (default: 25)"
    echo "  --train_ratio RATIO   : Ratio of training data (default: 0.8)"
    echo "  --n_splits N          : Number of train/test splits (default: 1)"
    echo "  --use_cpu             : Force CPU usage instead of GPU"
    echo ""
    echo "Example:"
    echo "  $0 /data/videos_cut /data/labels /data/preprocessed --target_fps 25"
    exit 1
fi

VIDEOS_DIR="$1"
LABELS_DIR="$2"
OUTPUT_DIR="$3"
shift 3

# Check if input directories exist
if [ ! -d "$VIDEOS_DIR" ]; then
    print_error "Videos directory does not exist: $VIDEOS_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    print_error "Labels directory does not exist: $LABELS_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

print_info "Starting Stanford dataset preprocessing..."
print_info "Videos directory: $VIDEOS_DIR"
print_info "Labels directory: $LABELS_DIR"
print_info "Output directory: $OUTPUT_DIR"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 is not installed or not in PATH"
    exit 1
fi

print_info "Python version: $(python3 --version)"

# Check for required Python packages
print_info "Checking required Python packages..."
REQUIRED_PACKAGES=("torch" "torchvision" "numpy" "pandas" "opencv-python" "tqdm")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import ${package//-/_}" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_warning "Missing Python packages: ${MISSING_PACKAGES[*]}"
    read -p "Do you want to install them now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing missing packages..."
        pip install --break-system-packages "${MISSING_PACKAGES[@]}"
    else
        print_error "Required packages not installed. Exiting."
        exit 1
    fi
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "No GPU detected. Processing will use CPU (slower)."
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if preprocessing script exists
PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess_stanford_dataset.py"
if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    print_error "Preprocessing script not found: $PREPROCESS_SCRIPT"
    exit 1
fi

# Run preprocessing
print_info "Running preprocessing pipeline..."
echo ""
echo "======================================================================"

python3 "$PREPROCESS_SCRIPT" \
    --videos_dir "$VIDEOS_DIR" \
    --labels_dir "$LABELS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

EXIT_CODE=$?

echo "======================================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    print_info "Preprocessing completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Verify the output in: $OUTPUT_DIR"
    echo "  2. Generate FACT config file:"
    echo "     python utils/gen_config.py \\"
    echo "         --dataset_path $OUTPUT_DIR \\"
    echo "         --dataset_name stanford \\"
    echo "         --output_config configs/stanford.yaml \\"
    echo "         --base_config configs/breakfast.yaml"
    echo ""
    echo "  3. Train the model:"
    echo "     python train.py --config configs/stanford.yaml"
else
    print_error "Preprocessing failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
