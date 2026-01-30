#!/bin/bash

# Stanford Dataset Binary Classification (Balanced) Preprocessing Script
# Converts to binary classification (background vs licking) and balances classes

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
    echo "  --i3d_weights PATH    : Path to I3D weights file (.pt)"
    echo "  --target_fps RATE     : Target FPS for feature extraction (default: 25)"
    echo "  --train_ratio RATIO   : Ratio of training data (default: 0.8)"
    echo "  --n_splits N          : Number of train/test splits (default: 1)"
    echo "  --balance_seed SEED   : Random seed for balancing (default: 42)"
    echo "  --use_cpu             : Force CPU usage instead of GPU"
    echo ""
    echo "Features:"
    echo "  - Binary classification: background vs licking"
    echo "  - Removes: shaking, licking_shaking classes"
    echo "  - Balances: downsamples background to match licking frames"
    echo ""
    echo "Example:"
    echo "  $0 /data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut \\"
    echo "     /data-8tb/nvaci/dataset/dataset_Stanford/raw/labels \\"
    echo "     /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class \\"
    echo "     --i3d_weights /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/pretrained_models/i3d_rgb_kinetics.pt"
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

print_info "Starting Stanford dataset preprocessing (binary + balanced)..."
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
REQUIRED_IMPORTS=("torch" "torchvision" "numpy" "pandas" "cv2" "tqdm")
MISSING_PACKAGES=()

for i in "${!REQUIRED_IMPORTS[@]}"; do
    if ! python3 -c "import ${REQUIRED_IMPORTS[$i]}" 2>/dev/null; then
        MISSING_PACKAGES+=("${REQUIRED_PACKAGES[$i]}")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_warning "Missing Python packages: ${MISSING_PACKAGES[*]}"
    print_info "Installing missing packages automatically..."
    pip install --break-system-packages "${MISSING_PACKAGES[@]}"
    if [ $? -ne 0 ]; then
        print_error "Failed to install packages. Exiting."
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
PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess_stanford_binary_balanced.py"
if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    print_error "Preprocessing script not found: $PREPROCESS_SCRIPT"
    exit 1
fi

# Check if i3d_model.py exists (needed if using I3D weights)
I3D_MODEL_SCRIPT="$SCRIPT_DIR/i3d_model.py"
if [ ! -f "$I3D_MODEL_SCRIPT" ]; then
    print_warning "i3d_model.py not found in script directory."
    print_warning "If you're using --i3d_weights, make sure i3d_model.py is in: $SCRIPT_DIR"
    print_warning "Otherwise, the script will use R3D-18 as fallback."
fi

# Run preprocessing
print_info "Running preprocessing pipeline..."
echo ""
echo "======================================================================"

python3 -u "$PREPROCESS_SCRIPT" \
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
    print_info "Dataset details:"
    echo "  - Binary classification: background vs licking"
    echo "  - Balanced: 1:1 ratio"
    echo "  - Removed: shaking and licking_shaking classes"
    echo ""
    print_info "Next steps:"
    echo "  1. Verify the output in: $OUTPUT_DIR"
    echo ""
    echo "  2. Generate FACT config file:"
    echo "     cd /path/to/FACT"
    echo "     python utils/gen_config.py \\"
    echo "         --dataset_path $OUTPUT_DIR \\"
    echo "         --dataset_name stanford_binary \\"
    echo "         --output_config configs/stanford_binary.yaml \\"
    echo "         --base_config configs/breakfast.yaml"
    echo ""
    echo "  3. Update the config to have num_classes: 2"
    echo ""
    echo "  4. Train the model:"
    echo "     python train.py --config configs/stanford_binary.yaml"
else
    print_error "Preprocessing failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi