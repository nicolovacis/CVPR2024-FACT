#!/bin/bash

# CaiLab Dataset to FACT Format Preprocessing Script
# This script provides both a quickstart guide and executes the preprocessing

# ============================================================================
# DEFAULT PATHS - MODIFY THESE TO MATCH YOUR SYSTEM
# ============================================================================
DEFAULT_VIDEOS_DIR="/data-8tb/nvaci/dataset/dataset_CaiLab/split_videos"
DEFAULT_EXCEL_PATH="/data-8tb/nvaci/dataset/dataset_CaiLab/Cai_Lab_caspase_CQ_behavior.xlsx"
DEFAULT_OUTPUT_DIR="/data-8tb/nvaci/dataset/dataset_CaiLab/preprocessed"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_section() {
    echo -e "${CYAN}========================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================================================${NC}"
}

# Function to show usage/help
show_usage() {
    print_section "CaiLab Dataset Preprocessing for FACT"
    echo ""
    echo "Usage: $0 [videos_dir] [excel_path] [output_dir] [options]"
    echo ""
    echo -e "${BLUE}Arguments (all optional - will use defaults if not provided):${NC}"
    echo "  videos_dir   : Directory containing split MP4 video files"
    echo "                 Default: $DEFAULT_VIDEOS_DIR"
    echo "  excel_path   : Path to Excel annotation file"
    echo "                 Default: $DEFAULT_EXCEL_PATH"
    echo "  output_dir   : Output directory for preprocessed dataset"
    echo "                 Default: $DEFAULT_OUTPUT_DIR"
    echo ""
    echo -e "${BLUE}Optional arguments:${NC}"
    echo "  --target_fps RATE     : Target FPS for feature extraction (default: 25)"
    echo "  --clip_length N       : Number of frames per I3D clip (default: 16)"
    echo "  --train_ratio RATIO   : Ratio of training data (default: 0.8)"
    echo "  --n_splits N          : Number of train/test splits (default: 1)"
    echo "  --use_cpu             : Force CPU usage instead of GPU"
    echo "  --cpu_threads N       : Number of CPU threads (default: 4)"
    echo "  --i3d_weights PATH    : Path to I3D pretrained weights"
    echo "  --cut_videos          : Cut videos from trial start time (not yet implemented)"
    echo ""
    print_section "Quick Start Examples"
    echo ""
    echo -e "${BLUE}1. Simple run (using default paths):${NC}"
    echo "   ./preprocess_cailab.sh"
    echo ""
    echo -e "${BLUE}2. With custom paths:${NC}"
    echo "   ./preprocess_cailab.sh \\"
    echo "       /data-8tb/nvaci/dataset/dataset_CaiLab/split_videos \\"
    echo "       /data-8tb/nvaci/dataset/dataset_CaiLab/Cai_Lab_caspase_CQ_behavior.xlsx \\"
    echo "       /data-8tb/nvaci/dataset/dataset_CaiLab/preprocessed"
    echo ""
    echo -e "${BLUE}3. With custom options:${NC}"
    echo "   ./preprocess_cailab.sh \\"
    echo "       /data-8tb/nvaci/dataset/dataset_CaiLab/split_videos \\"
    echo "       /data-8tb/nvaci/dataset/dataset_CaiLab/Cai_Lab_caspase_CQ_behavior.xlsx \\"
    echo "       /data-8tb/nvaci/dataset/dataset_CaiLab/preprocessed \\"
    echo "       --target_fps 30 --train_ratio 0.8"
    echo ""
    echo -e "${BLUE}4. CPU-only mode (if no GPU):${NC}"
    echo "   ./preprocess_cailab.sh --use_cpu --cpu_threads 8"
    echo ""
    print_section "Expected Output Structure"
    echo ""
    echo "preprocessed/"
    echo "├── features/         # I3D features for each video (.npy files)"
    echo "│   ├── cKO-A_down.npy"
    echo "│   ├── cKO-A_front.npy"
    echo "│   └── ..."
    echo "├── groundTruth/      # Frame-level labels (.txt files)"
    echo "│   ├── cKO-A_down.txt"
    echo "│   ├── cKO-A_front.txt"
    echo "│   └── ..."
    echo "├── mapping.txt       # Action class mappings"
    echo "│   # 0 background"
    echo "│   # 1 scratching"
    echo "└── splits/           # Train/test splits"
    echo "    ├── train1.split"
    echo "    └── test1.split"
    echo ""
    print_section "Next Steps After Preprocessing"
    echo ""
    echo -e "${BLUE}1. Generate FACT config:${NC}"
    echo "   cd /path/to/FACT"
    echo "   python utils/gen_config.py \\"
    echo "       --dataset_path /data-8tb/nvaci/dataset/dataset_CaiLab/preprocessed \\"
    echo "       --dataset_name cailab \\"
    echo "       --output_config configs/cailab.yaml \\"
    echo "       --base_config configs/breakfast.yaml"
    echo ""
    echo -e "${BLUE}2. Train the model:${NC}"
    echo "   python train.py --config configs/cailab.yaml"
    echo ""
    echo -e "${BLUE}3. Evaluate the model:${NC}"
    echo "   python eval.py --config configs/cailab.yaml --checkpoint /path/to/checkpoint.pth"
    echo ""
    print_section "Troubleshooting"
    echo ""
    echo -e "${YELLOW}Issue: No videos found${NC}"
    echo "  - Verify videos directory path is correct"
    echo "  - Ensure videos have .mp4 extension"
    echo "  - Check filename format: YYYY-MM-DD_CQ_mouseX_perspective.mp4"
    echo ""
    echo -e "${YELLOW}Issue: Excel file not found${NC}"
    echo "  - Check Excel file path is correct"
    echo "  - Ensure file has .xlsx extension"
    echo ""
    echo -e "${YELLOW}Issue: Missing Python packages${NC}"
    echo "  - Script will prompt to install automatically"
    echo "  - Or manually: pip install --break-system-packages torch torchvision numpy pandas opencv-python tqdm openpyxl"
    echo ""
    echo -e "${YELLOW}Issue: CUDA out of memory${NC}"
    echo "  - Use --use_cpu flag to force CPU usage"
    echo "  - Or increase --cpu_threads value"
    echo ""
    echo -e "${YELLOW}Issue: Preprocessing is slow${NC}"
    echo "  - GPU is much faster than CPU (check with: nvidia-smi)"
    echo "  - Increase CPU threads if using CPU"
    echo ""
    print_section "Important Notes"
    echo ""
    echo "• Videos should already be cut to the correct timeframe"
    echo "• Default behavior class is 'scratching'"
    echo "• Video naming: 2025-08-27_CQ_mouseX_perspective.mp4"
    echo "• Animals: mouseA → cKO-A, mouseC → cKO-C, etc."
    echo "• Perspectives: down, front"
    echo ""
    echo "For more details, see README_CAILAB.md and EXCEL_STRUCTURE_NOTES.md"
    echo ""
    exit 1
}

# Check if help is requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
fi

# Parse arguments - use defaults if not provided
VIDEOS_DIR="$DEFAULT_VIDEOS_DIR"
EXCEL_PATH="$DEFAULT_EXCEL_PATH"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
EXTRA_ARGS=()

# Check if first argument is a path or an option
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    VIDEOS_DIR="$1"
    shift
    
    # Check for second argument (excel_path)
    if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
        EXCEL_PATH="$1"
        shift
        
        # Check for third argument (output_dir)
        if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
            OUTPUT_DIR="$1"
            shift
        fi
    fi
fi

# Remaining arguments are options
EXTRA_ARGS=("$@")

print_section "CaiLab Dataset Preprocessing - Starting"
echo ""
print_info "Using paths:"
print_info "  Videos directory: $VIDEOS_DIR"
print_info "  Excel annotations: $EXCEL_PATH"
print_info "  Output directory: $OUTPUT_DIR"
echo ""

# Check if input paths exist
if [ ! -d "$VIDEOS_DIR" ]; then
    print_error "Videos directory does not exist: $VIDEOS_DIR"
    exit 1
fi

if [ ! -f "$EXCEL_PATH" ]; then
    print_error "Excel file does not exist: $EXCEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count video files
VIDEO_COUNT=$(ls -1 "$VIDEOS_DIR"/*.mp4 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    print_error "No MP4 files found in $VIDEOS_DIR"
    exit 1
fi
print_info "Found $VIDEO_COUNT video files to process"

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 is not installed or not in PATH"
    exit 1
fi

print_info "Python version: $(python3 --version)"

# Check for required Python packages
print_info "Checking required Python packages..."
REQUIRED_PACKAGES=("torch" "torchvision" "numpy" "pandas" "opencv-python" "tqdm" "openpyxl")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    pkg_import="${package//-/_}"
    if [ "$package" = "opencv-python" ]; then
        pkg_import="cv2"
    fi
    if ! python3 -c "import ${pkg_import}" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_warning "Missing Python packages: ${MISSING_PACKAGES[*]}"
    
    # Check if running in interactive mode
    if [ -t 0 ]; then
        # Interactive mode - ask user
        read -p "Do you want to install them now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installing missing packages..."
            pip install --break-system-packages "${MISSING_PACKAGES[@]}"
        else
            print_error "Required packages not installed. Exiting."
            exit 1
        fi
    else
        # Non-interactive mode (nohup, background job, etc.) - auto install
        print_info "Running in non-interactive mode. Auto-installing missing packages..."
        pip install --break-system-packages "${MISSING_PACKAGES[@]}"
        
        if [ $? -ne 0 ]; then
            print_error "Failed to install packages. Please install manually:"
            print_error "pip install --break-system-packages ${MISSING_PACKAGES[*]}"
            exit 1
        fi
        print_info "Packages installed successfully!"
    fi
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "No GPU detected. Processing will use CPU (slower)."
fi

# Check for ffmpeg (needed for video cutting)
if ! command -v ffmpeg &> /dev/null; then
    print_warning "ffmpeg not found. Video cutting (--cut_videos) will not work."
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if preprocessing script exists
PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess_cailab_dataset.py"
if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    print_error "Preprocessing script not found: $PREPROCESS_SCRIPT"
    exit 1
fi

# Check for I3D model file
I3D_MODEL="$SCRIPT_DIR/i3d_model.py"
if [ ! -f "$I3D_MODEL" ]; then
    print_warning "i3d_model.py not found. Will use fallback feature extractor (R3D-18)."
    print_warning "For best results, copy i3d_model.py from another preprocessing folder."
fi

# Run preprocessing
print_section "Running Preprocessing Pipeline"
echo ""

python3 "$PREPROCESS_SCRIPT" \
    --videos_dir "$VIDEOS_DIR" \
    --excel_path "$EXCEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
print_section "Preprocessing Complete"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    print_info "✓ Preprocessing completed successfully!"
    echo ""
    
    print_section "Output Summary"
    echo ""
    print_info "Output directory: $OUTPUT_DIR"
    
    # Count output files
    FEATURES_COUNT=$(ls -1 "$OUTPUT_DIR"/features/*.npy 2>/dev/null | wc -l)
    GT_COUNT=$(ls -1 "$OUTPUT_DIR"/groundTruth/*.txt 2>/dev/null | wc -l)
    
    echo "  • Features extracted:  $FEATURES_COUNT .npy files"
    echo "  • Labels created:      $GT_COUNT .txt files"
    echo "  • Mapping file:        mapping.txt"
    echo "  • Train/test splits:   splits/"
    echo ""
    
    print_section "Next Steps"
    echo ""
    echo -e "${BLUE}Step 1: Generate FACT configuration${NC}"
    echo "  cd /path/to/FACT"
    echo "  python utils/gen_config.py \\"
    echo "      --dataset_path $OUTPUT_DIR \\"
    echo "      --dataset_name cailab \\"
    echo "      --output_config configs/cailab.yaml \\"
    echo "      --base_config configs/breakfast.yaml"
    echo ""
    echo -e "${BLUE}Step 2: Train the model${NC}"
    echo "  python train.py --config configs/cailab.yaml"
    echo ""
    echo -e "${BLUE}Step 3: Evaluate the model${NC}"
    echo "  python eval.py --config configs/cailab.yaml --checkpoint /path/to/checkpoint.pth"
    echo ""
    
    print_section "Dataset Information"
    echo ""
    echo "• Behaviors: background (0), scratching (1)"
    echo "• Animals: mouseA-G (cKO-A to cKO-G)"
    echo "• Perspectives: down, front"
    echo "• Feature extraction: I3D or R3D-18"
    echo "• Target FPS: Check your arguments (default: 25)"
    echo ""
    
else
    print_error "✗ Preprocessing failed with exit code $EXIT_CODE"
    echo ""
    print_info "Check the logs above for error details"
    print_info "Common issues:"
    echo "  • Excel column structure doesn't match assumptions"
    echo "  • Video files have unexpected naming format"
    echo "  • Missing dependencies or CUDA errors"
    echo ""
    print_info "For help, check README_CAILAB.md and EXCEL_STRUCTURE_NOTES.md"
    exit $EXIT_CODE
fi