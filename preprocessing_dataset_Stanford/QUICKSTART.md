# Stanford Dataset to FACT Format - Quick Start Guide

## Files Overview

This package contains all necessary files to preprocess the Stanford animal behavior dataset for the FACT model:

1. **preprocess_stanford_dataset.py** - Main preprocessing script
2. **preprocess_stanford.sh** - Bash wrapper for easy execution
3. **i3d_model.py** - I3D model implementation
4. **download_i3d_weights.py** - Download pretrained I3D weights
5. **run_preprocessing_example.sh** - Complete example workflow
6. **requirements_preprocessing.txt** - Python dependencies
7. **README_PREPROCESSING.md** - Comprehensive documentation

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements_preprocessing.txt
```

### Step 2: Run Preprocessing

**Option A: Simple (uses fallback R3D-18 model)**
```bash
./preprocess_stanford.sh \
    /data-8tb/nvaci/dataset_Stanford/videos_cut \
    /data-8tb/nvaci/dataset_Stanford/labels \
    /output/path/stanford_preprocessed
```

**Option B: With I3D (Recommended)**
```bash
# Download I3D weights first
python download_i3d_weights.py

# Run with I3D
python preprocess_stanford_dataset.py \
    --videos_dir /data-8tb/nvaci/dataset_Stanford/videos_cut \
    --labels_dir /data-8tb/nvaci/dataset_Stanford/labels \
    --output_dir /output/path/stanford_preprocessed \
    --i3d_weights ./pretrained_models/i3d_rgb_kinetics.pt
```

### Step 3: Use with FACT

```bash
cd /path/to/CVPR2024-FACT-main

# Generate config
python utils/gen_config.py \
    --dataset_path /output/path/stanford_preprocessed \
    --dataset_name stanford \
    --output_config configs/stanford.yaml \
    --base_config configs/breakfast.yaml

# Train
python train.py --config configs/stanford.yaml
```

## Expected Output Structure

```
stanford_preprocessed/
├── features/              # 32 .npy files (I3D features, ~100MB each)
├── groundTruth/          # 32 .txt files (frame-level labels)
├── mapping.txt           # Action class mappings (3 classes)
└── splits/               # Train/test splits
    ├── train.split1.bundle
    └── test.split1.bundle
```

## Common Options

```bash
--target_fps 25           # FPS for feature extraction
--train_ratio 0.8         # 80% train, 20% test
--n_splits 3              # Create 3 different splits
--use_cpu                 # Force CPU (if no GPU available)
```

## Processing Time

- **With GPU**: ~30-60 minutes for 32 videos
- **With CPU**: ~2.5-5 hours for 32 videos

## Troubleshooting

**GPU out of memory?**
```bash
python preprocess_stanford_dataset.py ... --use_cpu
```

**Missing dependencies?**
```bash
pip install torch torchvision numpy pandas opencv-python tqdm
```

**Need help?**
Read the full documentation in `README_PREPROCESSING.md`

## Action Classes

The preprocessor handles 3 action classes:
- `0` - background (default)
- `1` - licking
- `2` - shaking

Priority when multiple actions occur: licking > shaking > background

## Complete Example

See `run_preprocessing_example.sh` for a complete workflow example that you can customize for your system.

## Hardware Requirements

- **Minimum**: 8GB RAM, 50GB storage
- **Recommended**: CUDA GPU, 16GB RAM, 100GB storage

---

For detailed documentation, see `README_PREPROCESSING.md`
