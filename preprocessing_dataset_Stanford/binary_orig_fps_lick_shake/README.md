# Multi-Label Preprocessing for Stanford Dataset

## Overview

This folder contains preprocessing scripts for **multi-label** temporal action segmentation where licking and shaking are treated as **independent binary labels** rather than mutually exclusive classes.

## Key Differences from Binary Classification

### Binary Classification (binary_orig_fps):
- **Output**: Single categorical label per second
- **Classes**: background (0) or licking (1)
- **Method**: Softmax over classes
- **Balancing**: Applied (downsample background to match licking)
- **Removes**: All frames with shaking

### Multi-Label Classification (binary_orig_fps_lick_shake):
- **Output**: Two independent binary labels per second
- **Labels**: licking (0/1) AND shaking (0/1)
- **Method**: Two-head sigmoid outputs
- **Balancing**: NOT applied (keeps dataset unbalanced)
- **Keeps**: All frames including those with both licking and shaking

## Label Format

Each second gets 2 independent binary values:
```
second_0: [licking=0, shaking=0]  # background
second_1: [licking=1, shaking=0]  # only licking
second_2: [licking=0, shaking=1]  # only shaking
second_3: [licking=1, shaking=1]  # both licking AND shaking
```

## Output Format

- **Features**: Saved as `.npy` files with shape `(n_seconds, 1024)`
- **Labels**: Saved as `.npy` files with shape `(n_seconds, 2)`
  - Column 0: licking (0 or 1)
  - Column 1: shaking (0 or 1)

## Running the Preprocessing

```bash
cd /home/nvaci/FACT/preprocessing_dataset_Stanford/binary_orig_fps_lick_shake

# Run directly
./run_preprocessing_multilabel_fps.sh

# Or with nohup (recommended for long runs)
nohup python3 preprocess_stanford_multilabel_original_fps.py \
    --input_videos /data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut \
    --input_labels /data-8tb/nvaci/dataset/dataset_Stanford/raw/labels \
    --output_dir /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS \
    --i3d_weights ../rgb_imagenet.pt \
    --aggregation mean \
    --train_ratio 0.78125 \
    --n_splits 1 \
    --seed 42 \
    --cpu_threads 4 \
    > preprocessing_multilabel.log 2>&1 &
```

## Check Progress

```bash
# Check if running
ps aux | grep "preprocess_stanford_multilabel" | grep -v grep

# View log
tail -f preprocessing_multilabel.log

# Check output directory
ls /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS/
```

## Output Structure

```
preprocessed_data_multilabel_originalFPS/
├── features/           # .npy files with shape (n_seconds, 1024)
├── groundTruth/        # .npy files with shape (n_seconds, 2) - [licking, shaking]
├── mapping.txt         # Label definitions
└── splits/             # Train/test splits
    ├── train.split1.bundle
    └── test.split1.bundle
```

## Next Steps for Model Training

To use this multi-label data with FACT, you will need to:

1. **Modify the model** (`models/blocks.py`):
   - Replace single softmax head with two sigmoid heads
   - Output shape: `[batch, time, 2]` instead of `[batch, time, num_classes]`

2. **Modify the loss** (`models/loss.py`):
   - Use Binary Cross Entropy (BCE) instead of Cross Entropy
   - Compute loss separately for each label

3. **Modify the dataset loader** (`utils/dataset.py`):
   - Load `.npy` label files instead of `.txt`
   - Handle 2D label arrays

4. **Create new config**:
   - Point to `preprocessed_data_multilabel_originalFPS`
   - Set `num_classes: 2` (but interpret as 2 independent sigmoids)

## Status

✅ Preprocessing script created
✅ Currently running in background (PID: check with `ps aux | grep preprocess_stanford_multilabel`)
✅ No balancing applied (dataset kept unbalanced)
✅ Multi-label format implemented

Processing all 32 videos at original FPS (~25 fps).
Expected completion time: ~2-3 hours (depending on GPU/CPU).
