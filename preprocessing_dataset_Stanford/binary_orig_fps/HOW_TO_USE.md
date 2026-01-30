# How to Use: Modified Preprocessing (Original FPS with Auto-Detection)

## Quick Start

### 1. Make the script executable
```bash
chmod +x run_preprocessing_original_fps.sh
```

### 2. Run the preprocessing
```bash
nohup bash run_preprocessing_original_fps.sh > preprocessing.log 2>&1 &
```

### 3. Monitor progress
```bash
tail -f preprocessing.log
```

That's it! All paths are already configured in the script.

---

## What It Does

This preprocessing script:
- **Auto-detects video FPS** and uses it as clip length
- Extracts **ALL frames** from each video
- Groups frames by 1-second windows
- Creates overlapping clips within each second
- Aggregates features per second (mean pooling)
- Outputs one feature vector per second (matching CSV labels)
- Shows detailed progress including:
  - Video FPS detected
  - Frames read vs. expected (missing frames count)
  - Processing progress for each video

---

## Configuration (Inside the Script)

The script is pre-configured with these paths:
```bash
VIDEOS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut"
LABELS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/labels"
OUTPUT_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS"
I3D_WEIGHTS="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/pretrained_models/i3d_rgb_kinetics.pt"
```

And these parameters:
- **Clip length**: AUTO-DETECTED from video FPS (e.g., 30 fps → 30 frames per clip)
- **Aggregation**: mean (alternative: max)
- **Train ratio**: 0.8
- **Number of splits**: 5
- **Balance seed**: 42
- **CPU threads**: 4

---

## What You'll See in the Log

For each video, you'll see:

```
================================================================================
VIDEO 1/10: video_name
================================================================================
[video_name] Found label file: 25-021_video_name.csv
[video_name] Step 1/4: Extracting features...

      ═══════════════════════════════════════════════════════
      VIDEO PROPERTIES
      ═══════════════════════════════════════════════════════
      FPS detected: 30.00
      Clip length (auto): 30 frames
      Total frames: 9000
      Duration: 300.00 seconds
      Expected seconds: 300
      Frames per second: ~30.0
      ═══════════════════════════════════════════════════════

      ───────────────────────────────────────────────────────
      READING ALL FRAMES FROM VIDEO
      ───────────────────────────────────────────────────────
      Progress: 500/9000 frames (5.6%)
      Progress: 1000/9000 frames (11.1%)
      ...
      ───────────────────────────────────────────────────────
      FRAME READING COMPLETE
      ───────────────────────────────────────────────────────
      Expected frames: 9000
      Frames read: 9000
      Frames missing: 0
      ───────────────────────────────────────────────────────

      FRAMES DISTRIBUTION BY SECOND:
        Second   0:  30 frames
        Second   1:  30 frames
        ...

      Frame statistics:
        Average frames/second: 30.0
        Min frames/second: 30
        Max frames/second: 30

      ───────────────────────────────────────────────────────
      EXTRACTING FEATURES PER SECOND
      ───────────────────────────────────────────────────────
      Total seconds to process: 300
        Second 10/300 - 30 frames, 2 clips
        Second 20/300 - 30 frames, 2 clips
        ...

[video_name] SUCCESS - Video processed successfully
```

---

## Key Information to Monitor

### 1. FPS Detection
Look for:
```
FPS detected: 30.00
Clip length (auto): 30 frames
```

### 2. Missing Frames
Look for:
```
Frames missing: 0
```
If this number is > 0, some frames couldn't be read from the video.

### 3. Progress
Videos being processed:
```
VIDEO 1/10: video_name
VIDEO 2/10: another_video
```

---

## Configuration (Inside the Script)

The script is pre-configured with these paths:
```bash
VIDEOS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/videos_cut"
LABELS_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/raw/labels"
OUTPUT_DIR="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS"
I3D_WEIGHTS="/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/pretrained_models/i3d_rgb_kinetics.pt"
```

And these parameters:
- **Clip length**: 16 frames
- **Aggregation**: mean (alternative: max)
- **Train ratio**: 0.8
- **Number of splits**: 5
- **Balance seed**: 42
- **CPU threads**: 4

---

## To Modify Settings

Edit the `run_preprocessing_original_fps.sh` file and change:

### Change aggregation method
```bash
AGGREGATION="max"  # Change from "mean" to "max"
```

### Force CPU usage
```bash
USE_CPU="--use_cpu"  # Set to force CPU
```

**Note**: Clip length is now auto-detected from video FPS and cannot be manually set.

---

## Output Structure

After running, you'll find:
```
/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS/
├── features/
│   ├── video1.npy  # shape: (num_seconds, feature_dim)
│   ├── video2.npy
│   └── ...
├── groundTruth/
│   ├── video1.txt  # one label per line (per second)
│   ├── video2.txt
│   └── ...
├── mapping.txt     # class names: background, licking
├── splits/
│   ├── train.split1.bundle
│   ├── test.split1.bundle
│   └── ...
└── preprocessing_original_fps_YYYYMMDD_HHMMSS.log
```

---

## Verification

After preprocessing, verify the output:

```python
import numpy as np

# Load one video's features
features = np.load('/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS/features/video1.npy')

# Load corresponding labels
with open('/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS/groundTruth/video1.txt', 'r') as f:
    labels = [line.strip() for line in f]

# Verify alignment
print(f"Features shape: {features.shape}")  # (num_seconds, feature_dim)
print(f"Number of labels: {len(labels)}")   # Should match num_seconds
print(f"Aligned: {features.shape[0] == len(labels)}")  # Should be True

# Check class distribution
from collections import Counter
print(f"Label distribution: {Counter(labels)}")
```

---

## Expected Processing Time

- **Per video**: 30-60 minutes on CPU, 1-2 minutes on GPU
- **Full dataset**: Several hours (depends on number of videos)

---

## Key Differences from Original

| Aspect | Original | Modified |
|--------|----------|----------|
| **FPS** | 1 fps (downsampled) | Original video FPS |
| **Clip Length** | Fixed at 16 frames | AUTO from video FPS (e.g., 30) |
| **Frames Used** | 1 per second | ALL frames |
| **Information** | ~4% | 100% |
| **Processing Time** | 1-2 min/video | 30-60 min/video |
| **Missing Frame Detection** | No | Yes (reported in log) |

---

## Troubleshooting

### Out of Memory?
Edit the script and add:
```bash
USE_CPU="--use_cpu"
```

### Too Slow?
- Use GPU (much faster)
- Reduce CPU threads if needed:
  ```bash
  CPU_THREADS=2
  ```

### Feature/Label Mismatch?
- Check that CSV files have one row per second
- Verify video files are not corrupted
- Check video duration matches CSV length

---

## Next Steps

After preprocessing completes:

1. **Verify outputs** (run verification code above)
2. **Train FACT model** using the new features:
   ```bash
   python train.py --config configs/stanford_binary.yaml \
       --data_root /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_2class_originalFPS
   ```
3. **Compare performance** with original preprocessing

---

## Log File

All output is saved to a log file with timestamp:
```
preprocessing_original_fps_YYYYMMDD_HHMMSS.log
```

Check this file for:
- Processing progress
- Any errors or warnings
- Final statistics

---

## Support

For issues:
1. Check the log file for errors
2. Verify input paths exist and contain correct files
3. Ensure I3D weights file is present
4. Check GPU/CPU availability and memory
