#!/usr/bin/env python3
"""
Modified Preprocessing script for Stanford dataset to FACT format
Binary classification: background vs licking (balanced)

KEY CHANGE: Uses ORIGINAL FPS instead of downsampling to 1 fps
- Extracts ALL frames at native video FPS
- Groups frames by 1-second windows
- Aggregates features from all frames within each second
- One label per second (matching CSV structure)
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from datetime import datetime

# Set CPU thread limits early
def set_cpu_threads(num_threads=4):
    """Set number of CPU threads for various libraries"""
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    print(f"Set CPU thread limits to {num_threads} threads")

try:
    import torch.nn as nn
    from torchvision import transforms
except ImportError:
    print("Error: PyTorch and torchvision are required")
    sys.exit(1)

# Import I3D model
try:
    from i3d_model import load_i3d_model as load_i3d_from_file
    I3D_AVAILABLE = True
except ImportError:
    I3D_AVAILABLE = False
    print("Warning: i3d_model.py not found, will use fallback feature extractor")


def load_i3d_model(device, weights_path=None):
    """Load pretrained I3D model for feature extraction"""
    print("Loading pretrained I3D model...")
    
    if I3D_AVAILABLE and weights_path is not None:
        try:
            print(f"Loading I3D model from weights: {weights_path}")
            model = load_i3d_from_file(weights_path=weights_path, device=device)
            print(f"I3D model loaded successfully on {device}")
            return model, 'i3d'
        except Exception as e:
            print(f"Error loading I3D model: {e}")
            print("Falling back to alternative feature extractor...")
    
    # Fallback to ResNet3D
    try:
        from torchvision.models.video import r3d_18
        print("Using R3D-18 as feature extractor")
        print("Note: For best results, use I3D with --i3d_weights argument")
        model = r3d_18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(device)
        model.eval()
        print(f"R3D-18 model loaded successfully on {device}")
        return model, 'r3d'
    except Exception as e:
        print(f"Error loading fallback model: {e}")
        sys.exit(1)


def extract_video_features_per_second(video_path, model, model_type, device, aggregation='mean'):
    """
    Extract features from video using ORIGINAL FPS, aggregating per second.
    
    NEW APPROACH:
    1. Detect video FPS and use it as clip length
    2. Extract ALL frames at original FPS
    3. Group frames by 1-second windows
    4. For each second, create overlapping clips and extract features
    5. Aggregate features within each second (mean/max pooling)
    6. Return one feature vector per second
    
    Args:
        video_path: Path to video file
        model: Feature extraction model (I3D or R3D)
        model_type: Type of model ('i3d' or 'r3d')
        device: torch device
        aggregation: How to aggregate features within a second ('mean' or 'max')
        
    Returns:
        numpy array of shape (num_seconds, feature_dim)
    """
    print(f"      Opening video file...", flush=True)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    num_seconds = int(np.ceil(duration))
    
    # Use FPS as clip length (rounded to nearest integer)
    clip_len = max(1, int(round(fps)))
    
    print(f"", flush=True)
    print(f"      ═══════════════════════════════════════════════════════", flush=True)
    print(f"      VIDEO PROPERTIES", flush=True)
    print(f"      ═══════════════════════════════════════════════════════", flush=True)
    print(f"      FPS detected: {fps:.2f}", flush=True)
    print(f"      Clip length (auto): {clip_len} frames", flush=True)
    print(f"      Total frames: {total_frames}", flush=True)
    print(f"      Duration: {duration:.2f} seconds", flush=True)
    print(f"      Expected seconds: {num_seconds}", flush=True)
    print(f"      Frames per second: ~{total_frames/num_seconds:.1f}" if num_seconds > 0 else "", flush=True)
    print(f"      ═══════════════════════════════════════════════════════", flush=True)
    print(f"", flush=True)
    
    # ImageNet normalization stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    print(f"      Using ORIGINAL FPS with per-second aggregation", flush=True)
    print(f"      Aggregation method: {aggregation}", flush=True)
    sys.stdout.flush()
    
    # Store all frames first (organized by second)
    frames_by_second = [[] for _ in range(num_seconds)]
    
    frame_idx = 0
    frames_read = 0
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      READING ALL FRAMES FROM VIDEO", flush=True)
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_read += 1
        
        # Determine which second this frame belongs to
        current_time = frame_idx / fps
        second_idx = int(current_time)
        
        if second_idx < num_seconds:
            # Preprocess frame
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            
            frames_by_second[second_idx].append(frame)
        
        frame_idx += 1
        
        # Progress update every 500 frames
        if frame_idx % 500 == 0:
            progress_pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"      Progress: {frame_idx}/{total_frames} frames ({progress_pct:.1f}%)", flush=True)
    
    cap.release()
    
    # Calculate missing frames
    frames_missing = total_frames - frames_read
    
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      FRAME READING COMPLETE", flush=True)
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      Expected frames: {total_frames}", flush=True)
    print(f"      Frames read: {frames_read}", flush=True)
    print(f"      Frames missing: {frames_missing}", flush=True)
    if frames_missing > 0:
        print(f"      WARNING: {frames_missing} frames could not be read!", flush=True)
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"", flush=True)
    
    # Show distribution of frames across seconds
    print(f"      FRAMES DISTRIBUTION BY SECOND:", flush=True)
    for sec_idx in range(min(10, num_seconds)):
        print(f"        Second {sec_idx:3d}: {len(frames_by_second[sec_idx]):3d} frames", flush=True)
    if num_seconds > 10:
        print(f"        ... ({num_seconds - 10} more seconds)", flush=True)
    
    # Calculate statistics
    frames_per_sec = [len(frames_by_second[i]) for i in range(num_seconds)]
    avg_frames = np.mean(frames_per_sec) if len(frames_per_sec) > 0 else 0
    min_frames = np.min(frames_per_sec) if len(frames_per_sec) > 0 else 0
    max_frames = np.max(frames_per_sec) if len(frames_per_sec) > 0 else 0
    
    print(f"      ", flush=True)
    print(f"      Frame statistics:", flush=True)
    print(f"        Average frames/second: {avg_frames:.1f}", flush=True)
    print(f"        Min frames/second: {min_frames}", flush=True)
    print(f"        Max frames/second: {max_frames}", flush=True)
    print(f"      ", flush=True)
    sys.stdout.flush()
    
    # Now extract features per second
    all_features = []
    
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      EXTRACTING FEATURES PER SECOND", flush=True)
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      Total seconds to process: {num_seconds}", flush=True)
    sys.stdout.flush()
    
    with torch.no_grad():
        for sec_idx in range(num_seconds):
            frames = frames_by_second[sec_idx]
            
            if len(frames) == 0:
                # No frames in this second, create zero feature
                # Determine feature dimension from model
                dummy_clip = torch.zeros(1, 3, clip_len, 224, 224).to(device)
                if model_type == 'i3d':
                    dummy_feat = model.extract_features(dummy_clip)
                else:
                    dummy_feat = model(dummy_clip)
                feat_dim = dummy_feat.squeeze().shape[0]
                
                second_feature = np.zeros(feat_dim, dtype=np.float32)
                all_features.append(second_feature)
                
                if (sec_idx + 1) % 10 == 0:
                    print(f"        Second {sec_idx + 1}/{num_seconds} - NO FRAMES (zero feature)", flush=True)
                
                continue
            
            # Create overlapping clips from frames in this second
            clip_features = []
            
            if len(frames) < clip_len:
                # Pad with last frame if not enough frames
                original_len = len(frames)
                while len(frames) < clip_len:
                    frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.float32))
                
                # Process single padded clip
                clip_array = np.stack(frames[:clip_len], axis=0)
                clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
                clip_tensor = clip_tensor.float().to(device)
                
                if model_type == 'i3d':
                    feat = model.extract_features(clip_tensor)
                else:
                    feat = model(clip_tensor)
                
                feat = feat.squeeze().cpu().numpy()
                clip_features.append(feat)
                
                if (sec_idx + 1) % 10 == 0:
                    print(f"        Second {sec_idx + 1}/{num_seconds} - {original_len} frames (padded to {clip_len})", flush=True)
                
            else:
                # Create overlapping clips with 50% stride
                stride = clip_len // 2
                num_clips = (len(frames) - clip_len) // stride + 1
                
                for clip_idx in range(num_clips):
                    start_idx = clip_idx * stride
                    end_idx = start_idx + clip_len
                    
                    if end_idx <= len(frames):
                        clip_frames = frames[start_idx:end_idx]
                        
                        clip_array = np.stack(clip_frames, axis=0)
                        clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
                        clip_tensor = clip_tensor.float().to(device)
                        
                        if model_type == 'i3d':
                            feat = model.extract_features(clip_tensor)
                        else:
                            feat = model(clip_tensor)
                        
                        feat = feat.squeeze().cpu().numpy()
                        clip_features.append(feat)
                
                # If no clips were created (shouldn't happen), process last clip_len frames
                if len(clip_features) == 0:
                    clip_frames = frames[-clip_len:]
                    clip_array = np.stack(clip_frames, axis=0)
                    clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
                    clip_tensor = clip_tensor.float().to(device)
                    
                    if model_type == 'i3d':
                        feat = model.extract_features(clip_tensor)
                    else:
                        feat = model(clip_tensor)
                    
                    feat = feat.squeeze().cpu().numpy()
                    clip_features.append(feat)
                
                if (sec_idx + 1) % 10 == 0:
                    print(f"        Second {sec_idx + 1}/{num_seconds} - {len(frames)} frames, {len(clip_features)} clips", flush=True)
            
            # Aggregate clip features for this second
            if aggregation == 'mean':
                second_feature = np.mean(clip_features, axis=0)
            elif aggregation == 'max':
                second_feature = np.max(clip_features, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            all_features.append(second_feature)
    
    features = np.array(all_features)
    
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      FEATURE EXTRACTION COMPLETE", flush=True)
    print(f"      ───────────────────────────────────────────────────────", flush=True)
    print(f"      Final features shape: {features.shape}", flush=True)
    print(f"      Features per second: {features.shape[0]}", flush=True)
    print(f"      Feature dimension: {features.shape[1]}", flush=True)
    print(f"      ═══════════════════════════════════════════════════════", flush=True)
    print(f"", flush=True)
    sys.stdout.flush()
    
    return features


def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration = total_frames / fps if fps > 0 else 0
    return duration


def convert_labels_to_binary_frame_level(label_file, num_seconds):
    """
    Convert CSV labels to binary per-second labels (background vs licking)
    CSV format: start_time,end_time,licking,shaking (binary indicators 1.0/0.0)
    REMOVES frames where shaking=1.0 (pure shaking or licking+shaking)
    
    Args:
        label_file: Path to CSV file
        num_seconds: Expected number of seconds in video
    
    Returns:
        tuple: (frame_labels, kept_indices)
            - frame_labels: list of labels for kept seconds
            - kept_indices: indices of seconds to keep (0-indexed)
    """
    df = pd.read_csv(label_file)
    
    print(f"      CSV columns: {df.columns.tolist()}")
    print(f"      CSV shape: {df.shape}")
    print(f"      Video seconds: {num_seconds}")
    
    # Initialize lists
    frame_labels = []
    frame_indices_to_keep = []
    
    # Process each row (each row = 1 second)
    for idx, row in df.iterrows():
        # Only process rows that correspond to actual video seconds
        if idx >= num_seconds:
            continue
            
        licking_val = row['licking']
        shaking_val = row['shaking']
        
        # Determine the label for this second
        # If shaking is present (1.0), skip this second entirely
        if shaking_val == 1.0:
            # Skip seconds with shaking (either pure shaking or licking+shaking)
            continue
        elif licking_val == 1.0:
            # Pure licking (no shaking)
            frame_labels.append('licking')
            frame_indices_to_keep.append(idx)
        else:
            # Background (no licking, no shaking)
            frame_labels.append('background')
            frame_indices_to_keep.append(idx)
    
    print(f"      Total rows in CSV: {len(df)}", flush=True)
    print(f"      Valid rows (within video duration): {min(len(df), num_seconds)}", flush=True)
    print(f"      Rows kept after removing shaking: {len(frame_labels)}", flush=True)
    sys.stdout.flush()
    
    return frame_labels, frame_indices_to_keep


def balance_dataset(features, labels, indices, seed=42):
    """
    Balance the dataset by downsampling background frames to match licking frames
    
    Args:
        features: numpy array of features
        labels: list of labels
        indices: list of second indices that were kept
        seed: random seed for reproducibility
        
    Returns:
        balanced_features, balanced_labels, balanced_indices
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Count instances of each class
    background_indices = [i for i, label in enumerate(labels) if label == 'background']
    licking_indices = [i for i, label in enumerate(labels) if label == 'licking']
    
    n_background = len(background_indices)
    n_licking = len(licking_indices)
    
    print(f"      Before balancing:")
    print(f"        Background: {n_background} samples")
    print(f"        Licking: {n_licking} samples")
    print(f"        Ratio: {n_background/n_licking:.2f}:1" if n_licking > 0 else "        Ratio: N/A")
    
    # Downsample background to match licking
    if n_background > n_licking:
        # Randomly select background frames
        selected_background_indices = random.sample(background_indices, n_licking)
        
        # Combine with all licking frames
        selected_indices = sorted(selected_background_indices + licking_indices)
        
        # Filter features and labels
        balanced_features = features[selected_indices]
        balanced_labels = [labels[i] for i in selected_indices]
        balanced_indices = [indices[i] for i in selected_indices]
        
        print(f"      After balancing:")
        print(f"        Background: {len([l for l in balanced_labels if l == 'background'])} samples")
        print(f"        Licking: {len([l for l in balanced_labels if l == 'licking'])} samples")
        print(f"        Total: {len(balanced_labels)} samples")
        
    else:
        print(f"      No balancing needed (more licking than background)")
        balanced_features = features
        balanced_labels = labels
        balanced_indices = indices
    
    return balanced_features, balanced_labels, balanced_indices


def create_mapping_file(mapping_file):
    """Create mapping.txt file with class labels"""
    with open(mapping_file, 'w') as f:
        f.write("0 background\n")
        f.write("1 licking\n")
    print(f"Created mapping file: {mapping_file}", flush=True)


def create_train_test_splits(video_ids, output_dir, train_ratio=0.8, n_splits=5, seed=42):
    """Create train/test splits"""
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    random.seed(seed)
    
    for split_idx in range(n_splits):
        # Shuffle videos
        shuffled_ids = video_ids.copy()
        random.shuffle(shuffled_ids)
        
        # Split into train/test
        split_point = int(len(shuffled_ids) * train_ratio)
        train_ids = shuffled_ids[:split_point]
        test_ids = shuffled_ids[split_point:]
        
        # Save train split
        train_file = splits_dir / f'train.split{split_idx + 1}.bundle'
        with open(train_file, 'w') as f:
            for video_id in train_ids:
                f.write(f"{video_id}\n")
        
        # Save test split
        test_file = splits_dir / f'test.split{split_idx + 1}.bundle'
        with open(test_file, 'w') as f:
            for video_id in test_ids:
                f.write(f"{video_id}\n")
        
        print(f"Created split {split_idx + 1}:", flush=True)
        print(f"  Train: {len(train_ids)} videos -> {train_file}", flush=True)
        print(f"  Test: {len(test_ids)} videos -> {test_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Stanford dataset for FACT (Modified: Original FPS)')
    parser.add_argument('--input_videos', type=str, required=True,
                        help='Path to directory containing input videos (.mp4)')
    parser.add_argument('--input_labels', type=str, required=True,
                        help='Path to directory containing label CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--i3d_weights', type=str, default=None,
                        help='Path to I3D model weights file')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'max'],
                        help='How to aggregate features within each second (default: mean)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/test splits to create (default: 5)')
    parser.add_argument('--balance_seed', type=int, default=42,
                        help='Random seed for balancing and splits (default: 42)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of CPU threads to use (default: 4)')
    
    args = parser.parse_args()
    
    # Set CPU threads
    set_cpu_threads(args.cpu_threads)
    
    print("=" * 80, flush=True)
    print("STANFORD BINARY DATASET PREPROCESSING (ORIGINAL FPS)", flush=True)
    print("=" * 80, flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"  Input videos: {args.input_videos}", flush=True)
    print(f"  Input labels: {args.input_labels}", flush=True)
    print(f"  Output directory: {args.output_dir}", flush=True)
    print(f"  I3D weights: {args.i3d_weights}", flush=True)
    print(f"  Clip length: AUTO-DETECTED from video FPS", flush=True)
    print(f"  Aggregation: {args.aggregation}", flush=True)
    print(f"  Train ratio: {args.train_ratio}", flush=True)
    print(f"  Number of splits: {args.n_splits}", flush=True)
    print(f"  Balance seed: {args.balance_seed}", flush=True)
    print(f"  Use CPU: {args.use_cpu}", flush=True)
    print(f"  CPU threads: {args.cpu_threads}", flush=True)
    sys.stdout.flush()
    
    # Setup paths
    videos_dir = Path(args.input_videos)
    labels_dir = Path(args.input_labels)
    output_dir = Path(args.output_dir)
    
    if not videos_dir.exists():
        print(f"Error: Videos directory not found: {videos_dir}")
        sys.exit(1)
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        sys.exit(1)
    
    # Create output directories
    features_dir = output_dir / 'features'
    groundtruth_dir = output_dir / 'groundTruth'
    
    features_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── features/")
    print(f"  ├── groundTruth/")
    print(f"  ├── mapping.txt")
    print(f"  └── splits/")
    
    # Setup device
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
        print(f"PyTorch CPU threads set to {args.cpu_threads}")
    
    # Load model
    model, model_type = load_i3d_model(device, weights_path=args.i3d_weights)
    print(f"Model type: {model_type}")
    
    # Get list of videos
    video_files = sorted(videos_dir.glob('*.mp4'))
    
    if len(video_files) == 0:
        print(f"Error: No MP4 files found in {videos_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(video_files)} videos to process")
    
    # Process each video
    video_ids = []
    successful_videos = 0
    failed_videos = 0
    
    # Statistics
    total_original_seconds = 0
    total_after_removal = 0
    total_after_balancing = 0
    
    print(f"\n{'='*80}")
    print(f"Starting to process {len(video_files)} videos...")
    print(f"{'='*80}\n")
    
    for video_idx, video_file in enumerate(video_files, 1):
        video_id = video_file.stem
        video_ids.append(video_id)
        
        print(f"\n{'='*80}")
        print(f"Processing video {video_idx}/{len(video_files)}: {video_id}")
        print(f"{'='*80}")
        
        # Find corresponding label file
        label_file = labels_dir / f"25-021_{video_id}.csv"
        
        if not label_file.exists():
            print(f"[{video_id}] ERROR: Label file not found at {label_file}")
            print(f"[{video_id}] Skipping...")
            failed_videos += 1
            continue
        
        print(f"[{video_id}] Found label file: {label_file.name}")
        
        # Extract video features (per second)
        try:
            print(f"[{video_id}] Step 1/4: Extracting features (per second)...", flush=True)
            sys.stdout.flush()
            
            features = extract_video_features_per_second(
                video_file, model, model_type, device,
                aggregation=args.aggregation
            )
            
            print(f"[{video_id}]   Features extracted: {features.shape}")
            total_original_seconds += len(features)
            
        except Exception as e:
            print(f"[{video_id}] ERROR extracting features: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Convert labels to binary
        try:
            print(f"[{video_id}] Step 2/4: Converting to binary labels and removing unwanted classes...", flush=True)
            sys.stdout.flush()
            
            num_seconds = len(features)
            print(f"[{video_id}]   Number of seconds: {num_seconds}", flush=True)
            
            frame_labels, kept_indices = convert_labels_to_binary_frame_level(
                label_file, num_seconds
            )
            
            print(f"[{video_id}]   Original seconds: {len(features)}", flush=True)
            print(f"[{video_id}]   After removing shaking/licking_shaking: {len(frame_labels)}", flush=True)
            sys.stdout.flush()
            
            # Filter features to match kept indices
            features_filtered = features[kept_indices]
            
            total_after_removal += len(features_filtered)
            
        except Exception as e:
            print(f"[{video_id}] ERROR processing labels: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            failed_videos += 1
            continue
        
        # Balance dataset
        try:
            print(f"[{video_id}] Step 3/4: Balancing classes...", flush=True)
            sys.stdout.flush()
            
            balanced_features, balanced_labels, balanced_indices = balance_dataset(
                features_filtered, frame_labels, kept_indices, seed=args.balance_seed
            )
            
            total_after_balancing += len(balanced_labels)
            
        except Exception as e:
            print(f"[{video_id}] ERROR balancing dataset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            failed_videos += 1
            continue
        
        # Save processed data
        try:
            print(f"[{video_id}] Step 4/4: Saving processed data...", flush=True)
            sys.stdout.flush()
            
            # Ensure features and labels match in length
            if len(balanced_features) != len(balanced_labels):
                print(f"[{video_id}]   WARNING: Length mismatch!", flush=True)
                print(f"[{video_id}]     Features: {len(balanced_features)}", flush=True)
                print(f"[{video_id}]     Labels: {len(balanced_labels)}", flush=True)
                min_len = min(len(balanced_features), len(balanced_labels))
                balanced_features = balanced_features[:min_len]
                balanced_labels = balanced_labels[:min_len]
                print(f"[{video_id}]     Trimmed to: {min_len}", flush=True)
            
            # Save features
            feature_file = features_dir / f"{video_id}.npy"
            np.save(feature_file, balanced_features)
            print(f"[{video_id}]   Saved features: {feature_file}", flush=True)
            print(f"[{video_id}]     Shape: {balanced_features.shape}", flush=True)
            print(f"[{video_id}]     Size: {balanced_features.nbytes / 1024 / 1024:.2f} MB", flush=True)
            
            # Count final distribution
            from collections import Counter
            label_counts = Counter(balanced_labels)
            print(f"[{video_id}]   Final label distribution:", flush=True)
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(balanced_labels)) * 100
                print(f"[{video_id}]     - {label}: {count} seconds ({percentage:.1f}%)", flush=True)
            
            # Save ground truth labels
            gt_file = groundtruth_dir / f"{video_id}.txt"
            with open(gt_file, 'w') as f:
                for label in balanced_labels:
                    f.write(f"{label}\n")
            print(f"[{video_id}]   Saved labels: {gt_file}", flush=True)
            
            print(f"[{video_id}] SUCCESS - Video processed successfully", flush=True)
            print(f"", flush=True)
            sys.stdout.flush()
            successful_videos += 1
            
        except Exception as e:
            print(f"[{video_id}] ERROR saving data: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            failed_videos += 1
            continue
    
    print("\n" + "=" * 80, flush=True)
    print("Processing Summary", flush=True)
    print("=" * 80, flush=True)
    print(f"Total videos: {len(video_files)}", flush=True)
    print(f"Successfully processed: {successful_videos}", flush=True)
    print(f"Failed: {failed_videos}", flush=True)
    print(f"Success rate: {(successful_videos/len(video_files)*100):.1f}%" if len(video_files) > 0 else "N/A", flush=True)
    print("", flush=True)
    print("Dataset Statistics:", flush=True)
    print(f"  Total original seconds: {total_original_seconds}", flush=True)
    print(f"  After removing shaking/licking_shaking: {total_after_removal}", flush=True)
    print(f"  After balancing: {total_after_balancing}", flush=True)
    print(f"  Reduction ratio: {total_after_balancing/total_original_seconds:.2%}" if total_original_seconds > 0 else "N/A", flush=True)
    sys.stdout.flush()
    
    print("\n" + "=" * 80)
    print("Creating dataset metadata files...")
    print("=" * 80)
    
    # Create mapping file
    mapping_file = output_dir / 'mapping.txt'
    create_mapping_file(mapping_file)
    
    # Create train/test splits
    create_train_test_splits(video_ids, output_dir,
                             train_ratio=args.train_ratio,
                             n_splits=args.n_splits,
                             seed=args.balance_seed)
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── features/        ({len(list(features_dir.glob('*.npy')))} .npy files)")
    print(f"  ├── groundTruth/     ({len(list(groundtruth_dir.glob('*.txt')))} .txt files)")
    print(f"  ├── mapping.txt      (binary: background, licking)")
    print(f"  └── splits/          ({args.n_splits} train/test splits)")
    print("\nKEY DIFFERENCES FROM ORIGINAL:")
    print("  - Uses ORIGINAL video FPS (no downsampling to 1 fps)")
    print("  - Extracts ALL frames from video")
    print("  - Groups frames by 1-second windows")
    print("  - Aggregates features within each second")
    print("  - One feature vector per second (matching CSV labels)")
    print("\nNext steps:")
    print(f"  1. Verify the output in: {output_dir}")
    print(f"  2. Generate FACT config file")
    print(f"  3. Train the model")


if __name__ == '__main__':
    main()