#!/usr/bin/env python3
"""
Preprocessing script for Stanford dataset to FACT format
Binary classification: background vs licking (balanced)
- Removes 'shaking' and 'licking_shaking' classes
- Balances dataset by downsampling background frames
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


def extract_video_features(video_path, model, model_type, device, target_fps=25, clip_len=16):
    """
    Extract I3D features from a video file using STREAMING processing
    
    Returns:
        numpy array of shape (num_frames, feature_dim)
    """
    print(f"      Opening video file...")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"      Original video FPS: {fps:.2f}")
    print(f"      Original total frames: {total_frames}")
    
    # Calculate frame sampling rate
    sample_rate = max(1, int(fps / target_fps))
    print(f"      Sample rate: 1 every {sample_rate} frames")
    
    # ImageNet normalization stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    print(f"      Using STREAMING mode - processing clips on-the-fly")
    print(f"      Clip length: {clip_len} frames")
    
    # Streaming: accumulate frames in a buffer
    frame_buffer = []
    all_features = []
    
    frame_idx = 0
    frames_sampled = 0
    clip_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % sample_rate == 0:
                # Preprocess frame
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - mean) / std
                
                frame_buffer.append(frame)
                frames_sampled += 1
                
                # When buffer has enough frames, process a clip
                if len(frame_buffer) >= clip_len:
                    # Extract clip from buffer
                    clip_frames = frame_buffer[:clip_len]
                    
                    # Convert to tensor: (1, C, T, H, W)
                    clip_array = np.stack(clip_frames, axis=0)  # (T, H, W, C)
                    clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
                    clip_tensor = clip_tensor.float().to(device)
                    
                    # First clip: verify GPU usage
                    if clip_count == 0:
                        print(f"      ✓ First clip on device: {clip_tensor.device}")
                        sys.stdout.flush()
                    
                    # Extract features using GPU
                    if model_type == 'i3d':
                        feat = model.extract_features(clip_tensor)
                    else:
                        feat = model(clip_tensor)
                    
                    feat = feat.squeeze().cpu().numpy()
                    
                    # Replicate feature for overlap (50% stride)
                    stride = clip_len // 2
                    for _ in range(stride):
                        all_features.append(feat)
                    
                    # Remove processed frames from buffer (50% overlap)
                    frame_buffer = frame_buffer[stride:]
                    
                    clip_count += 1
                    if clip_count % 100 == 0:
                        print(f"      Processed {clip_count} clips, {frames_sampled} frames sampled...")
                        sys.stdout.flush()
            
            frame_idx += 1
            
            # Progress for frame reading (less frequent to reduce I/O)
            if frame_idx % 5000 == 0:
                print(f"      Read {frame_idx}/{total_frames} frames ({frames_sampled} sampled, {clip_count} clips processed)...")
                sys.stdout.flush()
        
        # Process remaining frames in buffer
        while len(frame_buffer) >= clip_len:
            clip_frames = frame_buffer[:clip_len]
            
            clip_array = np.stack(clip_frames, axis=0)
            clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
            clip_tensor = clip_tensor.float().to(device)
            
            if model_type == 'i3d':
                feat = model.extract_features(clip_tensor)
            else:
                feat = model(clip_tensor)
            
            feat = feat.squeeze().cpu().numpy()
            
            stride = clip_len // 2
            for _ in range(stride):
                all_features.append(feat)
            
            frame_buffer = frame_buffer[stride:]
            clip_count += 1
        
        # Handle final partial buffer
        if len(frame_buffer) > 0:
            # Pad to clip_len
            while len(frame_buffer) < clip_len:
                frame_buffer.append(np.zeros((224, 224, 3), dtype=np.float32))
            
            clip_frames = frame_buffer[:clip_len]
            clip_array = np.stack(clip_frames, axis=0)
            clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
            clip_tensor = clip_tensor.float().to(device)
            
            if model_type == 'i3d':
                feat = model.extract_features(clip_tensor)
            else:
                feat = model(clip_tensor)
            
            feat = feat.squeeze().cpu().numpy()
            
            # Add remaining features
            remaining = frames_sampled - len(all_features)
            for _ in range(remaining):
                all_features.append(feat)
    
    cap.release()
    
    # Convert to array and trim to exact frame count
    features = np.array(all_features[:frames_sampled])
    
    print(f"      Total frames read: {frame_idx}")
    print(f"      Total frames sampled: {frames_sampled}")
    print(f"      Total clips processed: {clip_count}")
    print(f"      Feature extraction complete!")
    print(f"      Final features shape: {features.shape}")
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


def convert_labels_to_binary_frame_level(label_file, video_duration, fps=25):
    """
    Convert CSV labels to binary frame-level labels (background vs licking)
    CSV format: start_time,end_time,licking,shaking (binary indicators 1.0/0.0)
    REMOVES frames where shaking=1.0 (pure shaking or licking+shaking)
    
    Returns:
        list of labels (one per frame) with indices of frames to keep
    """
    df = pd.read_csv(label_file)
    
    print(f"      CSV columns: {df.columns.tolist()}")
    print(f"      CSV shape: {df.shape}")
    
    # Initialize lists
    frame_labels = []
    frame_indices_to_keep = []
    
    # Process each row (each row = 1 second)
    for idx, row in df.iterrows():
        licking_val = row['licking']
        shaking_val = row['shaking']
        
        # Determine the label for this second
        # If shaking is present (1.0), skip this frame entirely
        if shaking_val == 1.0:
            # Skip frames with shaking (either pure shaking or licking+shaking)
            continue
        elif licking_val == 1.0:
            # Pure licking (no shaking)
            frame_labels.append('licking')
            frame_indices_to_keep.append(idx)
        else:
            # Background (no licking, no shaking)
            frame_labels.append('background')
            frame_indices_to_keep.append(idx)
    
    print(f"      Total rows in CSV: {len(df)}")
    print(f"      Rows kept after removing shaking: {len(frame_labels)}")
    
    return frame_labels, frame_indices_to_keep


def balance_dataset(features, labels, indices, seed=42):
    """
    Balance the dataset by downsampling background frames to match licking frames
    
    Args:
        features: numpy array of features
        labels: list of labels
        indices: list of frame indices that were kept
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
    print(f"        Background: {n_background} frames")
    print(f"        Licking: {n_licking} frames")
    print(f"        Ratio: {n_background/n_licking:.2f}:1")
    
    # Downsample background to match licking
    if n_background > n_licking:
        # Randomly select background frames
        selected_background_indices = random.sample(background_indices, n_licking)
        
        # Combine with all licking frames
        selected_indices = sorted(selected_background_indices + licking_indices)
        
        # Filter features and labels
        balanced_features = features[selected_indices]
        balanced_labels = [labels[i] for i in selected_indices]
        balanced_original_indices = [indices[i] for i in selected_indices]
        
        print(f"      After balancing:")
        print(f"        Background: {n_licking} frames")
        print(f"        Licking: {n_licking} frames")
        print(f"        Total: {len(balanced_labels)} frames")
        print(f"        Ratio: 1:1")
        
        return balanced_features, balanced_labels, balanced_original_indices
    else:
        print(f"      No balancing needed (background <= licking)")
        return features, labels, indices


def create_mapping_file(output_file):
    """Create mapping file for binary classification"""
    with open(output_file, 'w') as f:
        f.write("0 background\n")
        f.write("1 licking\n")
    print(f"Created mapping file: {output_file}")


def create_train_test_splits(video_ids, output_dir, train_ratio=0.8, n_splits=1, seed=42):
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
        
        print(f"Created split {split_idx + 1}:")
        print(f"  Train: {len(train_ids)} videos -> {train_file}")
        print(f"  Test: {len(test_ids)} videos -> {test_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Stanford dataset for binary classification (balanced)')
    
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing video files (.mp4)')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing label files (.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed dataset')
    parser.add_argument('--i3d_weights', type=str, default=None,
                        help='Path to I3D weights file (.pt)')
    parser.add_argument('--target_fps', type=int, default=25,
                        help='Target FPS for feature extraction (default: 25)')
    parser.add_argument('--clip_length', type=int, default=16,
                        help='Number of frames per clip (default: 16)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--n_splits', type=int, default=1,
                        help='Number of train/test splits (default: 1)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU usage instead of GPU')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of CPU threads (default: 4)')
    parser.add_argument('--balance_seed', type=int, default=42,
                        help='Random seed for balancing (default: 42)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set CPU threads
    set_cpu_threads(args.cpu_threads)
    
    # Setup paths
    videos_dir = Path(args.videos_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    features_dir = output_dir / 'features'
    groundtruth_dir = output_dir / 'groundTruth'
    features_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Stanford Dataset - Binary Classification (Balanced)")
    print("=" * 80)
    print(f"Videos directory: {videos_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Clip length: {args.clip_length}")
    print(f"Balance seed: {args.balance_seed}")
    if args.i3d_weights:
        print(f"I3D weights: {args.i3d_weights}")
    print("")
    print("Configuration:")
    print("  - Classes: background vs licking (binary)")
    print("  - Removed: shaking, licking_shaking")
    print("  - Balanced: background downsampled to match licking")
    print("=" * 80)
    
    # Setup device
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
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
    total_original_frames = 0
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
        
        # Extract video features
        try:
            print(f"[{video_id}] Step 1/4: Extracting features...")
            
            features = extract_video_features(video_file, model, model_type, device, 
                                              target_fps=args.target_fps,
                                              clip_len=args.clip_length)
            
            print(f"[{video_id}]   Features extracted: {features.shape}")
            total_original_frames += len(features)
            
        except Exception as e:
            print(f"[{video_id}] ERROR extracting features: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Convert labels to binary
        try:
            print(f"[{video_id}] Step 2/4: Converting to binary labels and removing unwanted classes...")
            
            video_duration = get_video_duration(video_file)
            print(f"[{video_id}]   Video duration: {video_duration:.2f} seconds")
            
            frame_labels, kept_indices = convert_labels_to_binary_frame_level(
                label_file, video_duration, fps=args.target_fps
            )
            
            print(f"[{video_id}]   Original frames: {len(features)}")
            print(f"[{video_id}]   After removing shaking/licking_shaking: {len(frame_labels)}")
            
            # Filter features to match kept indices
            features_filtered = features[kept_indices]
            
            total_after_removal += len(features_filtered)
            
        except Exception as e:
            print(f"[{video_id}] ERROR processing labels: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Balance dataset
        try:
            print(f"[{video_id}] Step 3/4: Balancing classes...")
            
            balanced_features, balanced_labels, balanced_indices = balance_dataset(
                features_filtered, frame_labels, kept_indices, seed=args.balance_seed
            )
            
            total_after_balancing += len(balanced_labels)
            
        except Exception as e:
            print(f"[{video_id}] ERROR balancing dataset: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Save processed data
        try:
            print(f"[{video_id}] Step 4/4: Saving processed data...")
            
            # Ensure features and labels match in length
            if len(balanced_features) != len(balanced_labels):
                print(f"[{video_id}]   WARNING: Length mismatch!")
                print(f"[{video_id}]     Features: {len(balanced_features)}")
                print(f"[{video_id}]     Labels: {len(balanced_labels)}")
                min_len = min(len(balanced_features), len(balanced_labels))
                balanced_features = balanced_features[:min_len]
                balanced_labels = balanced_labels[:min_len]
                print(f"[{video_id}]     Trimmed to: {min_len}")
            
            # Save features
            feature_file = features_dir / f"{video_id}.npy"
            np.save(feature_file, balanced_features)
            print(f"[{video_id}]   Saved features: {feature_file}")
            print(f"[{video_id}]     Shape: {balanced_features.shape}")
            print(f"[{video_id}]     Size: {balanced_features.nbytes / 1024 / 1024:.2f} MB")
            
            # Count final distribution
            from collections import Counter
            label_counts = Counter(balanced_labels)
            print(f"[{video_id}]   Final label distribution:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(balanced_labels)) * 100
                print(f"[{video_id}]     - {label}: {count} frames ({percentage:.1f}%)")
            
            # Save ground truth labels
            gt_file = groundtruth_dir / f"{video_id}.txt"
            with open(gt_file, 'w') as f:
                for label in balanced_labels:
                    f.write(f"{label}\n")
            print(f"[{video_id}]   Saved labels: {gt_file}")
            
            print(f"[{video_id}] SUCCESS - Video processed successfully")
            successful_videos += 1
            
        except Exception as e:
            print(f"[{video_id}] ERROR saving data: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
    
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {successful_videos}")
    print(f"Failed: {failed_videos}")
    print(f"Success rate: {(successful_videos/len(video_files)*100):.1f}%")
    print("")
    print("Dataset Statistics:")
    print(f"  Total original frames: {total_original_frames}")
    print(f"  After removing shaking/licking_shaking: {total_after_removal}")
    print(f"  After balancing: {total_after_balancing}")
    print(f"  Reduction ratio: {total_after_balancing/total_original_frames:.2%}")
    
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
    print("\nNext steps:")
    print(f"  1. Verify the output in: {output_dir}")
    print(f"  2. Generate FACT config file")
    print(f"  3. Train the model")


if __name__ == '__main__':
    main()