#!/usr/bin/env python3
"""
Preprocessing script for Stanford dataset to FACT format
Extracts I3D features from videos and converts CSV labels to FACT-compatible format
"""

import os
import sys
import argparse

# Set CPU thread limits early (before importing numpy/torch)
# This helps optimize performance when using CPU or when GPU is overloaded
def set_cpu_threads(num_threads=4):
    """Set number of CPU threads for various libraries"""
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    print(f"Set CPU thread limits to {num_threads} threads")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from datetime import datetime

# I3D model imports
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
    
    # Try to load actual I3D model
    if I3D_AVAILABLE and weights_path is not None:
        try:
            print(f"Loading I3D model from weights: {weights_path}")
            model = load_i3d_from_file(weights_path=weights_path, device=device)
            print(f"I3D model loaded successfully on {device}")
            return model, 'i3d'
        except Exception as e:
            print(f"Error loading I3D model: {e}")
            print("Falling back to alternative feature extractor...")
    
    # Fallback to ResNet3D or other video models
    try:
        from torchvision.models.video import r3d_18
        print("Using R3D-18 as feature extractor")
        print("Note: For best results, use I3D with --i3d_weights argument")
        model = r3d_18(pretrained=True)
        # Remove final classification layer to get features
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
    Extract I3D features from a video file
    
    Args:
        video_path: Path to video file
        model: Pretrained model (I3D or R3D)
        model_type: Type of model ('i3d' or 'r3d')
        device: torch device
        target_fps: Target FPS for feature extraction
        clip_len: Number of frames per clip
        
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
    
    frames = []
    frame_idx = 0
    frames_read = 0
    
    print(f"      Reading and sampling frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            # Resize frame to model input size (224x224)
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frames_read += 1
        
        frame_idx += 1
        
        # Progress indicator for long videos
        if frame_idx % 500 == 0:
            print(f"        Processed {frame_idx}/{total_frames} frames ({frames_read} sampled)...")
    
    cap.release()
    
    print(f"      Total frames read: {frame_idx}")
    print(f"      Total frames sampled: {len(frames)}")
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")
    
    # Convert frames to tensor
    print(f"      Converting frames to tensor and normalizing...")
    frames = np.array(frames, dtype=np.float32) / 255.0
    
    # Normalize using ImageNet stats (keep as float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    
    print(f"      Frames array shape: {frames.shape}")
    print(f"      Frames dtype: {frames.dtype}")
    print(f"      Extracting features using {model_type.upper()} model...")
    print(f"      Clip length: {clip_len} frames")
    
    # Extract features in batches of clips
    features = []
    
    with torch.no_grad():
        # Process video in sliding window clips
        stride = max(1, clip_len // 2)  # 50% overlap
        print(f"      Stride: {stride} frames (50% overlap)")
        
        num_clips = (len(frames) + stride - 1) // stride
        print(f"      Expected number of clips: {num_clips}")
        
        clip_count = 0
        for i in range(0, len(frames), stride):
            # Get clip
            end_idx = min(i + clip_len, len(frames))
            clip = frames[i:end_idx]
            
            # Pad if necessary (ensure padding is also float32)
            if len(clip) < clip_len:
                pad_len = clip_len - len(clip)
                clip = np.concatenate([clip, np.zeros((pad_len, 224, 224, 3), dtype=np.float32)], axis=0)
            
            # Convert to tensor: (1, C, T, H, W) and ensure float32
            clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
            clip_tensor = clip_tensor.float()  # Explicitly convert to float32
            clip_tensor = clip_tensor.to(device)
            
            # Extract features
            if model_type == 'i3d':
                # I3D model
                feat = model.extract_features(clip_tensor)
            else:
                # R3D or other models
                feat = model(clip_tensor)
            
            feat = feat.squeeze().cpu().numpy()
            
            # Replicate feature for each frame in this clip
            num_frames_in_clip = min(stride, end_idx - i)
            for _ in range(num_frames_in_clip):
                features.append(feat)
            
            clip_count += 1
            if clip_count % 10 == 0:
                print(f"        Processed {clip_count}/{num_clips} clips...")
            
            if end_idx >= len(frames):
                break
    
    # Trim to match original frame count
    features = np.array(features[:len(frames)])
    
    print(f"      Feature extraction complete!")
    print(f"      Final feature shape: {features.shape}")
    print(f"      Feature dtype: {features.dtype}")
    
    return features


def parse_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS) to seconds"""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def convert_labels_to_frame_level(csv_path, video_duration, fps=25):
    """
    Convert CSV labels to frame-level labels
    
    Args:
        csv_path: Path to CSV file with labels
        video_duration: Duration of video in seconds
        fps: Frames per second
        
    Returns:
        list of action labels for each frame
    """
    print(f"      Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      CSV rows: {len(df)}")
    print(f"      CSV columns: {list(df.columns)}")
    
    # Get total number of frames
    total_frames = int(video_duration * fps)
    print(f"      Video duration: {video_duration:.2f}s")
    print(f"      Target FPS: {fps}")
    print(f"      Total frames to generate: {total_frames}")
    
    # Initialize all frames as background
    frame_labels = ['background'] * total_frames
    
    # Track action statistics
    licking_count = 0
    shaking_count = 0
    both_count = 0
    background_count = 0
    
    # Process each row in the CSV
    print(f"      Processing CSV rows...")
    for idx, row in df.iterrows():
        start_sec = parse_time_to_seconds(row['start_time'])
        end_sec = parse_time_to_seconds(row['end_time'])
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        licking_val = row['licking']
        shaking_val = row['shaking']
        
        # Determine the action for this segment
        # 4 classes: background, licking, shaking, licking_shaking
        if licking_val == 1.0 and shaking_val == 1.0:
            action = 'licking_shaking'
            both_count += 1
            if idx < 5 or both_count <= 3:  # Show first few occurrences
                print(f"        Row {idx}: {row['start_time']}-{row['end_time']} -> licking_shaking")
        elif licking_val == 1.0:
            action = 'licking'
            licking_count += 1
        elif shaking_val == 1.0:
            action = 'shaking'
            shaking_count += 1
        else:
            action = 'background'
            background_count += 1
        
        # Assign action to frames
        for frame_idx in range(start_frame, min(end_frame, total_frames)):
            frame_labels[frame_idx] = action
    
    print(f"      CSV processing complete:")
    print(f"        Segments with licking only: {licking_count}")
    print(f"        Segments with shaking only: {shaking_count}")
    print(f"        Segments with licking_shaking: {both_count}")
    print(f"        Segments with background: {background_count}")
    
    return frame_labels


def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def create_mapping_file(output_path):
    """Create mapping.txt file with action class mappings"""
    mapping = {
        0: 'background',
        1: 'licking',
        2: 'shaking',
        3: 'licking_shaking'
    }
    
    with open(output_path, 'w') as f:
        for idx, label in mapping.items():
            f.write(f"{idx} {label}\n")
    
    print(f"Created mapping file: {output_path}")
    print(f"  Classes: {len(mapping)}")
    for idx, label in mapping.items():
        print(f"    {idx}: {label}")
    
    return mapping


def create_train_test_splits(video_ids, output_dir, train_ratio=0.8, n_splits=1):
    """Create train/test split files"""
    splits_dir = Path(output_dir) / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)  # For reproducibility
    
    for split_num in range(1, n_splits + 1):
        # Shuffle and split
        shuffled_ids = video_ids.copy()
        np.random.shuffle(shuffled_ids)
        
        split_idx = int(len(shuffled_ids) * train_ratio)
        train_ids = shuffled_ids[:split_idx]
        test_ids = shuffled_ids[split_idx:]
        
        # Write train split
        train_file = splits_dir / f'train.split{split_num}.bundle'
        with open(train_file, 'w') as f:
            for vid_id in train_ids:
                f.write(f"{vid_id}\n")
        
        # Write test split
        test_file = splits_dir / f'test.split{split_num}.bundle'
        with open(test_file, 'w') as f:
            for vid_id in test_ids:
                f.write(f"{vid_id}\n")
        
        print(f"Created split {split_num}: {len(train_ids)} train, {len(test_ids)} test")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Stanford dataset to FACT format')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Path to directory containing video files (videos_cut)')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Path to directory containing CSV label files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for preprocessed dataset')
    parser.add_argument('--i3d_weights', type=str, default=None,
                        help='Path to pretrained I3D weights (optional, for best results)')
    parser.add_argument('--target_fps', type=int, default=25,
                        help='Target FPS for feature extraction (default: 25)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--n_splits', type=int, default=1,
                        help='Number of train/test splits to create (default: 1)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU usage instead of GPU')
    parser.add_argument('--clip_length', type=int, default=64,
                        help='Number of frames per clip for I3D (default: 64)')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of CPU threads for parallel processing (default: 4)')
    
    args = parser.parse_args()
    
    # Set CPU thread limits for optimization
    set_cpu_threads(args.cpu_threads)
    
    # Also set OpenCV threads
    cv2.setNumThreads(args.cpu_threads)
    print(f"OpenCV threads set to {args.cpu_threads}")
    
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
    print("Stanford Dataset to FACT Format Preprocessing")
    print("=" * 80)
    print(f"Videos directory: {videos_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Clip length: {args.clip_length}")
    if args.i3d_weights:
        print(f"I3D weights: {args.i3d_weights}")
    print("=" * 80)
    
    # Setup device
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        # Set PyTorch CPU threads for better performance
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
    
    for video_idx, video_file in enumerate(tqdm(video_files, desc="Processing videos"), 1):
        video_id = video_file.stem  # e.g., 'ID01'
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
            print(f"[{video_id}] Step 1/3: Reading video file...")
            print(f"[{video_id}]   Video path: {video_file}")
            
            # Get video info first
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            print(f"[{video_id}]   Video FPS: {fps:.2f}")
            print(f"[{video_id}]   Total frames: {total_frames}")
            print(f"[{video_id}]   Duration: {duration:.2f} seconds")
            print(f"[{video_id}]   Target FPS: {args.target_fps}")
            
            expected_output_frames = int(duration * args.target_fps)
            print(f"[{video_id}]   Expected output frames: {expected_output_frames}")
            
            print(f"[{video_id}] Step 2/3: Extracting I3D features...")
            print(f"[{video_id}]   Using model: {model_type.upper()}")
            print(f"[{video_id}]   Clip length: {args.clip_length} frames")
            
            features = extract_video_features(video_file, model, model_type, device, 
                                              target_fps=args.target_fps,
                                              clip_len=args.clip_length)
            
            print(f"[{video_id}]   Features extracted: {features.shape}")
            print(f"[{video_id}]   Feature dimension: {features.shape[1]}")
            print(f"[{video_id}]   Feature size: {features.nbytes / 1024 / 1024:.2f} MB")
            
            # Save features
            feature_file = features_dir / f"{video_id}.npy"
            np.save(feature_file, features)
            print(f"[{video_id}]   Saved to: {feature_file}")
            
        except Exception as e:
            print(f"[{video_id}] ERROR extracting features: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Convert labels to frame-level
        try:
            print(f"[{video_id}] Step 3/3: Converting labels to frame-level...")
            print(f"[{video_id}]   Reading CSV: {label_file.name}")
            
            video_duration = get_video_duration(video_file)
            print(f"[{video_id}]   Video duration: {video_duration:.2f} seconds")
            
            frame_labels = convert_labels_to_frame_level(label_file, video_duration, 
                                                         fps=args.target_fps)
            
            print(f"[{video_id}]   Labels created: {len(frame_labels)} frames")
            
            # Count action occurrences
            from collections import Counter
            label_counts = Counter(frame_labels)
            print(f"[{video_id}]   Label distribution:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(frame_labels)) * 100
                print(f"[{video_id}]     - {label}: {count} frames ({percentage:.1f}%)")
            
            # Ensure features and labels match in length
            if len(features) != len(frame_labels):
                min_len = min(len(features), len(frame_labels))
                print(f"[{video_id}]   WARNING: Length mismatch!")
                print(f"[{video_id}]     Features: {len(features)} frames")
                print(f"[{video_id}]     Labels: {len(frame_labels)} frames")
                print(f"[{video_id}]     Trimming both to: {min_len} frames")
                frame_labels = frame_labels[:min_len]
                features = features[:min_len]
                # Re-save trimmed features
                np.save(feature_file, features)
                print(f"[{video_id}]     Updated feature file")
            else:
                print(f"[{video_id}]   Features and labels aligned: {len(features)} frames")
            
            # Save ground truth labels
            gt_file = groundtruth_dir / f"{video_id}.txt"
            with open(gt_file, 'w') as f:
                for label in frame_labels:
                    f.write(f"{label}\n")
            print(f"[{video_id}]   Saved labels to: {gt_file}")
            
            print(f"[{video_id}] ✓ SUCCESS - Video processed successfully")
            successful_videos += 1
            
        except Exception as e:
            print(f"[{video_id}] ERROR processing labels: {e}")
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
    
    print("\n" + "=" * 80)
    print("Creating dataset metadata files...")
    print("=" * 80)
    
    # Create mapping file
    mapping_file = output_dir / 'mapping.txt'
    create_mapping_file(mapping_file)
    
    # Create train/test splits
    create_train_test_splits(video_ids, output_dir, 
                             train_ratio=args.train_ratio,
                             n_splits=args.n_splits)
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── features/        ({len(list(features_dir.glob('*.npy')))} .npy files)")
    print(f"  ├── groundTruth/     ({len(list(groundtruth_dir.glob('*.txt')))} .txt files)")
    print(f"  ├── mapping.txt")
    print(f"  └── splits/          ({args.n_splits} train/test splits)")
    print("\nYou can now use this dataset with FACT by running:")
    print(f"  python utils/gen_config.py --dataset_path {output_dir} \\")
    print(f"      --dataset_name stanford --output_config configs/stanford.yaml \\")
    print(f"      --base_config configs/breakfast.yaml")


if __name__ == '__main__':
    main()