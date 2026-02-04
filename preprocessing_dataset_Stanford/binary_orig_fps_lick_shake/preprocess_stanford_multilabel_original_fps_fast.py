#!/usr/bin/env python3
"""
Multi-label Preprocessing script for Stanford dataset to FACT format
Multi-label classification: licking AND shaking (independent binary outputs)

KEY FEATURES:
- Uses ORIGINAL FPS instead of downsampling
- Multi-label: Each second can have licking=0/1 AND shaking=0/1 independently
- No balancing - keeps dataset as-is (unbalanced)
- Outputs two separate sigmoid predictions instead of softmax
- Background is implicit (when both licking=0 and shaking=0)

FAST VERSION:
- Uses direct NumPy operations instead of PIL/Tensor transforms
- ~10x faster frame reading compared to original
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
import time

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
except ImportError:
    print("Error: PyTorch is required")
    sys.exit(1)

# Import I3D model
sys.path.insert(0, str(Path(__file__).parent.parent))
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
    
    FAST VERSION:
    - Uses direct NumPy operations instead of PIL/Tensor transforms
    - ~10x faster frame reading
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    clip_len = int(round(fps)) if fps > 0 else 25
    num_seconds = int(np.ceil(duration))
    
    print(f"      Opening video file...", flush=True)
    print(f"", flush=True)
    print(f"      ===================================================", flush=True)
    print(f"      VIDEO PROPERTIES", flush=True)
    print(f"      ===================================================", flush=True)
    print(f"      FPS detected: {fps:.2f}", flush=True)
    print(f"      Clip length (auto): {clip_len} frames", flush=True)
    print(f"      Total frames: {total_frames}", flush=True)
    print(f"      Duration: {duration:.2f} seconds", flush=True)
    print(f"      Expected seconds: {num_seconds}", flush=True)
    print(f"      Frames per second: ~{total_frames/num_seconds:.1f}", flush=True)
    print(f"      ===================================================", flush=True)
    print(f"", flush=True)
    print(f"      Using ORIGINAL FPS with per-second aggregation", flush=True)
    print(f"      Aggregation method: {aggregation}", flush=True)
    sys.stdout.flush()
    
    # Read all frames
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      READING ALL FRAMES FROM VIDEO (FAST MODE)", flush=True)
    print(f"      ---------------------------------------------------", flush=True)
    sys.stdout.flush()
    
    frames_by_second = [[] for _ in range(num_seconds)]
    frame_count = 0
    missed_frames = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 200
    MAX_PROCESSING_TIME = 7200
    start_time = time.time()
    
    # FAST: Use NumPy arrays directly instead of PIL/Tensor transforms
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > MAX_PROCESSING_TIME:
            print(f"      WARNING: Processing timeout after {elapsed:.1f}s", flush=True)
            print(f"      Breaking loop with {frame_count} frames read", flush=True)
            break
        
        ret, frame = cap.read()
        if not ret:
            missed_frames += 1
            consecutive_failures += 1
            
            # Check consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"      WARNING: {consecutive_failures} consecutive frame read failures", flush=True)
                print(f"      Breaking loop at frame {frame_count}/{total_frames}", flush=True)
                print(f"      Frames successfully read: {frame_count}", flush=True)
                print(f"      Total missed frames: {missed_frames}", flush=True)
                break
            
            # Original exit condition
            if frame_count >= total_frames:
                break
            continue
        
        # Reset consecutive failures on successful read
        consecutive_failures = 0
        
        # FAST: Direct NumPy operations (no PIL/Tensor conversions)
        frame = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_np = frame_rgb.astype(np.float32) / 255.0
        frame_np = (frame_np - mean) / std
        
        second_idx = int(frame_count / fps) if fps > 0 else 0
        if second_idx < num_seconds:
            frames_by_second[second_idx].append(frame_np)
        
        frame_count += 1
        
        if frame_count % 500 == 0:
            print(f"      Progress: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)", flush=True)
            sys.stdout.flush()
    
    cap.release()
    
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      FRAME READING COMPLETE", flush=True)
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      Expected frames: {total_frames}", flush=True)
    print(f"      Frames read: {frame_count}", flush=True)
    print(f"      Frames missing: {missed_frames}", flush=True)
    if missed_frames > 0:
        print(f"      WARNING: {missed_frames} frames could not be read!", flush=True)
    if frame_count < total_frames:
        print(f"      WARNING: Read fewer frames than expected ({frame_count}/{total_frames})", flush=True)
        print(f"      This may be due to video corruption or codec issues", flush=True)
    print(f"      ---------------------------------------------------", flush=True)
    print(f"", flush=True)
    sys.stdout.flush()
    
    # Extract features per second
    all_features = []
    
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      EXTRACTING FEATURES PER SECOND", flush=True)
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      Total seconds to process: {num_seconds}", flush=True)
    sys.stdout.flush()
    
    with torch.no_grad():
        for sec_idx in range(num_seconds):
            frames = frames_by_second[sec_idx]
            
            if len(frames) == 0:
                dummy_clip = torch.zeros(1, 3, clip_len, 224, 224).to(device)
                if model_type == 'i3d':
                    dummy_feat = model.extract_features(dummy_clip)
                else:
                    dummy_feat = model(dummy_clip)
                feat_dim = dummy_feat.squeeze().shape[0]
                second_feature = np.zeros(feat_dim, dtype=np.float32)
                all_features.append(second_feature)
                continue
            
            clip_features = []
            
            if len(frames) < clip_len:
                while len(frames) < clip_len:
                    frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.float32))
                
                clip_array = np.stack(frames[:clip_len], axis=0)
                clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).unsqueeze(0)
                clip_tensor = clip_tensor.float().to(device)
                
                if model_type == 'i3d':
                    feat = model.extract_features(clip_tensor)
                else:
                    feat = model(clip_tensor)
                
                feat = feat.squeeze().cpu().numpy()
                clip_features.append(feat)
            else:
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
            
            if aggregation == 'mean':
                second_feature = np.mean(clip_features, axis=0)
            else:
                second_feature = np.max(clip_features, axis=0)
            
            all_features.append(second_feature)
            
            if (sec_idx + 1) % 100 == 0:
                print(f"        Second {sec_idx + 1}/{num_seconds}", flush=True)
                sys.stdout.flush()
    
    features = np.array(all_features, dtype=np.float32)
    
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      FEATURE EXTRACTION COMPLETE", flush=True)
    print(f"      ---------------------------------------------------", flush=True)
    print(f"      Final features shape: {features.shape}", flush=True)
    print(f"", flush=True)
    sys.stdout.flush()
    
    return features


def convert_labels_to_multilabel(label_file, num_seconds):
    """
    Convert CSV labels to multi-label format with licking and shaking.
    Each second gets TWO binary labels: [licking, shaking]
    
    Returns:
        numpy array of shape (num_seconds, 2) with values 0 or 1
    """
    df = pd.read_csv(label_file)
    
    # Initialize multi-label array: (num_seconds, 2)
    # Column 0: licking (0/1)
    # Column 1: shaking (0/1)
    multilabels = np.zeros((num_seconds, 2), dtype=np.int32)
    
    for _, row in df.iterrows():
        start_sec = int(row['start_sec'])
        end_sec = int(row['end_sec'])
        behavior = str(row['behavior']).strip().lower()
        
        # Determine which label to set
        if 'lick' in behavior:
            label_idx = 0
        elif 'shake' in behavior:
            label_idx = 1
        else:
            continue
        
        # Mark all seconds in this range
        for sec in range(start_sec, min(end_sec, num_seconds)):
            multilabels[sec, label_idx] = 1
    
    return multilabels


def create_mapping_file(output_file):
    """Create mapping.txt file for multi-label format"""
    with open(output_file, 'w') as f:
        f.write("0 licking\n")
        f.write("1 shaking\n")
    print(f"\nCreated mapping file: {output_file}")


def create_train_test_splits(video_ids, output_dir, train_ratio=0.8, n_splits=5, seed=42):
    """Create train/test split files"""
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    random.seed(seed)
    n_videos = len(video_ids)
    n_train = int(n_videos * train_ratio)
    
    print(f"\nCreating {n_splits} train/test splits...")
    print(f"  Train ratio: {train_ratio} ({n_train}/{n_videos} videos)")
    
    for split_idx in range(1, n_splits + 1):
        shuffled_ids = video_ids.copy()
        random.shuffle(shuffled_ids)
        
        train_ids = shuffled_ids[:n_train]
        test_ids = shuffled_ids[n_train:]
        
        train_file = splits_dir / f'train.split{split_idx}.bundle'
        test_file = splits_dir / f'test.split{split_idx}.bundle'
        
        with open(train_file, 'w') as f:
            for vid in train_ids:
                f.write(f"{vid}\n")
        
        with open(test_file, 'w') as f:
            for vid in test_ids:
                f.write(f"{vid}\n")
        
        print(f"  Split {split_idx}: {len(train_ids)} train, {len(test_ids)} test")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Stanford dataset for FACT (Multi-label: licking & shaking) - FAST VERSION')
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits (default: 42)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of CPU threads to use (default: 4)')
    
    args = parser.parse_args()
    
    set_cpu_threads(args.cpu_threads)
    
    print("=" * 80, flush=True)
    print("STANFORD MULTI-LABEL DATASET PREPROCESSING (FAST VERSION)", flush=True)
    print("Multi-label: Licking & Shaking (Independent Binary Outputs)", flush=True)
    print("=" * 80, flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"  Input videos: {args.input_videos}", flush=True)
    print(f"  Input labels: {args.input_labels}", flush=True)
    print(f"  Output directory: {args.output_dir}", flush=True)
    print(f"  Label format: Multi-label (licking AND shaking)", flush=True)
    print(f"  NO BALANCING - Dataset kept as-is", flush=True)
    print(f"  FAST MODE: Using NumPy instead of PIL/Tensor transforms", flush=True)
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
    
    # Setup device
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
        torch.set_num_threads(args.cpu_threads)
    
    # Load model
    model, model_type = load_i3d_model(device, weights_path=args.i3d_weights)
    
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
            print(f"[{video_id}] ERROR: Label file not found")
            failed_videos += 1
            continue
        
        print(f"[{video_id}] Found label file: {label_file.name}")
        
        # Extract video features
        try:
            print(f"[{video_id}] Step 1/3: Extracting features...", flush=True)
            features = extract_video_features_per_second(
                video_file, model, model_type, device,
                aggregation=args.aggregation
            )
            print(f"[{video_id}]   Features extracted: {features.shape}")
            
        except Exception as e:
            print(f"[{video_id}] ERROR extracting features: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Convert labels to multi-label format
        try:
            print(f"[{video_id}] Step 2/3: Converting to multi-label format...", flush=True)
            num_seconds = len(features)
            multilabels = convert_labels_to_multilabel(label_file, num_seconds)
            
            # Ensure same length
            if len(features) != len(multilabels):
                min_len = min(len(features), len(multilabels))
                features = features[:min_len]
                multilabels = multilabels[:min_len]
                print(f"[{video_id}]   Trimmed to {min_len} seconds")
            
        except Exception as e:
            print(f"[{video_id}] ERROR processing labels: {e}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
        
        # Save processed data
        try:
            print(f"[{video_id}] Step 3/3: Saving processed data...", flush=True)
            
            # Save features
            feature_file = features_dir / f"{video_id}.npy"
            np.save(feature_file, features)
            print(f"[{video_id}]   Saved features: {feature_file}")
            print(f"[{video_id}]     Shape: {features.shape}")
            
            # Save multi-label ground truth
            # Format: Each row has 2 values: [licking, shaking]
            gt_file = groundtruth_dir / f"{video_id}.npy"
            np.save(gt_file, multilabels)
            print(f"[{video_id}]   Saved multi-labels: {gt_file}")
            print(f"[{video_id}]     Shape: {multilabels.shape}")
            
            print(f"[{video_id}] SUCCESS")
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
    
    # Create mapping file
    mapping_file = output_dir / 'mapping.txt'
    create_mapping_file(mapping_file)
    
    # Create train/test splits
    create_train_test_splits(video_ids, output_dir,
                             train_ratio=args.train_ratio,
                             n_splits=args.n_splits,
                             seed=args.seed)
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"\nKey features:")
    print(f"  - Multi-label format: licking AND shaking (independent)")
    print(f"  - No balancing applied")
    print(f"  - Labels saved as .npy with shape (n_seconds, 2)")
    print(f"  - Ready for two-head sigmoid output model")
    print(f"  - FAST VERSION: ~10x faster frame reading")


if __name__ == '__main__':
    main()