#!/usr/bin/env python3
"""
Preprocessing script for CaiLab dataset to FACT format
Extracts I3D features from videos and converts Excel annotations to FACT-compatible format
"""

import os
import sys
import argparse

# Set CPU thread limits early
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
from datetime import datetime, timedelta
import re

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
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    
    print(f"      Frames array shape: {frames.shape}")
    print(f"      Extracting features using {model_type.upper()} model...")
    
    # Extract features in batches of clips
    features = []
    
    with torch.no_grad():
        stride = max(1, clip_len // 2)  # 50% overlap
        print(f"      Stride: {stride} frames (50% overlap)")
        
        num_clips = (len(frames) + stride - 1) // stride
        print(f"      Expected number of clips: {num_clips}")
        
        clip_count = 0
        for i in range(0, len(frames), stride):
            end_idx = min(i + clip_len, len(frames))
            clip = frames[i:end_idx]
            
            # Pad if necessary
            if len(clip) < clip_len:
                pad_len = clip_len - len(clip)
                clip = np.concatenate([clip, np.zeros((pad_len, 224, 224, 3), dtype=np.float32)], axis=0)
            
            # Convert to tensor: (1, C, T, H, W)
            clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
            clip_tensor = clip_tensor.float()
            clip_tensor = clip_tensor.to(device)
            
            # Extract features
            if model_type == 'i3d':
                feat = model.extract_features(clip_tensor)
            else:
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
    print(f"      Final features shape: {features.shape}")
    
    return features


def parse_time_string(time_str):
    """
    Parse time string from Excel (e.g., '9:37' or '0:05')
    Returns time in seconds
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    # Handle different time formats
    time_str = str(time_str).strip()
    
    # Format: MM:SS or M:SS
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    
    return None


def load_cailab_annotations(excel_path):
    """
    Load annotations from CaiLab Excel file
    
    Returns:
        Dictionary mapping animal_id -> {trial_start_time, events}
    """
    print(f"Loading annotations from: {excel_path}")
    
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    print(f"Excel shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    annotations = {}
    animal_trial_starts = {}  # Track trial start times per animal
    
    # Process each row
    for idx, row in df.iterrows():
        animal_id = str(row.iloc[0]).strip()  # First column is animal ID
        
        # Skip header rows or empty rows
        if pd.isna(animal_id) or animal_id == '' or 'cKO' not in animal_id:
            continue
        
        # Skip cKO-B as it has no labels
        if animal_id == 'cKO-B':
            print(f"Skipping {animal_id} - no labels")
            continue
        
        # Get trial number (second column)
        trial_num = row.iloc[1] if not pd.isna(row.iloc[1]) else 1
        
        # Create unique video identifier
        video_id = f"{animal_id}_trial{int(trial_num)}"
        
        if video_id not in annotations:
            annotations[video_id] = {
                'animal_id': animal_id,
                'trial_num': int(trial_num),
                'trial_start_time': None,  # Will be set from first row with "start" marker
                'events': []
            }
        
        # Check if this is a "start" marker row (indicates trial start time)
        # This is typically in the first row for each animal with a special marker
        # We need to check the actual Excel structure - for now, we'll look for:
        # - A row with just animal_id and trial_num but no behavior event, OR
        # - A specific column that marks trial start time
        
        # Parse annotation columns (start_time, duration, note)
        # Column layout: animal_id, trial_num, start_time, duration, note
        start_time_col = 2  # Third column
        duration_col = 3    # Fourth column  
        note_col = 4        # Fifth column
        
        if len(row) > start_time_col:
            start_time_str = row.iloc[start_time_col]
            duration_str = row.iloc[duration_col] if len(row) > duration_col else None
            note_str = row.iloc[note_col] if len(row) > note_col else None
            
            start_time = parse_time_string(start_time_str)
            
            # Check if this is a trial start marker (has start_time but no duration or is marked as "start")
            is_trial_start = False
            if note_str and not pd.isna(note_str):
                note_lower = str(note_str).lower()
                if 'start' in note_lower or 'begin' in note_lower or 'trial' in note_lower:
                    is_trial_start = True
            
            # If no duration and has start_time, might be trial start marker
            if start_time is not None and (pd.isna(duration_str) or duration_str == '' or duration_str == 0):
                is_trial_start = True
            
            if is_trial_start and start_time is not None:
                # This is the trial start time
                if annotations[video_id]['trial_start_time'] is None:
                    annotations[video_id]['trial_start_time'] = start_time
                    print(f"  {video_id}: Trial starts at {start_time}s ({start_time//60}:{start_time%60:02d})")
                continue
            
            # Regular behavior annotation
            duration = float(duration_str) if not pd.isna(duration_str) and duration_str != '' else 0
            
            if start_time is not None and duration > 0:
                annotations[video_id]['events'].append({
                    'start_time': start_time,
                    'duration': duration,
                    'end_time': start_time + duration,
                    'behavior': 'scratching',  # Default behavior
                    'note': str(note_str) if not pd.isna(note_str) else ''
                })
    
    print(f"\nLoaded annotations for {len(annotations)} video IDs")
    for video_id, data in list(annotations.items())[:5]:
        trial_start = data['trial_start_time']
        trial_start_str = f"{trial_start}s" if trial_start else "NOT SET"
        print(f"  {video_id}: trial_start={trial_start_str}, {len(data['events'])} events")
    
    return annotations


def cut_video_from_start_time(input_video_path, output_video_path, start_time_seconds):
    """
    Cut video from start_time to end using ffmpeg
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        start_time_seconds: Start time in seconds
    """
    import subprocess
    
    print(f"      Cutting video from {start_time_seconds}s to end...")
    
    cmd = [
        'ffmpeg',
        '-i', str(input_video_path),
        '-ss', str(start_time_seconds),
        '-c', 'copy',
        '-y',  # Overwrite output file
        str(output_video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"      FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to cut video: {result.stderr}")
    
    print(f"      Video cut successfully: {output_video_path}")


def downsample_background_frames(frame_labels, target_ratio=0.5, seed=42):
    """
    Downsample background frames to achieve target ratio of action/background
    
    Args:
        frame_labels: List of frame-level labels
        target_ratio: Target ratio of action frames (default 0.5 for 50-50)
        seed: Random seed for reproducibility
        
    Returns:
        List of indices to keep
    """
    np.random.seed(seed)
    
    # Find indices of action and background frames
    action_indices = [i for i, label in enumerate(frame_labels) if label != 'background']
    background_indices = [i for i, label in enumerate(frame_labels) if label == 'background']
    
    num_action = len(action_indices)
    num_background = len(background_indices)
    
    print(f"      Original: {num_action} action frames, {num_background} background frames")
    
    if num_action == 0:
        print(f"      WARNING: No action frames found, keeping all frames")
        return list(range(len(frame_labels)))
    
    # Calculate how many background frames to keep
    # If target_ratio = 0.5, then: action / (action + background_keep) = 0.5
    # So: background_keep = action / target_ratio - action
    target_background = int(num_action * (1 - target_ratio) / target_ratio)
    
    if target_background >= num_background:
        # Already have fewer background frames than needed
        print(f"      Already balanced: keeping all {num_background} background frames")
        return list(range(len(frame_labels)))
    
    # Randomly sample background frames to keep
    background_keep = sorted(np.random.choice(background_indices, target_background, replace=False))
    
    # Combine action and sampled background indices
    keep_indices = sorted(action_indices + background_keep)
    
    action_ratio = num_action / len(keep_indices) * 100
    print(f"      Downsampled: {num_action} action, {len(background_keep)} background (action: {action_ratio:.1f}%)")
    
    return keep_indices


def convert_labels_to_frame_level(events, video_duration_seconds, fps=25):
    """
    Convert event-based labels to frame-level labels
    
    Args:
        events: List of event dictionaries with start_time, duration, behavior
        video_duration_seconds: Total video duration in seconds
        fps: Target FPS
        
    Returns:
        List of frame-level labels
    """
    num_frames = int(video_duration_seconds * fps)
    frame_labels = ['background'] * num_frames
    
    for event in events:
        start_frame = int(event['start_time'] * fps)
        end_frame = int(event['end_time'] * fps)
        behavior = event['behavior']
        
        # Clamp to valid frame range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames))
        
        for frame_idx in range(start_frame, end_frame):
            frame_labels[frame_idx] = behavior
    
    return frame_labels


def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration


def create_mapping_file(mapping_file):
    """Create mapping.txt file with action label mappings"""
    # For CaiLab dataset, we have: background and scratching
    mappings = [
        "0 background",
        "1 scratching"
    ]
    
    with open(mapping_file, 'w') as f:
        for mapping in mappings:
            f.write(f"{mapping}\n")
    
    print(f"Created mapping file: {mapping_file}")


def create_train_test_splits(video_ids, output_dir, train_ratio=0.8, n_splits=1):
    """Create train/test split files"""
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    for split_idx in range(n_splits):
        # Shuffle video IDs
        shuffled_ids = video_ids.copy()
        np.random.shuffle(shuffled_ids)
        
        # Split into train and test
        split_point = int(len(shuffled_ids) * train_ratio)
        train_ids = shuffled_ids[:split_point]
        test_ids = shuffled_ids[split_point:]
        
        # Save split files
        train_file = splits_dir / f'train{split_idx + 1}.split'
        test_file = splits_dir / f'test{split_idx + 1}.split'
        
        with open(train_file, 'w') as f:
            for vid in train_ids:
                f.write(f"{vid}.txt\n")
        
        with open(test_file, 'w') as f:
            for vid in test_ids:
                f.write(f"{vid}.txt\n")
        
        print(f"Split {split_idx + 1}: {len(train_ids)} train, {len(test_ids)} test")


def parse_video_filename(video_filename):
    """
    Parse video filename to extract animal_id and perspective
    Expected format: 2025-08-27_CQ_mouseA_down.mp4
    Returns: (animal_id, perspective, trial_num)
    """
    stem = Path(video_filename).stem
    
    # Match pattern: date_CQ_mouseX_perspective
    pattern = r'(\d{4}-\d{2}-\d{2})_CQ_(mouse[A-Z])_(down|front)'
    match = re.match(pattern, stem)
    
    if match:
        date, animal, perspective = match.groups()
        # Map mouseA -> cKO-A
        animal_id = f"cKO-{animal[-1]}"
        return animal_id, perspective, 1  # Default trial 1
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Preprocess CaiLab dataset to FACT format')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing split video files')
    parser.add_argument('--excel_path', type=str, required=True,
                        help='Path to Excel annotation file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed dataset')
    parser.add_argument('--target_fps', type=int, default=25,
                        help='Target FPS for feature extraction')
    parser.add_argument('--clip_length', type=int, default=16,
                        help='Number of frames per clip for I3D')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--n_splits', type=int, default=1,
                        help='Number of train/test splits')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of CPU threads to use')
    parser.add_argument('--i3d_weights', type=str, default=None,
                        help='Path to I3D weights file')
    parser.add_argument('--cut_videos', action='store_true', default=True,
                        help='Cut videos from trial start time (DEFAULT: True)')
    parser.add_argument('--no_cut_videos', dest='cut_videos', action='store_false',
                        help='Disable video cutting (NOT RECOMMENDED)')
    parser.add_argument('--balance_classes', action='store_true', default=True,
                        help='Downsample background frames to achieve 50-50 balance (DEFAULT: True)')
    parser.add_argument('--no_balance', dest='balance_classes', action='store_false',
                        help='Disable class balancing')
    
    args = parser.parse_args()
    
    # Set CPU threads early
    set_cpu_threads(args.cpu_threads)
    
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    features_dir = output_dir / 'features'
    groundtruth_dir = output_dir / 'groundTruth'
    cut_videos_dir = output_dir / 'videos_cut' if args.cut_videos else None
    
    features_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    if cut_videos_dir:
        cut_videos_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CaiLab Dataset to FACT Format Preprocessing")
    print("=" * 80)
    print(f"Videos directory: {videos_dir}")
    print(f"Excel annotations: {args.excel_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Clip length: {args.clip_length}")
    print(f"Cut videos from trial start: {args.cut_videos}")
    print(f"Balance classes (50-50): {args.balance_classes}")
    if args.i3d_weights:
        print(f"I3D weights: {args.i3d_weights}")
    print("=" * 80)
    
    # Load annotations
    annotations = load_cailab_annotations(args.excel_path)
    
    # Setup device
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
    
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
    
    for video_idx, video_file in enumerate(tqdm(video_files, desc="Processing videos"), 1):
        # Parse filename to get animal_id and perspective
        animal_id, perspective, trial_num = parse_video_filename(video_file.name)
        
        if animal_id is None:
            print(f"\n[{video_file.name}] ERROR: Could not parse filename")
            failed_videos += 1
            continue
        
        # Skip cKO-B (mouseB) as it has no labels
        if animal_id == 'cKO-B':
            print(f"\n[{video_file.name}] Skipping cKO-B (no labels)")
            continue
        
        # Create video ID: animal_perspective (e.g., cKO-A_down)
        video_id = f"{animal_id}_{perspective}"
        video_ids.append(video_id)
        
        print(f"\n{'='*80}")
        print(f"Processing video {video_idx}/{len(video_files)}: {video_id}")
        print(f"  File: {video_file.name}")
        print(f"  Animal: {animal_id}, Perspective: {perspective}")
        print(f"{'='*80}")
        
        # Find corresponding annotations
        annot_key = f"{animal_id}_trial{trial_num}"
        if annot_key not in annotations:
            print(f"[{video_id}] WARNING: No annotations found for {annot_key}")
            print(f"[{video_id}] Skipping this video")
            failed_videos += 1
            continue
        
        annot_data = annotations[annot_key]
        events = annot_data['events']
        trial_start_time = annot_data['trial_start_time']
        
        print(f"[{video_id}] Found {len(events)} behavior events")
        
        if trial_start_time is None:
            print(f"[{video_id}] ERROR: No trial start time found in annotations")
            print(f"[{video_id}] Cannot determine where to cut video. Skipping.")
            failed_videos += 1
            continue
        
        print(f"[{video_id}] Trial start time: {trial_start_time}s ({trial_start_time//60}:{trial_start_time%60:02d})")
        
        # Determine which video to process
        video_to_process = video_file
        
        # Cut video from trial start time
        if args.cut_videos:
            cut_video_path = cut_videos_dir / f"{video_id}.mp4"
            
            if cut_video_path.exists():
                print(f"[{video_id}] Using existing cut video: {cut_video_path}")
                video_to_process = cut_video_path
            else:
                try:
                    print(f"[{video_id}] Cutting video from {trial_start_time}s...")
                    cut_video_from_start_time(video_file, cut_video_path, trial_start_time)
                    video_to_process = cut_video_path
                    print(f"[{video_id}] Video cut successfully")
                except Exception as e:
                    print(f"[{video_id}] ERROR cutting video: {e}")
                    print(f"[{video_id}] Skipping this video")
                    failed_videos += 1
                    continue
            
            # Adjust event times relative to cut point (subtract trial_start_time)
            print(f"[{video_id}] Adjusting annotation timestamps relative to trial start...")
            adjusted_events = []
            for event in events:
                adj_start = event['start_time'] - trial_start_time
                adj_end = event['end_time'] - trial_start_time
                
                # Skip events that are before trial start (shouldn't happen but safety check)
                if adj_end <= 0:
                    print(f"[{video_id}]   Skipping event before trial start: {event}")
                    continue
                
                # Clamp start to 0 if it's slightly negative
                adj_start = max(0, adj_start)
                
                adjusted_events.append({
                    'start_time': adj_start,
                    'end_time': adj_end,
                    'duration': adj_end - adj_start,
                    'behavior': event['behavior'],
                    'note': event['note']
                })
                
                print(f"[{video_id}]   Adjusted: {event['start_time']}s -> {adj_start}s")
            
            events = adjusted_events
            print(f"[{video_id}] Using {len(events)} adjusted events")
        else:
            print(f"[{video_id}] WARNING: --cut_videos not enabled, labels may not align with video!")
            print(f"[{video_id}] Enable --cut_videos flag to properly align annotations")

        
        # Extract features
        try:
            print(f"[{video_id}] Step 1/2: Extracting I3D features...")
            
            features = extract_video_features(video_to_process, model, model_type, device,
                                             target_fps=args.target_fps,
                                             clip_len=args.clip_length)
            
            print(f"[{video_id}]   Features shape: {features.shape}")
            
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
            print(f"[{video_id}] Step 2/2: Creating frame-level labels...")
            
            video_duration = get_video_duration(video_to_process)
            print(f"[{video_id}]   Video duration: {video_duration:.2f} seconds")
            
            frame_labels = convert_labels_to_frame_level(events, video_duration,
                                                        fps=args.target_fps)
            
            print(f"[{video_id}]   Labels created: {len(frame_labels)} frames")
            
            # Count behavior occurrences BEFORE downsampling
            from collections import Counter
            label_counts = Counter(frame_labels)
            print(f"[{video_id}]   Label distribution (BEFORE downsampling):")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(frame_labels)) * 100
                print(f"[{video_id}]     - {label}: {count} frames ({percentage:.1f}%)")
            
            # Downsample background frames to achieve 50-50 balance
            if args.balance_classes:
                print(f"[{video_id}]   Downsampling background frames for 50-50 balance...")
                keep_indices = downsample_background_frames(frame_labels, target_ratio=0.5)
                
                # Apply downsampling to both features and labels
                features = features[keep_indices]
                frame_labels = [frame_labels[i] for i in keep_indices]
                
                # Count AFTER downsampling
                label_counts_after = Counter(frame_labels)
                print(f"[{video_id}]   Label distribution (AFTER downsampling):")
                for label, count in sorted(label_counts_after.items()):
                    percentage = (count / len(frame_labels)) * 100
                    print(f"[{video_id}]     - {label}: {count} frames ({percentage:.1f}%)")
            
            # Ensure features and labels match
            if len(features) != len(frame_labels):
                min_len = min(len(features), len(frame_labels))
                print(f"[{video_id}]   WARNING: Length mismatch!")
                print(f"[{video_id}]     Features: {len(features)} frames")
                print(f"[{video_id}]     Labels: {len(frame_labels)} frames")
                print(f"[{video_id}]     Trimming both to: {min_len} frames")
                frame_labels = frame_labels[:min_len]
                features = features[:min_len]
            
            print(f"[{video_id}]   Final: Features and labels aligned: {len(features)} frames")
            
            # Save features
            feature_file = features_dir / f"{video_id}.npy"
            np.save(feature_file, features)
            print(f"[{video_id}]   Saved features to: {feature_file}")
            
            # Save ground truth labels
            gt_file = groundtruth_dir / f"{video_id}.txt"
            with open(gt_file, 'w') as f:
                for label in frame_labels:
                    f.write(f"{label}\n")
            print(f"[{video_id}]   Saved labels to: {gt_file}")
            
            print(f"[{video_id}] SUCCESS")
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
    print(f"      --dataset_name cailab --output_config configs/cailab.yaml \\")
    print(f"      --base_config configs/breakfast.yaml")


if __name__ == '__main__':
    main()