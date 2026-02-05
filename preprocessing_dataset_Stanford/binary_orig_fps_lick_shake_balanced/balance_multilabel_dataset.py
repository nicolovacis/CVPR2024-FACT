#!/usr/bin/env python3
"""
Balance Multi-Label Dataset for FACT Training

This script:
1. Reuses existing extracted features (no re-processing!)
2. Analyzes class distribution (licking vs shaking)
3. Creates balanced dataset through intelligent frame/segment sampling
4. Maintains temporal coherence by keeping segments intact
5. Saves to new directory: preprocessed_data_multilabel_originalFPS_balanced

Usage:
    python balance_multilabel_dataset.py --input_dir /path/to/preprocessed_data_multilabel_originalFPS
"""

import os
import sys
import argparse
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random


def analyze_dataset(features_dir, groundtruth_dir, split_file):
    """Analyze the dataset to understand class distribution"""
    print("\n" + "="*80)
    print("ANALYZING DATASET")
    print("="*80)
    
    with open(split_file, 'r') as f:
        video_list = [v.strip() for v in f.read().split('\n') if v.strip()]
    
    stats = {
        'total_frames': 0,
        'licking_frames': 0,
        'shaking_frames': 0,
        'both_frames': 0,
        'background_frames': 0,
        'videos': defaultdict(lambda: {'total': 0, 'licking': 0, 'shaking': 0, 'both': 0, 'bg': 0})
    }
    
    for video in tqdm(video_list, desc="Analyzing videos"):
        gt_file = os.path.join(groundtruth_dir, f"{video}.npy")
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth not found for {video}")
            continue
        
        labels = np.load(gt_file)  # (T, 2)
        
        licking = labels[:, 0]
        shaking = labels[:, 1]
        
        stats['total_frames'] += len(labels)
        stats['licking_frames'] += licking.sum()
        stats['shaking_frames'] += shaking.sum()
        stats['both_frames'] += ((licking == 1) & (shaking == 1)).sum()
        stats['background_frames'] += ((licking == 0) & (shaking == 0)).sum()
        
        stats['videos'][video]['total'] = len(labels)
        stats['videos'][video]['licking'] = licking.sum()
        stats['videos'][video]['shaking'] = shaking.sum()
        stats['videos'][video]['both'] = ((licking == 1) & (shaking == 1)).sum()
        stats['videos'][video]['bg'] = ((licking == 0) & (shaking == 0)).sum()
    
    print(f"\nDataset Statistics:")
    print(f"  Total videos: {len(video_list)}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Licking frames: {stats['licking_frames']} ({100*stats['licking_frames']/stats['total_frames']:.2f}%)")
    print(f"  Shaking frames: {stats['shaking_frames']} ({100*stats['shaking_frames']/stats['total_frames']:.2f}%)")
    print(f"  Both frames: {stats['both_frames']} ({100*stats['both_frames']/stats['total_frames']:.2f}%)")
    print(f"  Background frames: {stats['background_frames']} ({100*stats['background_frames']/stats['total_frames']:.2f}%)")
    
    return stats


def find_segments(labels):
    """
    Find contiguous segments of each label type
    
    Returns:
        List of segments: [(start, end, label_type), ...]
        label_type: 'licking', 'shaking', 'both', 'background'
    """
    segments = []
    
    if len(labels) == 0:
        return segments
    
    # Determine label type for each frame
    label_types = []
    for i in range(len(labels)):
        lick = labels[i, 0]
        shake = labels[i, 1]
        
        if lick == 1 and shake == 1:
            label_types.append('both')
        elif lick == 1:
            label_types.append('licking')
        elif shake == 1:
            label_types.append('shaking')
        else:
            label_types.append('background')
    
    # Find contiguous segments
    current_type = label_types[0]
    start = 0
    
    for i in range(1, len(label_types)):
        if label_types[i] != current_type:
            segments.append((start, i, current_type))
            start = i
            current_type = label_types[i]
    
    # Add last segment
    segments.append((start, len(label_types), current_type))
    
    return segments


def balance_video(features, labels, target_distribution, strategy='oversample_rare'):
    """
    Balance a single video by sampling segments
    
    Args:
        features: (T, D) features
        labels: (T, 2) labels
        target_distribution: dict with target counts for each class
        strategy: 'oversample_rare' or 'undersample_common'
    
    Returns:
        balanced_features, balanced_labels
    """
    segments = find_segments(labels)
    
    # Group segments by type
    segments_by_type = defaultdict(list)
    for start, end, label_type in segments:
        segments_by_type[label_type].append((start, end))
    
    # Current counts
    current_counts = {
        'licking': labels[:, 0].sum(),
        'shaking': labels[:, 1].sum(),
        'background': ((labels[:, 0] == 0) & (labels[:, 1] == 0)).sum()
    }
    
    # Determine which segments to keep/repeat
    selected_segments = []
    
    if strategy == 'oversample_rare':
        # Keep all segments, repeat rare ones
        for label_type in ['licking', 'shaking', 'both', 'background']:
            segs = segments_by_type[label_type]
            
            if label_type == 'both':
                # Keep all "both" segments
                selected_segments.extend(segs)
            elif label_type == 'shaking':
                # Oversample shaking heavily (it's rarest)
                target_count = max(current_counts['licking'] // 2, len(segs) * 10)
                repeat_factor = max(1, target_count // max(1, len(segs)))
                selected_segments.extend(segs * repeat_factor)
            elif label_type == 'licking':
                # Keep all licking
                selected_segments.extend(segs)
            else:  # background
                # Downsample background to match licking
                target_count = current_counts['licking']
                if len(segs) > 0:
                    downsample_factor = max(0.1, target_count / max(1, current_counts['background']))
                    num_to_keep = max(1, int(len(segs) * downsample_factor))
                    selected_segments.extend(random.sample(segs, min(num_to_keep, len(segs))))
    
    elif strategy == 'undersample_common':
        # Downsample common classes to match rare ones
        target_shake = current_counts['shaking']
        target_count = max(target_shake * 3, 100)  # At least 3x shaking frames
        
        for label_type in ['shaking', 'both', 'licking', 'background']:
            segs = segments_by_type[label_type]
            
            if label_type in ['shaking', 'both']:
                # Keep all rare segments
                selected_segments.extend(segs)
            else:
                # Downsample common segments
                if len(segs) > 0:
                    num_to_keep = max(1, int(len(segs) * 0.3))  # Keep 30%
                    selected_segments.extend(random.sample(segs, min(num_to_keep, len(segs))))
    
    # Sort segments by start time to maintain temporal order
    selected_segments.sort()
    
    # Concatenate selected segments
    balanced_features = []
    balanced_labels = []
    
    for start, end in selected_segments:
        balanced_features.append(features[start:end])
        balanced_labels.append(labels[start:end])
    
    if len(balanced_features) == 0:
        # Fallback: return original if something went wrong
        return features, labels
    
    balanced_features = np.concatenate(balanced_features, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)
    
    return balanced_features, balanced_labels


def create_balanced_dataset(input_dir, output_dir, strategy='oversample_rare', seed=42):
    """
    Create balanced dataset from existing preprocessed data
    
    Args:
        input_dir: Path to preprocessed_data_multilabel_originalFPS
        output_dir: Path to output directory (will be created)
        strategy: 'oversample_rare' or 'undersample_common'
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    features_dir = os.path.join(input_dir, 'features')
    groundtruth_dir = os.path.join(input_dir, 'groundTruth')
    splits_dir = os.path.join(input_dir, 'splits')
    mapping_file = os.path.join(input_dir, 'mapping.txt')
    
    # Verify input paths
    assert os.path.exists(features_dir), f"Features directory not found: {features_dir}"
    assert os.path.exists(groundtruth_dir), f"Ground truth directory not found: {groundtruth_dir}"
    assert os.path.exists(splits_dir), f"Splits directory not found: {splits_dir}"
    assert os.path.exists(mapping_file), f"Mapping file not found: {mapping_file}"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    out_features = os.path.join(output_dir, 'features')
    out_groundtruth = os.path.join(output_dir, 'groundTruth')
    out_splits = os.path.join(output_dir, 'splits')
    
    os.makedirs(out_features, exist_ok=True)
    os.makedirs(out_groundtruth, exist_ok=True)
    os.makedirs(out_splits, exist_ok=True)
    
    # Copy mapping file
    shutil.copy(mapping_file, os.path.join(output_dir, 'mapping.txt'))
    print(f"✓ Copied mapping.txt")
    
    # Process each split
    for split_file in os.listdir(splits_dir):
        if not split_file.endswith('.bundle'):
            continue
        
        split_path = os.path.join(splits_dir, split_file)
        split_name = split_file.replace('.bundle', '')
        
        print(f"\n{'='*80}")
        print(f"Processing split: {split_name}")
        print(f"{'='*80}")
        
        # Analyze this split
        stats = analyze_dataset(features_dir, groundtruth_dir, split_path)
        
        # Load video list
        with open(split_path, 'r') as f:
            video_list = [v.strip() for v in f.read().split('\n') if v.strip()]
        
        # Process each video
        print(f"\nBalancing videos using strategy: {strategy}")
        balanced_stats = {
            'total_frames': 0,
            'licking_frames': 0,
            'shaking_frames': 0,
            'both_frames': 0,
            'background_frames': 0
        }
        
        for video in tqdm(video_list, desc="Balancing videos"):
            # Load features and labels
            feat_file = os.path.join(features_dir, f"{video}.npy")
            gt_file = os.path.join(groundtruth_dir, f"{video}.npy")
            
            if not os.path.exists(feat_file) or not os.path.exists(gt_file):
                print(f"Warning: Missing files for {video}, skipping")
                continue
            
            features = np.load(feat_file)
            labels = np.load(gt_file)
            
            # Balance this video
            balanced_features, balanced_labels = balance_video(
                features, labels, target_distribution=None, strategy=strategy
            )
            
            # Save balanced version
            np.save(os.path.join(out_features, f"{video}.npy"), balanced_features)
            np.save(os.path.join(out_groundtruth, f"{video}.npy"), balanced_labels)
            
            # Update stats
            balanced_stats['total_frames'] += len(balanced_labels)
            balanced_stats['licking_frames'] += balanced_labels[:, 0].sum()
            balanced_stats['shaking_frames'] += balanced_labels[:, 1].sum()
            balanced_stats['both_frames'] += ((balanced_labels[:, 0] == 1) & 
                                              (balanced_labels[:, 1] == 1)).sum()
            balanced_stats['background_frames'] += ((balanced_labels[:, 0] == 0) & 
                                                    (balanced_labels[:, 1] == 0)).sum()
        
        # Copy split file
        shutil.copy(split_path, os.path.join(out_splits, split_file))
        print(f"✓ Copied {split_file}")
        
        # Print balanced statistics
        print(f"\n{'='*80}")
        print(f"BALANCED DATASET STATISTICS - {split_name}")
        print(f"{'='*80}")
        print(f"Original:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Licking: {stats['licking_frames']} ({100*stats['licking_frames']/stats['total_frames']:.2f}%)")
        print(f"  Shaking: {stats['shaking_frames']} ({100*stats['shaking_frames']/stats['total_frames']:.2f}%)")
        print(f"  Both: {stats['both_frames']} ({100*stats['both_frames']/stats['total_frames']:.2f}%)")
        print(f"  Background: {stats['background_frames']} ({100*stats['background_frames']/stats['total_frames']:.2f}%)")
        
        print(f"\nBalanced:")
        print(f"  Total frames: {balanced_stats['total_frames']}")
        print(f"  Licking: {balanced_stats['licking_frames']} ({100*balanced_stats['licking_frames']/balanced_stats['total_frames']:.2f}%)")
        print(f"  Shaking: {balanced_stats['shaking_frames']} ({100*balanced_stats['shaking_frames']/balanced_stats['total_frames']:.2f}%)")
        print(f"  Both: {balanced_stats['both_frames']} ({100*balanced_stats['both_frames']/balanced_stats['total_frames']:.2f}%)")
        print(f"  Background: {balanced_stats['background_frames']} ({100*balanced_stats['background_frames']/balanced_stats['total_frames']:.2f}%)")
        
        print(f"\nImprovement:")
        orig_ratio = stats['licking_frames'] / max(1, stats['shaking_frames'])
        balanced_ratio = balanced_stats['licking_frames'] / max(1, balanced_stats['shaking_frames'])
        print(f"  Licking/Shaking ratio: {orig_ratio:.1f}:1 → {balanced_ratio:.1f}:1")
    
    print(f"\n{'='*80}")
    print(f"✓ BALANCED DATASET CREATED")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo use this balanced dataset, update your config file:")
    print(f"  feature_path: {out_features}")
    print(f"  groundTruth_path: {out_groundtruth}")
    print(f"  split_path: {out_splits}")


def main():
    parser = argparse.ArgumentParser(
        description='Balance multi-label dataset by reusing existing features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: oversample rare classes (recommended)
  python balance_multilabel_dataset.py \\
    --input_dir /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS
  
  # Alternative: undersample common classes (reduces dataset size)
  python balance_multilabel_dataset.py \\
    --input_dir /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS \\
    --strategy undersample_common
  
  # Custom output directory
  python balance_multilabel_dataset.py \\
    --input_dir /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS \\
    --output_dir /data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/my_balanced_dataset
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to preprocessed_data_multilabel_originalFPS directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: input_dir + "_balanced")'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['oversample_rare', 'undersample_common'],
        default='oversample_rare',
        help='Balancing strategy (default: oversample_rare)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip('/') + '_balanced'
    
    print("="*80)
    print("DATASET BALANCING SCRIPT")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Random seed: {args.seed}")
    
    # Create balanced dataset
    create_balanced_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        seed=args.seed
    )
    
    print("\n✓ Done!")
    print("\nNext steps:")
    print("1. Update your config file to point to the balanced dataset")
    print("2. Run training with the balanced dataset")
    print("3. Compare results with unbalanced dataset + loss weighting")


if __name__ == '__main__':
    main()
