#!/usr/bin/env python3
"""
Quick diagnostic script to identify where dataset loading hangs.
Run this before running the full training to identify issues.

Usage:
    python3 diagnose_dataset.py
"""

import numpy as np
import os
import time
import sys

# Paths from config
FEATURE_PATH = '/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS/features'
GT_PATH = '/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS/groundTruth'
MAPPING_FILE = '/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS/mapping.txt'
SPLIT_PATH = '/data-8tb/nvaci/dataset/dataset_Stanford/FACT_method/preprocessed_data_multilabel_originalFPS/splits'
TEST_SPLIT = os.path.join(SPLIT_PATH, 'test.split1.bundle')
TRAIN_SPLIT = os.path.join(SPLIT_PATH, 'train.split1.bundle')

def check_paths():
    """Check if all required paths exist"""
    print("="*80)
    print("STEP 1: Checking if all paths exist")
    print("="*80)
    
    paths_to_check = {
        'Feature directory': FEATURE_PATH,
        'Ground truth directory': GT_PATH,
        'Mapping file': MAPPING_FILE,
        'Split directory': SPLIT_PATH,
        'Test split file': TEST_SPLIT,
        'Train split file': TRAIN_SPLIT,
    }
    
    all_exist = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist

def load_mapping():
    """Load and verify mapping file"""
    print("="*80)
    print("STEP 2: Loading mapping file")
    print("="*80)
    
    try:
        start = time.time()
        with open(MAPPING_FILE, 'r') as f:
            content = f.read().split('\n')[:-1]
        elapsed = time.time() - start
        
        print(f"  ✓ Loaded in {elapsed:.3f}s")
        print(f"  Found {len(content)} labels:")
        for line in content:
            print(f"    {line}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return False

def load_split_files():
    """Load split files and get video lists"""
    print("="*80)
    print("STEP 3: Loading split files")
    print("="*80)
    
    try:
        # Test split
        start = time.time()
        with open(TEST_SPLIT, 'r') as f:
            test_videos = f.read().split('\n')[:-1]
        elapsed = time.time() - start
        print(f"  ✓ Test split loaded in {elapsed:.3f}s")
        print(f"    Found {len(test_videos)} videos")
        if test_videos:
            print(f"    First video: {test_videos[0]}")
        
        # Train split
        start = time.time()
        with open(TRAIN_SPLIT, 'r') as f:
            train_videos = f.read().split('\n')[:-1]
        elapsed = time.time() - start
        print(f"  ✓ Train split loaded in {elapsed:.3f}s")
        print(f"    Found {len(train_videos)} videos")
        if train_videos:
            print(f"    First video: {train_videos[0]}")
        
        print()
        return test_videos, train_videos
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return None, None

def test_load_video(video_name, verbose=True):
    """Test loading a single video"""
    if verbose:
        print(f"  Testing video: {video_name}")
    
    try:
        # Remove .txt extension if present
        if video_name.endswith('.txt'):
            video_name = video_name[:-4]
        
        # Load feature
        feature_file = os.path.join(FEATURE_PATH, video_name + '.npy')
        if verbose:
            print(f"    Loading feature: {feature_file}")
        
        if not os.path.exists(feature_file):
            print(f"    ✗ Feature file does not exist!")
            return False
        
        file_size = os.path.getsize(feature_file) / (1024*1024)  # MB
        if verbose:
            print(f"    Feature file size: {file_size:.2f} MB")
        
        start = time.time()
        feature = np.load(feature_file)
        elapsed = time.time() - start
        
        if verbose:
            print(f"    ✓ Feature loaded in {elapsed:.3f}s, shape: {feature.shape}, dtype: {feature.dtype}")
        
        # Load ground truth
        gt_file = os.path.join(GT_PATH, video_name + '.npy')
        if verbose:
            print(f"    Loading ground truth: {gt_file}")
        
        if not os.path.exists(gt_file):
            print(f"    ✗ Ground truth file does not exist!")
            return False
        
        file_size = os.path.getsize(gt_file) / (1024*1024)  # MB
        if verbose:
            print(f"    Ground truth file size: {file_size:.2f} MB")
        
        start = time.time()
        gt_label = np.load(gt_file)
        elapsed = time.time() - start
        
        if verbose:
            print(f"    ✓ Ground truth loaded in {elapsed:.3f}s, shape: {gt_label.shape}, dtype: {gt_label.dtype}")
        
        # Check dimensions match
        if feature.shape[0] != gt_label.shape[0]:
            print(f"    ⚠ Warning: Feature and label lengths don't match!")
            print(f"      Feature: {feature.shape[0]} frames")
            print(f"      Label: {gt_label.shape[0]} frames")
        
        # Check label values
        unique_vals = np.unique(gt_label)
        if verbose:
            print(f"    Label unique values: {unique_vals}")
            print(f"    Label shape: {gt_label.shape}")
            if gt_label.ndim == 2:
                for i in range(gt_label.shape[1]):
                    count = np.sum(gt_label[:, i])
                    print(f"      Column {i}: {count}/{len(gt_label)} frames active ({100*count/len(gt_label):.1f}%)")
        
        return True
    except Exception as e:
        print(f"    ✗ Error loading video: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_first_videos(test_videos, train_videos):
    """Test loading the first few videos from each split"""
    print("="*80)
    print("STEP 4: Testing first video loading (this is where training hangs)")
    print("="*80)
    
    if not test_videos or not train_videos:
        print("  ✗ Split files not loaded, skipping\n")
        return False
    
    # Test first test video
    print("Testing FIRST TEST VIDEO (this is loaded during dataset init):")
    success = test_load_video(test_videos[0], verbose=True)
    print()
    
    if not success:
        print("  ✗ Failed to load first test video - THIS IS YOUR PROBLEM!")
        return False
    
    # Test first train video
    print("Testing FIRST TRAIN VIDEO:")
    success = test_load_video(train_videos[0], verbose=True)
    print()
    
    if not success:
        print("  ⚠ Failed to load first train video")
    
    return True

def test_all_videos_exist(test_videos, train_videos):
    """Check if all video files exist"""
    print("="*80)
    print("STEP 5: Checking if all video files exist")
    print("="*80)
    
    if not test_videos or not train_videos:
        print("  ✗ Split files not loaded, skipping\n")
        return
    
    print(f"Checking {len(test_videos)} test videos...")
    missing_test = []
    for video in test_videos:
        vname = video[:-4] if video.endswith('.txt') else video
        feature_file = os.path.join(FEATURE_PATH, vname + '.npy')
        gt_file = os.path.join(GT_PATH, vname + '.npy')
        if not os.path.exists(feature_file) or not os.path.exists(gt_file):
            missing_test.append(video)
    
    if missing_test:
        print(f"  ✗ Missing {len(missing_test)} test videos:")
        for v in missing_test[:5]:
            print(f"    {v}")
        if len(missing_test) > 5:
            print(f"    ... and {len(missing_test)-5} more")
    else:
        print(f"  ✓ All test videos exist")
    
    print(f"\nChecking {len(train_videos)} train videos...")
    missing_train = []
    for video in train_videos:
        vname = video[:-4] if video.endswith('.txt') else video
        feature_file = os.path.join(FEATURE_PATH, vname + '.npy')
        gt_file = os.path.join(GT_PATH, vname + '.npy')
        if not os.path.exists(feature_file) or not os.path.exists(gt_file):
            missing_train.append(video)
    
    if missing_train:
        print(f"  ✗ Missing {len(missing_train)} train videos:")
        for v in missing_train[:5]:
            print(f"    {v}")
        if len(missing_train) > 5:
            print(f"    ... and {len(missing_train)-5} more")
    else:
        print(f"  ✓ All train videos exist")
    
    print()

def estimate_loading_time(test_videos, train_videos, sample_size=5):
    """Estimate total loading time"""
    print("="*80)
    print("STEP 6: Estimating loading times")
    print("="*80)
    
    if not test_videos:
        print("  ✗ No videos to test\n")
        return
    
    print(f"Testing {sample_size} random videos to estimate loading time...")
    
    import random
    sample = random.sample(test_videos[:min(20, len(test_videos))], min(sample_size, len(test_videos)))
    
    times = []
    for video in sample:
        sys.stdout.write(f"  Testing {video}... ")
        sys.stdout.flush()
        
        vname = video[:-4] if video.endswith('.txt') else video
        feature_file = os.path.join(FEATURE_PATH, vname + '.npy')
        gt_file = os.path.join(GT_PATH, vname + '.npy')
        
        start = time.time()
        try:
            feature = np.load(feature_file)
            gt = np.load(gt_file)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"{elapsed:.3f}s")
        except Exception as e:
            print(f"FAILED: {e}")
    
    if times:
        avg_time = np.mean(times)
        total_videos = len(test_videos) + len(train_videos)
        estimated_total = avg_time * total_videos
        
        print(f"\n  Average loading time: {avg_time:.3f}s per video")
        print(f"  Total videos: {total_videos}")
        print(f"  Estimated time to load all: {estimated_total:.1f}s ({estimated_total/60:.1f} minutes)")
        print()

def main():
    print("\n" + "="*80)
    print("FACT MULTILABEL DATASET DIAGNOSTIC TOOL")
    print("="*80)
    print()
    
    # Run diagnostics
    if not check_paths():
        print("\n❌ CRITICAL: Some required paths don't exist!")
        print("Cannot proceed with dataset loading.")
        return 1
    
    if not load_mapping():
        print("\n❌ CRITICAL: Cannot load mapping file!")
        return 1
    
    test_videos, train_videos = load_split_files()
    if test_videos is None:
        print("\n❌ CRITICAL: Cannot load split files!")
        return 1
    
    if not test_first_videos(test_videos, train_videos):
        print("\n❌ CRITICAL: Cannot load first video!")
        print("This is where your training is hanging!")
        return 1
    
    test_all_videos_exist(test_videos, train_videos)
    estimate_loading_time(test_videos, train_videos)
    
    print("="*80)
    print("✓ DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nIf all checks passed, the dataset should load fine.")
    print("If first video loading took >5 seconds, you might experience slow startup.")
    print()
    
    return 0

if __name__ == '__main__':
    exit(main())
