#!/usr/bin/env python3
"""
Quick test script to verify the visualization functionality.
Tests with an existing checkpoint without matplotlib dependency.
"""

import os
import sys

# Add FACT to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FACT.utils.evaluate import Checkpoint

def test_checkpoint_loading():
    """Test loading a checkpoint and inspecting its contents"""
    
    # Find an existing checkpoint
    checkpoint_paths = [
        "/data-8tb/nvaci/runs/FACT/stanford/split1/stanford/0/best_ckpt.gz",
        "/data-8tb/nvaci/runs/FACT/stanford/split1/stanford/0/saves/1200.gz",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("No checkpoint found to test with")
        return False
    
    print(f"Testing with checkpoint: {checkpoint_path}")
    print("="*60)
    
    try:
        # Load checkpoint
        print("\n1. Loading checkpoint...")
        ckpt = Checkpoint.load(checkpoint_path)
        print(f"   ✓ Loaded successfully")
        print(f"   - Iteration: {ckpt.iteration}")
        print(f"   - Number of videos: {len(ckpt.videos)}")
        print(f"   - Background class: {ckpt.bg_class}")
        
        # Check video structure
        print("\n2. Inspecting video data...")
        video_names = list(ckpt.videos.keys())
        print(f"   - Video names: {video_names[:5]}{'...' if len(video_names) > 5 else ''}")
        
        # Check first video
        first_video = ckpt.videos[video_names[0]]
        print(f"\n3. Checking first video ({video_names[0]})...")
        
        # Check attributes
        has_gt = hasattr(first_video, 'gt_label')
        has_pred = hasattr(first_video, 'pred_label')
        has_pred_raw = hasattr(first_video, 'pred')
        
        print(f"   - Has gt_label: {has_gt}")
        print(f"   - Has pred_label: {has_pred}")
        print(f"   - Has pred: {has_pred_raw}")
        
        if has_gt:
            print(f"   - GT shape: {first_video.gt_label.shape}")
            print(f"   - GT unique values: {set(first_video.gt_label)}")
        
        if has_pred:
            print(f"   - Pred shape: {first_video.pred_label.shape}")
            print(f"   - Pred unique values: {set(first_video.pred_label)}")
        elif has_pred_raw:
            print(f"   - Raw pred shape: {first_video.pred.shape}")
            print(f"   ⚠ Note: pred_label not computed yet. Need to call compute_metrics()")
        
        # Try to compute metrics if needed
        if not has_pred:
            print("\n4. Computing metrics to generate pred_label...")
            ckpt.compute_metrics()
            first_video = ckpt.videos[video_names[0]]
            if hasattr(first_video, 'pred_label'):
                print(f"   ✓ pred_label generated")
                print(f"   - Shape: {first_video.pred_label.shape}")
                print(f"   - Unique values: {set(first_video.pred_label)}")
        
        # Check metrics
        if hasattr(ckpt, 'metrics'):
            print("\n5. Checkpoint metrics:")
            for key, value in list(ckpt.metrics.items())[:10]:
                print(f"   - {key}: {value:.2f}")
        
        print("\n" + "="*60)
        print("✓ Checkpoint structure is valid for visualization!")
        print("\nTo generate visualizations with matplotlib installed, run:")
        print(f"python -m FACT.visualize_checkpoint \\")
        print(f"    --checkpoint {checkpoint_path} \\")
        print(f"    --output /data-8tb/nvaci/runs/FACT/test_visualizations")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)
