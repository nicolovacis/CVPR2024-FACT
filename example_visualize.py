#!/usr/bin/env python3
"""
Simple example script showing how to use the visualization after training completes.
This demonstrates how to visualize existing checkpoints.
"""

import os
import sys

# Ensure FACT is in the path
sys.path.insert(0, '/home/nvaci/FACT')

from FACT.utils.evaluate import Checkpoint
from FACT.utils.visualize_temporal_segmentation import plot_multiple_videos_summary

def visualize_existing_checkpoint():
    """
    Example: Visualize an existing checkpoint from your training run
    """
    
    # Path to your checkpoint
    checkpoint_path = "/data-8tb/nvaci/runs/FACT/stanford/split1/stanford/0/best_ckpt.gz"
    output_dir = "/data-8tb/nvaci/runs/FACT/test_visualizations"
    
    # Check if matplotlib is available
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__} is installed")
    except ImportError:
        print("✗ matplotlib is not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "--quiet"])
        print("✓ matplotlib installed successfully")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = Checkpoint.load(checkpoint_path)
    print(f"✓ Loaded checkpoint with {len(ckpt.videos)} videos")
    
    # Get class names (adjust based on your dataset)
    # Stanford dataset typically has: background, licking, and possibly other behaviors
    class_names = ['background', 'licking', 'shaking', 'other']
    
    # Generate visualizations
    print(f"\nGenerating visualizations to: {output_dir}")
    accuracies = plot_multiple_videos_summary(
        checkpoint=ckpt,
        save_dir=output_dir,
        class_names=class_names,
        max_videos=None,  # Visualize all videos
        max_frames_per_video=None  # Show all frames
    )
    
    print(f"\n{'='*60}")
    print("✓ Visualization complete!")
    print(f"{'='*60}")
    print(f"\nGenerated visualizations for {len(accuracies)} videos")
    print(f"Output location: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {len(accuracies)} video plots (PNG)")
    print(f"  - 1 summary file (TXT)")
    print(f"\nView summary:")
    print(f"  cat {output_dir}/summary.txt")
    print(f"\nView plots:")
    print(f"  ls {output_dir}/*.png")


if __name__ == '__main__':
    try:
        visualize_existing_checkpoint()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
