#!/usr/bin/env python3
"""
Standalone script to generate temporal segmentation visualizations from saved checkpoints.
Usage:
    python -m FACT.visualize_checkpoint --checkpoint /path/to/checkpoint.gz --output /path/to/output/dir
"""

import argparse
import os
import sys

# Add FACT to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FACT.utils.evaluate import Checkpoint
from FACT.utils.visualize_temporal_segmentation import plot_multiple_videos_summary


def main():
    parser = argparse.ArgumentParser(description='Generate temporal segmentation visualizations from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.gz)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--class-names', nargs='+', default=['background', 'licking'],
                       help='Names of classes (default: background licking)')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Maximum number of videos to plot (default: all)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames per video (default: all)')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    try:
        ckpt = Checkpoint.load(args.checkpoint)
        print(f"✓ Loaded checkpoint with {len(ckpt.videos)} videos")
        print(f"  Iteration: {ckpt.iteration}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {args.output}")
    
    try:
        accuracies = plot_multiple_videos_summary(
            checkpoint=ckpt,
            save_dir=args.output,
            class_names=args.class_names,
            max_videos=args.max_videos,
            max_frames_per_video=args.max_frames
        )
        
        print(f"\n✓ Successfully generated visualizations for {len(accuracies)} videos")
        print(f"✓ Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
