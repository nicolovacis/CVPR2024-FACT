#!/usr/bin/env python3
"""
Generate visualizations for the training set.
"""
import sys
import os
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FACT.utils.dataset import DataLoader, create_dataset
from FACT.utils.evaluate import Checkpoint
from FACT.utils.train_tools import save_results
from FACT.utils.visualize_temporal_segmentation import plot_multiple_videos_summary
from FACT.configs.utils import setup_cfg
from FACT.models.blocks import FACT


def evaluate_trainset(model_path, cfg_path, output_dir):
    """Evaluate model on training set and generate visualizations"""
    
    # Load config
    print(f"Loading config from: {cfg_path}")
    cfg = setup_cfg([cfg_path], [])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, test_dataset = create_dataset(cfg)
    trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"Training set: {len(train_dataset)} videos")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    if cfg.dataset == 'epic':
        net = FACT(cfg, train_dataset.input_dimension, 98, 301)
    else:
        net = FACT(cfg, train_dataset.input_dimension, train_dataset.nclasses)
    
    # Load checkpoint
    ckpt_data = torch.load(model_path, map_location='cpu')
    if 'frame_pe.pe' in ckpt_data:
        del ckpt_data['frame_pe.pe']
    if 'action_pe.pe' in ckpt_data:
        del ckpt_data['action_pe.pe']
    net.load_state_dict(ckpt_data, strict=False)
    net.cuda()
    net.eval()
    
    print("\nRunning inference on training set...")
    # Create checkpoint to store results
    ckpt = Checkpoint(0, bg_class=([] if cfg.eval_bg else train_dataset.bg_class))
    
    with torch.no_grad():
        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(trainloader):
            seq_list = [s.cuda() for s in seq_list]
            train_label_list = [s.cuda() for s in train_label_list]
            
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(trainloader)} batches")
    
    print(f"\nComputing metrics...")
    ckpt.compute_metrics()
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    class_names = ['background', 'licking']
    accuracies = plot_multiple_videos_summary(
        checkpoint=ckpt,
        save_dir=output_dir,
        class_names=class_names,
        max_videos=None,
        max_frames_per_video=None
    )
    
    print(f"\n{'='*60}")
    print("✓ Training set visualization complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Number of videos: {len(accuracies)}")
    
    return accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training set predictions')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.net file)')
    parser.add_argument('--cfg', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    try:
        evaluate_trainset(args.model, args.cfg, args.output)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
