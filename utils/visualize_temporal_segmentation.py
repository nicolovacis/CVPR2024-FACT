"""
Visualization utilities for temporal action segmentation.
Creates plots comparing ground truth labels vs predictions over time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import Dict, List, Optional


def plot_temporal_segmentation(
    video_name: str,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    colors: Optional[Dict[int, str]] = None,
    max_frames: Optional[int] = None
):
    """
    Plot temporal action segmentation comparing ground truth and predictions.
    
    Args:
        video_name: Name of the video
        gt_labels: Ground truth labels (1D array of frame-level labels)
        pred_labels: Predicted labels (1D array of frame-level labels)
        save_path: Path to save the figure
        class_names: List of class names (optional)
        colors: Dictionary mapping class indices to colors (optional)
        max_frames: Maximum number of frames to plot (optional, for very long videos)
    """
    # Ensure labels are numpy arrays
    gt_labels = np.array(gt_labels).flatten()
    pred_labels = np.array(pred_labels).flatten()
    
    # Ensure same length
    min_len = min(len(gt_labels), len(pred_labels))
    gt_labels = gt_labels[:min_len]
    pred_labels = pred_labels[:min_len]
    
    # Limit frames if specified
    if max_frames is not None and len(gt_labels) > max_frames:
        gt_labels = gt_labels[:max_frames]
        pred_labels = pred_labels[:max_frames]
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([gt_labels, pred_labels]))
    num_classes = len(unique_classes)
    
    # Create default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Create default colors if not provided
    if colors is None:
        # Use a colormap for distinct colors
        cmap = plt.cm.get_cmap('tab10' if num_classes <= 10 else 'tab20')
        colors = {i: cmap(i / max(num_classes - 1, 1)) for i in unique_classes}
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 8), sharex=True)
    
    # Time axis (in seconds)
    time_axis = np.arange(len(gt_labels))
    
    # Plot 1: Ground Truth
    for i in range(len(gt_labels)):
        ax1.axvspan(i, i + 1, facecolor=colors[gt_labels[i]], alpha=0.8)
    ax1.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Plot 2: Predictions
    for i in range(len(pred_labels)):
        ax2.axvspan(i, i + 1, facecolor=colors[pred_labels[i]], alpha=0.8)
    ax2.set_ylabel('Prediction', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Plot 3: Difference (Errors)
    correct = (gt_labels == pred_labels).astype(int)
    for i in range(len(correct)):
        color = 'green' if correct[i] else 'red'
        ax3.axvspan(i, i + 1, facecolor=color, alpha=0.6)
    ax3.set_ylabel('Correct/Error', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Create legend
    legend_patches = []
    for class_idx in unique_classes:
        if class_idx < len(class_names):
            label = class_names[class_idx]
        else:
            label = f'Class {class_idx}'
        legend_patches.append(mpatches.Patch(color=colors[class_idx], label=label))
    
    # Add error legend
    legend_patches.append(mpatches.Patch(color='green', label='Correct', alpha=0.6))
    legend_patches.append(mpatches.Patch(color='red', label='Error', alpha=0.6))
    
    # Place legend outside the plot
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
               ncol=1, fontsize=10, framealpha=0.9)
    
    # Calculate and display accuracy
    accuracy = np.mean(correct) * 100
    fig.suptitle(f'{video_name}\nFrame-wise Accuracy: {accuracy:.2f}%', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return accuracy


def plot_multiple_videos_summary(
    checkpoint,
    save_dir: str,
    class_names: Optional[List[str]] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None
):
    """
    Generate temporal segmentation plots for all videos in a checkpoint.
    
    Args:
        checkpoint: Checkpoint object containing videos with predictions
        save_dir: Directory to save all plots
        class_names: List of class names (optional)
        max_videos: Maximum number of videos to plot (optional)
        max_frames_per_video: Maximum frames per video plot (optional)
    
    Returns:
        Dictionary with video names and their accuracies
    """
    os.makedirs(save_dir, exist_ok=True)
    
    accuracies = {}
    video_names = list(checkpoint.videos.keys())
    
    if max_videos is not None:
        video_names = video_names[:max_videos]
    
    # Define colors for binary classification (background vs licking)
    colors = {
        0: '#e74c3c',  # Red for background
        1: '#3498db',  # Blue for licking
    }
    
    print(f"\nGenerating temporal segmentation plots...")
    print(f"Saving to: {save_dir}")
    print(f"Number of videos: {len(video_names)}")
    
    for idx, vname in enumerate(video_names):
        video = checkpoint.videos[vname]
        
        # Get labels
        gt_labels = video.gt_label
        pred_labels = video.pred_label
        
        # Create plot
        save_path = os.path.join(save_dir, f'{vname}_temporal_segmentation.png')
        
        try:
            accuracy = plot_temporal_segmentation(
                video_name=vname,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                save_path=save_path,
                class_names=class_names,
                colors=colors,
                max_frames=max_frames_per_video
            )
            accuracies[vname] = accuracy
            
            if (idx + 1) % 5 == 0 or (idx + 1) == len(video_names):
                print(f"  Processed {idx + 1}/{len(video_names)} videos")
        
        except Exception as e:
            print(f"  Error processing {vname}: {e}")
            accuracies[vname] = None
    
    # Create summary statistics
    valid_accuracies = [acc for acc in accuracies.values() if acc is not None]
    if valid_accuracies:
        mean_acc = np.mean(valid_accuracies)
        std_acc = np.std(valid_accuracies)
        
        summary_path = os.path.join(save_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Temporal Segmentation Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total videos: {len(video_names)}\n")
            f.write(f"Successfully processed: {len(valid_accuracies)}\n")
            f.write(f"Mean frame-wise accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
            f.write(f"Min accuracy: {min(valid_accuracies):.2f}%\n")
            f.write(f"Max accuracy: {max(valid_accuracies):.2f}%\n\n")
            f.write(f"Per-video accuracies:\n")
            f.write("-" * 50 + "\n")
            
            for vname in sorted(accuracies.keys()):
                acc = accuracies[vname]
                if acc is not None:
                    f.write(f"{vname}: {acc:.2f}%\n")
                else:
                    f.write(f"{vname}: ERROR\n")
        
        print(f"\n✓ Summary saved to: {summary_path}")
        print(f"✓ Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return accuracies


def visualize_confusion_regions(
    video_name: str,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: str,
    context_window: int = 50
):
    """
    Visualize regions where the model makes errors with context.
    
    Args:
        video_name: Name of the video
        gt_labels: Ground truth labels
        pred_labels: Predicted labels
        save_path: Path to save the figure
        context_window: Number of frames to show around each error
    """
    gt_labels = np.array(gt_labels).flatten()
    pred_labels = np.array(pred_labels).flatten()
    
    # Find error indices
    errors = np.where(gt_labels != pred_labels)[0]
    
    if len(errors) == 0:
        print(f"No errors found for {video_name}")
        return
    
    # Group consecutive errors
    error_groups = []
    current_group = [errors[0]]
    
    for i in range(1, len(errors)):
        if errors[i] - errors[i-1] <= context_window:
            current_group.append(errors[i])
        else:
            error_groups.append(current_group)
            current_group = [errors[i]]
    error_groups.append(current_group)
    
    # Plot top error regions (max 10)
    num_plots = min(10, len(error_groups))
    fig, axes = plt.subplots(num_plots, 1, figsize=(20, 2*num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    for idx, error_group in enumerate(error_groups[:num_plots]):
        ax = axes[idx]
        
        # Determine plot range
        center = (error_group[0] + error_group[-1]) // 2
        start = max(0, center - context_window)
        end = min(len(gt_labels), center + context_window)
        
        # Plot
        time_range = np.arange(start, end)
        ax.fill_between(time_range, 0, 1, where=(gt_labels[start:end] == 0), 
                        color='#e74c3c', alpha=0.3, label='GT: Background')
        ax.fill_between(time_range, 0, 1, where=(gt_labels[start:end] == 1), 
                        color='#3498db', alpha=0.3, label='GT: Licking')
        
        # Overlay predictions
        ax.plot(time_range, pred_labels[start:end] * 0.5 + 0.25, 
                'g-', linewidth=2, label='Prediction', alpha=0.7)
        
        # Highlight errors
        error_mask = gt_labels[start:end] != pred_labels[start:end]
        ax.scatter(time_range[error_mask], 
                  pred_labels[start:end][error_mask] * 0.5 + 0.25,
                  color='red', s=50, marker='x', label='Error', zorder=5)
        
        ax.set_ylabel(f'Region {idx+1}', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    fig.suptitle(f'{video_name} - Error Regions Analysis', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
