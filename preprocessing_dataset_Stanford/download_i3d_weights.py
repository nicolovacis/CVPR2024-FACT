#!/usr/bin/env python3
"""
Download pretrained I3D model weights
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    parser = argparse.ArgumentParser(description='Download pretrained I3D weights')
    parser.add_argument('--output_dir', type=str, default='./pretrained_models',
                        help='Directory to save pretrained weights')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading pretrained I3D model weights")
    print("=" * 80)
    
    # I3D RGB model pretrained on Kinetics
    i3d_rgb_url = "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"
    i3d_rgb_path = output_dir / "i3d_rgb_kinetics.pt"
    
    if i3d_rgb_path.exists():
        print(f"\nI3D RGB model already exists at: {i3d_rgb_path}")
        print("Skipping download.")
    else:
        print(f"\nDownloading I3D RGB model from: {i3d_rgb_url}")
        print(f"Saving to: {i3d_rgb_path}")
        try:
            download_url(i3d_rgb_url, i3d_rgb_path)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading I3D weights: {e}")
            print("\nAlternative: You can manually download from:")
            print("  https://github.com/piergiaj/pytorch-i3d")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Setup complete!")
    print("=" * 80)
    print(f"\nPretrained models saved in: {output_dir}")
    print("\nYou can now run the preprocessing script with:")
    print(f"  python preprocess_stanford_dataset.py --i3d_weights {i3d_rgb_path} ...")


if __name__ == '__main__':
    main()
