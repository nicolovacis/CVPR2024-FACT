#!/usr/bin/python3
"""
Modified dataset loader for multi-label classification

Key differences from standard dataset.py:
1. Loads labels from .npy files instead of .txt files
2. Labels are shape (T, num_labels) instead of (T,)
3. Each frame can have multiple active labels
"""

import numpy as np
import os
import torch
from yacs.config import CfgNode

def load_feature(feature_dir, video, transpose):
    file_name = os.path.join(feature_dir, video+'.npy')
    feature = np.load(file_name)

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature

def load_action_mapping(map_fname, sep=" "):
    """Load mapping file for label names"""
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label

class MultiLabelDataset(object):
    """
    Multi-label dataset where each frame can have multiple labels
    
    self.features[video]: feature array (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of label types (e.g., 2 for licking and shaking)
    """

    def __init__(self, video_list, nclasses, load_video_func, bg_class):
        self.video_list = video_list
        self.load_video = load_video_func

        # Store dataset information
        self.nclasses = nclasses
        self.bg_class = bg_class
        self.data = {}
        self.data[video_list[0]] = load_video_func(video_list[0])
        self.input_dimension = self.data[video_list[0]][0].shape[1] 
    
    def __str__(self):
        string = "< MultiLabelDataset %d videos, %d feat-size, %d labels >"
        string = string % (len(self.video_list), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


class DataLoader():
    """Compatible data loader for multi-label datasets"""

    def __init__(self, dataset: MultiLabelDataset, batch_size, shuffle=False):
        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batch = int(np.ceil(self.num_video/self.batch_size))
        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
            self.index = 0
            raise StopIteration
        else:
            video_idx = self.selector[self.index : self.index+self.batch_size]
            if len(video_idx) < self.batch_size:
                video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence = []
            batch_train_label = []
            batch_eval_label = []
            for vname in videos:
                sequence, train_label, eval_label = self.dataset[vname]
                batch_sequence.append(torch.from_numpy(sequence))
                # For multi-label, convert to float tensor
                batch_train_label.append(torch.FloatTensor(train_label))
                batch_eval_label.append(eval_label)

            return videos, batch_sequence, batch_train_label, batch_eval_label


def create_multilabel_dataset(cfg: CfgNode):
    """
    Create multi-label dataset
    
    Expected file structure:
    - features/*.npy: shape (T, D)
    - groundTruth/*.npy: shape (T, num_labels) with binary values
    - mapping.txt: "0 label_name_0\n1 label_name_1\n..."
    """
    
    map_fname = cfg.map_fname
    feature_path = cfg.feature_path
    groundTruth_path = cfg.groundTruth_path
    train_split_fname = os.path.join(cfg.split_path, f'train.{cfg.split}.bundle')
    test_split_fname = os.path.join(cfg.split_path, f'test.{cfg.split}.bundle')
    feature_transpose = cfg.feature_transpose
    # For multilabel, we don't use bg_class (background is implicit [0,0])
    # But we keep it as empty list for compatibility with evaluation code
    bg_class = []
    
    print("Loading Multi-Label Dataset")
    print("Loading Feature from", feature_path)
    print("Loading Label from", groundTruth_path)

    label2index, index2label = load_action_mapping(map_fname)
    nclasses = len(label2index)
    print(f"Number of label types: {nclasses}")
    print(f"Label names: {list(index2label.values())}")

    def load_video(vname):
        """
        Load video features and multi-label annotations
        
        Returns:
            feature: (T, D) array
            train_label: (T, num_labels) binary array for training
            eval_label: (T, num_labels) binary array for evaluation
        """
        if vname.endswith('.txt'):
            vname = vname[:-4]
        
        # Load features
        feature = load_feature(feature_path, vname, feature_transpose)
        
        # Load multi-label ground truth from .npy file
        gt_label_file = os.path.join(groundTruth_path, vname + '.npy')
        gt_label = np.load(gt_label_file)  # Shape: (T, num_labels)
        
        # Ensure compatibility
        if feature.shape[0] != gt_label.shape[0]:
            l = min(feature.shape[0], gt_label.shape[0])
            feature = feature[:l]
            gt_label = gt_label[:l]
        
        # Downsample if necessary
        sr = cfg.sr
        if sr > 1:
            feature = feature[::sr]
            # For multi-label, downsample by taking max over window
            gt_label_sampled = []
            for i in range(0, len(gt_label), sr):
                window = gt_label[i:i+sr]
                # Take max (if any frame in window has label, keep it)
                gt_label_sampled.append(np.max(window, axis=0))
            gt_label_sampled = np.array(gt_label_sampled)
        else:
            gt_label_sampled = gt_label
        
        return feature, gt_label_sampled, gt_label

    # Load test dataset
    with open(test_split_fname, 'r') as f:
        test_video_list = f.read().split('\n')[0:-1]
    test_dataset = MultiLabelDataset(test_video_list, nclasses, load_video, bg_class)

    # Load train dataset
    if cfg.aux.debug:
        dataset = test_dataset
    else:
        with open(train_split_fname, 'r') as f:
            video_list = f.read().split('\n')[0:-1]
        dataset = MultiLabelDataset(video_list, nclasses, load_video, bg_class)
    
    # Add metadata
    dataset.label2index = label2index
    dataset.index2label = index2label
    test_dataset.label2index = label2index
    test_dataset.index2label = index2label

    return dataset, test_dataset