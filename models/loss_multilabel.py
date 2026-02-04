"""
Modified loss functions for multi-label classification

Key changes:
1. Use Binary Cross Entropy (BCE) instead of Cross Entropy
2. Each label is predicted independently with sigmoid activation
3. Support multiple active labels per frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def smooth_loss_multilabel(logit):
    """
    Smoothness loss for multi-label predictions
    logit: B, T, num_labels
    """
    loss = torch.clamp((logit[:, 1:] - logit[:, :-1])**2, min=0, max=16)
    loss = loss.mean()
    return loss


def torch_multilabel_to_segments(label):
    """
    Convert multi-label frame annotations to action segments
    
    For multi-label, we need to handle combinations of labels as separate "actions"
    E.g., [1,0], [0,1], [1,1] are three different action types
    
    Args:
        label: (T, num_labels) binary array
    
    Returns:
        transcript: unique action combinations in order
        segment_label: segment ID for each frame
    """
    # Convert multi-label to single composite label
    # E.g., [1, 0] -> 1, [0, 1] -> 2, [1, 1] -> 3, [0, 0] -> 0
    num_labels = label.shape[1]
    composite_label = torch.zeros(label.shape[0], dtype=torch.long, device=label.device)
    
    for i in range(num_labels):
        composite_label += label[:, i] * (2 ** i)
    
    # Now use standard segment extraction
    segment_label = torch.zeros_like(composite_label)
    current = composite_label[0]
    transcript = [current]
    aid = 0
    
    for i, l in enumerate(composite_label):
        if l == current:
            pass
        else:
            current = l
            aid += 1
            transcript.append(l)
        segment_label[i] = aid
    
    transcript = torch.stack(transcript).to(label.device)
    return transcript, segment_label


class MultiLabelMatchCriterion():
    """
    Matching criterion for multi-label temporal action segmentation
    """
    
    def __init__(self, cfg, nclasses, bg_ids=[], class_weight=None):
        self.cfg = cfg
        self.nclasses = nclasses  # Number of label types (e.g., 2 for licking/shaking)
        self.bg_ids = bg_ids
        self._class_weight = class_weight
    
    def set_label(self, label):
        """
        Set ground truth label
        
        Args:
            label: (T, num_labels) binary tensor
        """
        self.multilabel = label  # Store original multi-label
        
        # Convert to composite segments for matching
        self.transcript, self.seg_label = torch_multilabel_to_segments(label)
        
        # Create one-hot encodings
        self.onehot_seg_label = self._label_to_onehot(self.seg_label, len(self.transcript))
        
        # Create label weight
        # For multi-label, we use BCE loss so weights are per-label
        label_weight = torch.ones(self.nclasses).to(label.device)
        if self._class_weight is not None:
            for i in range(self.nclasses):
                label_weight[i] = self._class_weight[i]
        else:
            for i in self.bg_ids:
                if i < self.nclasses:
                    label_weight[i] = self.cfg.Loss.bgw
        
        self.label_weight = label_weight
        
        # Create segment weights
        sweight = torch.ones(len(self.transcript), dtype=torch.float32).to(label.device)
        self.sweight = sweight
    
    def _label_to_onehot(self, label, nclass):
        onehot_label = torch.zeros(len(label), nclass).to(label.device)
        onehot_label[torch.arange(len(label)), label] = 1
        return onehot_label
    
    @classmethod
    def a2f_soft_iou(cls, a2f_attn, onehot_seg_label):
        """
        Compute soft IoU between action tokens and frame segments
        """
        a2f_attn = a2f_attn[0].unsqueeze(-1)  # f, a, 1
        onehot_seg_label = onehot_seg_label.unsqueeze(1)  # f, 1, s
        
        a2f_attn_np = a2f_attn.cpu().numpy()
        onehot_seg_label_np = onehot_seg_label.cpu().numpy()
        
        overlap = np.einsum('tax,txs->as', a2f_attn_np, onehot_seg_label_np)
        union = np.minimum(a2f_attn_np + onehot_seg_label_np, 1.0).sum(0)
        iou = np.nan_to_num(overlap / union, nan=0.0)
        
        return iou
    
    def match(self, clogit, a2f_attn):
        """
        Match action tokens to ground truth segments
        
        For multi-label, clogit should be: (num_tokens, 1, num_labels)
        where each label is predicted independently
        """
        assert clogit.shape[1] == 1  # batch_size == 1
        
        match_cfg = self.cfg.Loss
        
        # Sequential matching
        if match_cfg.match == 'seq':
            A = clogit.shape[0]
            S = self.onehot_seg_label.shape[-1]
            assert A >= S, (A, S)
            action_ind = seg_ind = torch.as_tensor(list(range(S)), dtype=torch.int64)
            return action_ind, seg_ind
        
        # Compute matching cost
        cost = 0
        with torch.no_grad():
            # For multi-label, we need to compute similarity differently
            # Convert predictions to probabilities
            prob = torch.sigmoid(clogit.squeeze(1))  # (num_tokens, num_labels)
            
            # Get ground truth labels for each segment
            seg_multilabels = []
            for s in range(len(self.transcript)):
                seg_mask = (self.seg_label == s)
                seg_label = self.multilabel[seg_mask].float().mean(0)  # Average over segment
                seg_multilabels.append(seg_label)
            seg_multilabels = torch.stack(seg_multilabels)  # (num_segs, num_labels)
            
            # Compute similarity (negative L2 distance)
            if match_cfg.pc > 0:
                prob_np = prob.cpu().numpy()  # (num_tokens, num_labels)
                seg_np = seg_multilabels.cpu().numpy()  # (num_segs, num_labels)
                
                # Compute pairwise distances
                similarity = -np.sum((prob_np[:, None, :] - seg_np[None, :, :]) ** 2, axis=2)
                cost += match_cfg.pc * similarity
            
            if match_cfg.a2fc > 0:
                a2f_iou = self.a2f_soft_iou(a2f_attn, self.onehot_seg_label)
                cost -= match_cfg.a2fc * a2f_iou
        
        cost = cost.cpu().numpy() if isinstance(cost, torch.Tensor) else cost
        
        # Find optimal matching
        if match_cfg.match == 'o2o':
            action_ind, seg_ind = linear_sum_assignment(cost)
        elif match_cfg.match == 'o2m':
            # For one-to-many, we can still use standard approach
            # but it's less relevant for multi-label
            action_ind, seg_ind = self._one_to_many_match(cost)
        
        action_ind = torch.as_tensor(action_ind, dtype=torch.int64)
        seg_ind = torch.as_tensor(seg_ind, dtype=torch.int64)
        
        return action_ind, seg_ind
    
    def _one_to_many_match(self, cost):
        """One-to-many matching (simplified for multi-label)"""
        action_ind, seg_ind = linear_sum_assignment(cost)
        return action_ind, seg_ind
    
    def action_token_loss(self, match, action_logit):
        """
        Multi-label BCE loss for action tokens
        
        Args:
            match: (action_ind, seg_ind) matching
            action_logit: (num_tokens, 1, num_labels) logits
        """
        aind, sind = match
        A = action_logit.shape[0]
        
        # Create target labels for each token
        target = torch.zeros(A, self.nclasses).to(action_logit.device)
        
        for ai, si in zip(aind, sind):
            seg_mask = (self.seg_label == si)
            seg_label = self.multilabel[seg_mask].float().mean(0)
            target[ai] = seg_label
        
        # Binary cross entropy with logits
        action_logit = action_logit.squeeze(1)  # (A, num_labels)
        loss = F.binary_cross_entropy_with_logits(
            action_logit, 
            target,
            pos_weight=self.label_weight
        )
        
        return loss
    
    def cross_attn_loss(self, match, attn, dim=None):
        """Cross attention loss (same as standard)"""
        assert dim >= 1
        onehot_seg_label = self.onehot_seg_label
        aind, sind = match
        
        frame_tgt = onehot_seg_label[:, sind]  # f, s
        attn = attn[0, :, aind]  # f, s
        attn_logp = torch.log_softmax(attn, dim=dim-1)
        loss = -attn_logp * frame_tgt
        
        if self.sweight is not None:
            loss = loss * self.sweight
        loss = loss.sum(1).sum() / self.onehot_seg_label.sum()
        
        return loss
    
    def frame_loss(self, frame_logit):
        """
        Multi-label BCE loss for frame predictions
        
        Args:
            frame_logit: (T, 1, num_labels) logits
        """
        frame_logit = frame_logit.squeeze(1)  # (T, num_labels)
        target = self.multilabel.float()
        
        # Binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            frame_logit,
            target,
            pos_weight=self.label_weight
        )
        
        return loss


# Export functions for compatibility
def logit2prob_multilabel(logit, dim=-1):
    """Convert logits to probabilities for multi-label (sigmoid)"""
    return torch.sigmoid(logit)
