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
        label: (T, num_labels) binary array (can be float or long)
    
    Returns:
        transcript: unique action combinations in order
        segment_label: segment ID for each frame
    """
    # Convert multi-label to single composite label
    # E.g., [1, 0] -> 1, [0, 1] -> 2, [1, 1] -> 3, [0, 0] -> 0
    num_labels = label.shape[1]
    composite_label = torch.zeros(label.shape[0], dtype=torch.long, device=label.device)
    
    # Convert label to long type for proper integer operations
    label_long = label.long()
    
    for i in range(num_labels):
        composite_label += label_long[:, i] * (2 ** i)
    
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
        
        Args:
            a2f_attn: (1, f, a) attention weights from frames to action tokens
            onehot_seg_label: (f, num_segs) one-hot segment labels
        
        Returns:
            iou: (a, num_segs) IoU matrix
        """
        # a2f_attn: (1, f, a) -> (f, a)
        a2f_attn = a2f_attn[0]  # f, a
        
        a2f_attn_np = a2f_attn.cpu().numpy()  # (f, a)
        onehot_seg_label_np = onehot_seg_label.cpu().numpy()  # (f, s)
        
        # Compute overlap: for each (action, segment) pair, sum of min(attn, label) over frames
        # a2f_attn_np: (f, a) -> (f, a, 1)
        # onehot_seg_label_np: (f, s) -> (f, 1, s)
        attn_expanded = a2f_attn_np[:, :, np.newaxis]  # (f, a, 1)
        label_expanded = onehot_seg_label_np[:, np.newaxis, :]  # (f, 1, s)
        
        # Compute overlap and union per (action, segment) pair
        overlap = np.einsum('fa,fs->as', a2f_attn_np, onehot_seg_label_np)  # (a, s)
        
        # Union = sum over frames of max(attn, label) for each (a, s) pair
        # But for soft IoU, we use: union = attn_sum + label_sum - overlap
        attn_sum = a2f_attn_np.sum(0)[:, np.newaxis]  # (a, 1)
        label_sum = onehot_seg_label_np.sum(0)[np.newaxis, :]  # (1, s)
        union = attn_sum + label_sum - overlap  # (a, s)
        
        # Avoid division by zero
        union = np.maximum(union, 1e-8)
        iou = overlap / union
        iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
        
        return iou
    
    def match(self, clogit, a2f_attn):
        """
        Match action tokens to ground truth segments
        
        For multi-label, clogit should be: (num_tokens, 1, num_labels)
        where each label is predicted independently
        """
        assert clogit.shape[1] == 1  # batch_size == 1
        
        match_cfg = self.cfg.Loss
        num_tokens = clogit.shape[0]
        num_segs = len(self.transcript)
        
        # Sequential matching
        if match_cfg.match == 'seq':
            A = num_tokens
            S = self.onehot_seg_label.shape[-1]
            assert A >= S, (A, S)
            action_ind = seg_ind = torch.as_tensor(list(range(S)), dtype=torch.int64)
            return action_ind, seg_ind
        
        # Initialize cost matrix
        cost = np.zeros((num_tokens, num_segs), dtype=np.float64)
        
        with torch.no_grad():
            # For multi-label, we need to compute similarity differently
            # Convert predictions to probabilities
            prob = torch.sigmoid(clogit.squeeze(1))  # (num_tokens, num_labels)
            
            # Get ground truth labels for each segment
            seg_multilabels = []
            for s in range(num_segs):
                seg_mask = (self.seg_label == s)
                seg_label = self.multilabel[seg_mask].float().mean(0)  # Average over segment
                seg_multilabels.append(seg_label)
            seg_multilabels = torch.stack(seg_multilabels)  # (num_segs, num_labels)
            
            # Compute similarity (negative L2 distance)
            if match_cfg.pc > 0:
                prob_np = prob.cpu().numpy()  # (num_tokens, num_labels)
                seg_np = seg_multilabels.cpu().numpy()  # (num_segs, num_labels)
                
                # Compute pairwise distances: cost[i,j] = -||prob[i] - seg[j]||^2
                similarity = -np.sum((prob_np[:, None, :] - seg_np[None, :, :]) ** 2, axis=2)
                cost += match_cfg.pc * similarity
            
            if match_cfg.a2fc > 0:
                a2f_iou = self.a2f_soft_iou(a2f_attn, self.onehot_seg_label)
                cost -= match_cfg.a2fc * a2f_iou
        
        # Ensure cost matrix has no invalid values
        cost = np.nan_to_num(cost, nan=0.0, posinf=1e10, neginf=-1e10)
        
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
    
    def cross_attn_loss_tdu(self, match, attn, tdu, dim=None):
        """
        Cross attention loss for TDU blocks where attention is at segment level
        
        Args:
            match: (action_ind, seg_ind) matching
            attn: (1, num_segs, num_matched_actions) attention logits
            tdu: TemporalDownsampleUpsample object
            dim: dimension for softmax
        """
        assert dim >= 1
        onehot_seg_label = self.onehot_seg_label
        aind, sind = match
        
        # Downsample frame-level one-hot labels to segment level
        # f, num_transcript_segs -> tdu_num_segs, num_transcript_segs
        zoomed_label = torch.zeros([tdu.num_seg, onehot_seg_label.shape[1]], 
                                   dtype=onehot_seg_label.dtype).to(onehot_seg_label.device)
        zoomed_label.index_add_(0, tdu.seg_label, onehot_seg_label)
        zoomed_label = zoomed_label / tdu.seg_lens[:, None]
        
        frame_tgt = zoomed_label[:, sind]  # tdu_num_segs, num_matched
        attn = attn[0, :, aind]  # tdu_num_segs, num_matched
        attn_logp = torch.log_softmax(attn, dim=dim-1)
        
        loss = -attn_logp * frame_tgt
        if self.sweight is not None:
            loss = loss * self.sweight
        
        loss = loss.sum(1).sum() / zoomed_label.sum()
        
        return loss
    
    def frame_loss(self, frame_logit):
        """
        Multi-label BCE loss for frame predictions
        
        Args:
            frame_logit: (T, num_labels) logits (already squeezed from T, 1, num_labels)
        """
        # Ensure proper shape - should be (T, num_labels)
        if frame_logit.dim() == 3:
            frame_logit = frame_logit.squeeze(1)  # (T, num_labels)
        target = self.multilabel.float()
        
        # Binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            frame_logit,
            target,
            pos_weight=self.label_weight
        )
        
        return loss
    
    def segment_loss(self, seg_logit, tdu):
        """
        Multi-label BCE loss for segment predictions
        
        Args:
            seg_logit: (S, num_labels) logits for segments
            tdu: TemporalDownsampleUpsample object with segment info
        """
        # Ensure proper shape - should be (S, num_labels)
        if seg_logit.dim() == 3:
            seg_logit = seg_logit.squeeze(1)  # (S, num_labels)
        
        # Compute segment-level target by averaging frame labels within each segment
        seg_targets = []
        for seg in tdu.segs:
            # Get frames in this segment
            seg_labels = self.multilabel[seg.start:seg.end].float()
            # Average over segment (or could use max for "any frame has label")
            seg_target = seg_labels.mean(0)
            seg_targets.append(seg_target)
        
        target = torch.stack(seg_targets).to(seg_logit.device)
        
        # Binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            seg_logit,
            target,
            pos_weight=self.label_weight
        )
        
        return loss


# Export functions for compatibility
def logit2prob_multilabel(logit, dim=-1):
    """Convert logits to probabilities for multi-label (sigmoid)"""
    return torch.sigmoid(logit)