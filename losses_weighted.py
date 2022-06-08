"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from utils import AvgSimilarity

class SupConLossWeighted(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,num_classes, bsz,penalty = 2, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossWeighted, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.num_classes = num_classes
        self.class_mask = torch.ones((2*bsz, 2*bsz))
        self.penalty = penalty
        self.avg = AvgSimilarity(num_classes)
        self.avg_sim_score = torch.ones((num_classes, num_classes))
        self.avg_sim_score.fill_diagonal_(0)
 
    def forward(self, index, length, num_classes, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            dup_labels = labels
            labels = labels.contiguous().view(-1, 1)         
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T)

        dup_labels = dup_labels.repeat(contrast_count)
        dup_labels = dup_labels.contiguous().view(-1, 1)
        classes = torch.arange(0,self.num_classes).view(1, self.num_classes).to(device)
        mask1 = torch.eq(dup_labels,classes)*1.0

        if(index% 20 == 0):
            self.avg.updateSimilarityScore(similarity_matrix, dup_labels,  mask1)
        
        if(index == length - 1):
            self.avg_sim_score = self.avg.getSimilarityScore(length)
            print(self.avg_sim_score, flush = True)
            self.avg.clean()
            self.avg_sim_score.fill_diagonal_(0)

        self.avg_sim_score = self.avg_sim_score.to(device)
        mask2 = torch.matmul(self.avg_sim_score,mask1.T)
        self.class_mask = torch.matmul(mask1, mask2).detach()
        self.class_mask = 1 + torch.div(self.class_mask, self.penalty)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature/self.class_mask)
            
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
