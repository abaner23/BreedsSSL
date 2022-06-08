# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:26:19 2021

@author: barna
"""

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128, num_classes= 13, dim_in=2048):
        super(SupConResNet, self).__init__()
        model = models.resnet50(pretrained = False, num_classes = num_classes)
        self.encoder = nn.Sequential(*(list(model.children())[:-1]))
        self.dim_in = dim_in
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        b_size = feat.shape[0]
        feat = torch.reshape(feat, (b_size, self.dim_in))
        feat = F.normalize(self.head(feat), dim=1)
        print(feat, flush = True)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=13, feat_dim = 2048):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
