# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:05:19 2021

@author: barna
"""
from PIL import Image
from robustness.tools.breeds_helpers import ClassHierarchy
from robustness.tools.breeds_helpers import make_entity13,make_entity30,make_living17, make_nonliving26
import os
import numpy as np
import torch 

class AvgSimilarity:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.avg_sim = torch.zeros((num_classes, num_classes))
        self.cnt = 0
                
    def updateSimilarityScore(self, features, labels, mask1):
        self.cnt = self.cnt + 1
        mask_sum = torch.sum(mask1, dim = 0)
        classwise_sim = torch.matmul(features,mask1)
        update = torch.empty((self.num_classes, self.num_classes))
        
        for i in range(0, self.num_classes):
             class_sim = classwise_sim[mask1[:,i].bool()]
             divby = (class_sim.shape[0]*mask_sum)
             if(class_sim.shape[0] > 0):
                 sim = torch.div(torch.sum(class_sim,dim = 0), divby)
                 sim[torch.isnan(sim)] = 0
             else:
                 sim = torch.zeros_like(update[i])
             update[i] = sim
     
        self.avg_sim = self.avg_sim.detach() + update        
              
    def getSimilarityScore(self, bsz):
        return torch.div(self.avg_sim, self.cnt).detach()

    def clean(self):
        self.cnt = 0
        self.avg_sim.zero_()

def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def breeds_helper(info_dir, data_dir_train, dataset):
    hier = ClassHierarchy(info_dir)
    if(dataset == 'entity13'):
        ret = make_entity13(info_dir, split="rand")
    elif(dataset == 'entity30'):
        ret = make_entity30(info_dir, split="rand")
    elif(dataset == 'living17'):
        ret = make_living17(info_dir, split="rand")
    elif(dataset == 'nonliving26'):
        ret = make_nonliving26(info_dir, split="rand")
    else:
        raise ValueError('Unknown dataset')

    superclasses, subclass_split, label_map = ret
    train_subclasses, test_subclasses = subclass_split

    folder_list = os.listdir(data_dir_train)
    label_map_seq = {}
    c= 0
    for directories in sorted(folder_list):
        label_map_seq[c] = directories
        c = c + 1
    return train_subclasses, test_subclasses, label_map_seq, len(superclasses)
