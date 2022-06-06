# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:35:50 2021

@author: barna
"""

from torch.utils.data import Dataset
import os
import random 
from PIL import Image
import numpy as np
import torch 

def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
    
def make_dataset(root, train_subclasses,n, test_subclasses, c, label_map):
    images = []
    for metaclass,subclass in enumerate(train_subclasses,0):
       for classidx in subclass:
           folder_path = root+ '/'+label_map[classidx]
           if(n == -1):
              for fname in os.listdir(folder_path):
                 item = (folder_path+'/'+fname,metaclass)
                 images.append(item)
           else:
              for fname in random.sample(os.listdir(folder_path), n):
                 item = (folder_path+'/'+fname,metaclass)
                 images.append(item)

    
    
    for metaclass,subclass in enumerate(test_subclasses,0):
       for classidx in subclass:
          folder_path = root+ '/'+label_map[classidx]
          if(c == -1):
              for fname in os.listdir(folder_path):
                 item = (folder_path+'/'+fname,metaclass)
                 images.append(item)
          else:
              for fname in random.sample(os.listdir(folder_path), c):
                 item = (folder_path+'/'+fname,metaclass)
                 images.append(item)

    images_np = np.array(images)
    return images_np

class ImageNetDataset(Dataset):
    
    def __init__(self,root, train_index,n, test_index, c,train_transform, label_map, loader = default_loader):
        
        samples= make_dataset(root, train_index,n, test_index, c, label_map)
        self.samples = samples
        self.loader = loader
        self.transform = train_transform
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        image_sample = self.loader(path)
        image_sample = self.transform(image_sample)
        target = torch.tensor(target.astype(int))
        return image_sample,target
    
    def __len__(self):
        return (self.samples.shape[0])
            