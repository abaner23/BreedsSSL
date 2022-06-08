# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:40:44 2021

@author: barna
"""




import torch
import os
import argparse
import torch.optim as optim
import torch.nn as nn
from trainCon import train_con
from SupConModel import SupConResNet
from losses import SupConLoss
from losses_weighted import SupConLossWeighted
from dataAugmentation import trainTransform, TwoCropTransform
from utils import breeds_helper
from customImageNetDataset import ImageNetDataset


parser = argparse.ArgumentParser('argument for training')

parser.add_argument('data_dir_train', type=str,
                        help='training directory for ImageNet')

parser.add_argument('data_dir_val', type=str,
                        help='validation directory for ImageNet')

parser.add_argument('info_dir', type=str,help = 'Breeds hierarchy file path')

parser.add_argument('dataset', type=str,help = 'Breeds hierarchy file path')

parser.add_argument('model_pre', help = 'Folder name to store first training of model')

parser.add_argument('model_tune', help = 'Folder name to store fine tuning of model')

parser.add_argument('temperature', type =float, help = 'Temperature for Contrastive Learning')

parser.add_argument('size', type=int, default=224, help='parameter for RandomResizedCrop')

parser.add_argument('--weighted', help = 'Flag to perform weighted training', action = 'store_true')

opt = parser.parse_args()

model_pre = os.path.join(os.getcwd(), opt.model_pre)
model_tune = os.path.join(os.getcwd(), opt.model_tune)

if(not os.path.isdir(model_pre)):
    os.mkdir(model_pre)

if(not os.path.isdir(model_tune)):
    os.mkdir(model_tune)

print(opt.dataset)

filename = opt.dataset

train_subclasses, test_subclasses, label_map_seq, num_classes = breeds_helper(opt.info_dir, opt.data_dir_train, opt.dataset)

print(train_subclasses, flush = True)
print(test_subclasses, flush = True)
print(num_classes, flush = True)

train_subclasses = [train_subclasses[i] for i in [0,5]]
test_subclasses = [test_subclasses[i] for i in [0,5]]
num_classes = 2

print(train_subclasses, flush = True)
print(test_subclasses, flush = True)
print(num_classes, flush = True)

trainset = ImageNetDataset(opt.data_dir_train, train_subclasses,-1, test_subclasses, 0, TwoCropTransform(trainTransform(opt.size)), label_map_seq)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle = True, num_workers= 5)

print(len(train_loader), flush = True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Fetching an untrained model')
model =  SupConResNet(num_classes = num_classes, dim_in = 2048)
model= nn.DataParallel(model)
model = model.to(device)

if torch.cuda.device_count() > 1:
    print('Using multiple GPUs', flush = True)
    

optimizer = optim.SGD(model.parameters(), lr= 0.2,momentum= 0.9, weight_decay = 0.0001)
learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [200,275,325])

if(opt.weighted):
   penalty = 5
   loss_function = SupConLossWeighted(num_classes, 128, penalty, opt.temperature).to(device)
   filename = filename + '_weighted'+str(penalty)
else:
   loss_function = SupConLoss(temperature = opt.temperature).to(device)

# Perform standard training
print('Performing standard pre-training', flush = True)
print(filename, flush = True)

train_con(model, 350, train_loader, optimizer, learning_rate_scheduler, loss_function, device, num_classes, model_pre, filename, opt.weighted)