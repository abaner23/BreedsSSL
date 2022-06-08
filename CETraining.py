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
from trainCE import train_CE
from utils import breeds_helper
from customImageNetDataset import ImageNetDataset
from dataAugmentation import TEST_TRANSFORMS_IMAGENET, trainTransform, TRAIN_TRANSFORMS_IMAGENET
import torchvision.models as models

parser = argparse.ArgumentParser('argument for training')

parser.add_argument('data_dir_train', type=str,
                        help='training directory for ImageNet')

parser.add_argument('data_dir_val', type=str,
                        help='validation directory for ImageNet')

parser.add_argument('info_dir', type=str,help = 'Breeds hierarchy file path')

parser.add_argument('dataset', type=str,help = 'Breeds hierarchy file path')

parser.add_argument('model_pre', type=str, help = 'Folder name to store first training of model')

parser.add_argument('model_tune', type=str, help = 'Folder name to store fine tuning of model')

parser.add_argument('size', type=int, default=224, help='parameter for RandomResizedCrop')

parser.add_argument('--fine_tune', help = 'Flag to first[0]/second[1] training phase', action = 'store_true')

parser.add_argument('--feature_extract', help = 'Flag to fine tune the last layer/entire model', action= 'store_true')


opt = parser.parse_args()


model_pre = os.path.join(os.getcwd(), opt.model_pre)
model_tune = os.path.join(os.getcwd(), opt.model_tune)

if(not os.path.isdir(model_pre)):
    os.mkdir(model_pre)

if(not os.path.isdir(model_tune)):
    os.mkdir(model_tune)

train_subclasses, test_subclasses, label_map_seq, num_classes = breeds_helper(opt.info_dir, opt.data_dir_train, opt.dataset)

if(not opt.fine_tune):
    trainset = ImageNetDataset(opt.data_dir_train, train_subclasses,-1, test_subclasses, 0, trainTransform(opt.size), label_map_seq)
else:
    print('Fine-Tuning on test subclasses', flush = True)
    trainset = ImageNetDataset(opt.data_dir_train, [], test_subclasses, 50, trainTransform(opt.size), label_map_seq)
    print(len(trainset), flush = True)

source_validation = ImageNetDataset(opt.data_dir_val, [], -1, train_subclasses, 50, TEST_TRANSFORMS_IMAGENET, label_map_seq)
target_validation = ImageNetDataset(opt.data_dir_val, [], -1, test_subclasses, 50, TEST_TRANSFORMS_IMAGENET, label_map_seq)

    

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle = True, num_workers= 5)
source_val_loader = torch.utils.data.DataLoader(source_validation, batch_size=10, shuffle = True, num_workers= 2)
target_val_loader = torch.utils.data.DataLoader(target_validation, batch_size=10, shuffle = True, num_workers= 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.module.fc.parameters():
        param.requires_grad = True


if(opt.fine_tune):
    model =  models.resnet50(pretrained = False, num_classes = num_classes)
    model= nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(model_pre, 'train.pt')))
    if(opt.feature_extract):
        freeze_layers(model)
        #model.module.fc = nn.Linear(2048, num_classes)
else:
    if(os.path.isfile(os.path.join(model_pre, opt.dataset+ '.pt'))):
        print('Fetching a previously trained model' , flush = True)
        model =  models.resnet50(pretrained = False, num_classes = num_classes)
        model= nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(model_pre, 'train.pt')))
    else:
        print(' Fetching an untrained model')
        model =  models.resnet50(pretrained = False, num_classes = num_classes)
        model= nn.DataParallel(model)

if torch.cuda.device_count() > 1:
    print('Using multiple GPUs', flush = True)
    
model = model.to(device)

if(opt.fine_tune):
    print("Params to learn:")
    optimizer = optim.SGD(model.module.fc.parameters(), lr= 0.1,momentum= 0.9,  weight_decay = 0.0001)
    learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [120])
else:
    optimizer = optim.SGD(model.parameters(), lr= 0.05,momentum= 0.9, weight_decay = 0.0001)
    learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [120,180,225])

loss_function = nn.CrossEntropyLoss()


# Perform standard training
if(not opt.fine_tune):
   print('Performing standard pre-training', flush = True)
   train_CE(model,240, train_loader, optimizer, learning_rate_scheduler, loss_function, source_val_loader,target_val_loader, device, num_classes, model_pre, opt.dataset)
else:
   print(' Performing fine tuning', flush = True) 
   train_CE(model, 100, train_loader, optimizer, learning_rate_scheduler, loss_function, source_val_loader,target_val_loader, device, num_classes, model_tune, opt.dataset)