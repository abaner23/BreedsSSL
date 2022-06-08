# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:31:42 2020

@author: barna
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import os
import numpy as np
        
def train_CE(model,n_epochs,train_data,optimizer, scheduler, loss_function,source_validation, target_validation, device, num_classes, model_save, dataset):
    model.train()
    writer = SummaryWriter(f'runs/ImageNet/tensorboard')
    step = 0
    max_overall = 0.0
    print('Entering here', flush = True)
    conv1= model.module.conv1
    print(conv1.weight.requires_grad, flush = True)

    for epoch in range(0,n_epochs):
        epoch_loss = 0.0
        for index,batch_data in enumerate(train_data,0):
            nat_inputs, labels = batch_data
            nat_inputs = nat_inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()        
            output = model(nat_inputs)
            loss = loss_function(output,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
           
        # print statistics
        print('epoch: %d, Epoch loss: %.3f' % (epoch+1, epoch_loss / len(train_data)), flush = True)
        scheduler.step()
        
        if((epoch+1)% 10 == 0):
            source_accuracy= calculateAccuracy(model,source_validation,device,writer,0,epoch+1, num_classes)
            target_accuracy= calculateAccuracy(model,target_validation,device,writer,1,epoch+1, num_classes)
            model.train()
            if(target_accuracy> max_overall):
               print('Saving model', flush = True)
               torch.save(model.state_dict(), os.path.join(model_save, dataset+ '.pt'))
               max_overall = target_accuracy
            
    print(' Finished Training \n\n', flush = True)
    
def calculateAccuracy(model,test_data,device,writer,accuracy_type,epoch,num_classes):
    model.eval()
    total = 0
    correct = 0
    n_classes = num_classes
    
    confusion_matrix = torch.zeros(n_classes, n_classes)
    batch = 0
    for data in test_data:
          test_inputs, labels = data
          batch = batch + 1
          with  torch.no_grad():    
                test_inputs= test_inputs.to(device)
                labels= labels.to(device)
                outputs = F.softmax(model(test_inputs).data, dim= 1)
                maxval, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct+=(predicted == labels).sum().item()
            
                indx = []
                cntr = 0
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    if(t.long() != p.long()):
                        indx.append(cntr)
                    cntr = cntr + 1

    if(accuracy_type == 0):
          print(' Source Accuracy on %d standard test images: %d %%' % (total,100.0 * correct / total), flush = True)
          print('Per class accuracy ', flush = True)
          print(confusion_matrix.diag()/confusion_matrix.sum(1), flush = True)
    else:
          print(' Target Accuracy on %d standard test images: %d %%' % (total,100.0 * correct / total), flush = True)
          print('Per class accuracy ', flush = True)
          print(confusion_matrix.diag()/confusion_matrix.sum(1), flush = True)

    return 100.0 * correct / total