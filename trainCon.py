import torch
import os

def train_con(model,n_epochs,train_data,optimizer, scheduler, loss_function, device, num_classes, model_save, filename, weighted):
    model.train()
    step = 0
    min_loss = 1000
    for epoch in range(0,n_epochs):
        epoch_loss = 0.0
        for index,batch_data in enumerate(train_data,0):
            nat_inputs, labels = batch_data
            nat_inputs = torch.cat([nat_inputs[0], nat_inputs[1]], dim=0)
            nat_inputs = nat_inputs.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]
            
            optimizer.zero_grad()
            
            features = model(nat_inputs)
            print(features, flush = True)
            print(features.shape, flush = True)
            f1,f2 = torch.split(features,[bsz,bsz],dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if(weighted):
               loss = loss_function(index, len(train_data),num_classes, features, labels)
            else:
               loss = loss_function(features, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / len(train_data)
        # print statistics
        print('epoch: %d, Epoch loss: %.3f' % (epoch+1, epoch_loss), flush = True)
        scheduler.step()
        
        if((epoch+1)% 10 == 0):        
            if(epoch_loss< min_loss):
               print('Saving model', flush = True)
               torch.save(model.state_dict(), os.path.join(model_save, filename+ '.pt'))
               min_loss = epoch_loss
               
    print(' Finished Training \n\n', flush = True)
   

