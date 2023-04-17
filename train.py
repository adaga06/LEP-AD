import sys, os
import torch
import torch.nn as nn
import wandb
# training function at each epoch
scaler = torch.cuda.amp.GradScaler()
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 100
    TRAIN_BATCH_SIZE = 512
    loss_fn = torch.nn.MSELoss()
    
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        # data_mol = [item.to(device) for item in data[0]]
        # data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data_mol, data_pro)
            # print(data_mol)
            # print(data_mol.y)
            # labels= [sample.y.float().to(device) for sample in data_mol]
            # labels=torch.stack(labels).view(-1, 1)
            labels = data_mol.y.view(-1, 1)
            # print(output.shape,labels.shape)
            loss = loss_fn(output, labels)
        #loss.backward()
        scaler.scale(loss).backward()
        wandb.log({"loss per batch": loss})
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            labels = data_mol.y.view(-1, 1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)

            total_labels = torch.cat((total_labels, labels.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()