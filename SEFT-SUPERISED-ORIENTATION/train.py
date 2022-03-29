import torch
import numpy as np
import glob
from torch import nn as nn
from torch.optim import Adam
from dataset import MyDataset
from net import Network
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# Hyper parameters
Epochs = 50
batch_size = 32
learning_rate = 0.001
# Dataloader
path_images = glob.glob("images/*")
data_train = MyDataset(path_images)
data_test = MyDataset(path_images)
dataloader_train = DataLoader(data_train,batch_size=64,shuffle=True)
dataloader_test = DataLoader(data_test,batch_size=batch_size,shuffle=True)

net = Network(num_class=4).to(device)
# Summary model
summary(net,input_size=(3,256,256))
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(),lr=learning_rate)

@torch.no_grad()
def test(model,data_loader,device):
    model.eval()
    running_corrects = 0
    for (images,labels) in tqdm(dataloader_test):
        images = images.to(device)
        labels = labels.to(device)
        predicts = model(images)
        _, predicted = torch.max(predicts, 1)
        running_corrects += torch.sum((predicted == labels).squeeze())

    epoch_acc = running_corrects / len(data_loader.dataset)
    print('Test  Acc : {:.4f}'.format(epoch_acc))
    return epoch_acc

def train(model,data_loader,loss_fn,optimizer,device):
    running_corrects = 0
    running_loss = 0.0
    for (images,labels) in tqdm(dataloader_test):
        images = images.to(device)
        labels = labels.to(device)
        predicts = model(images)
        loss = loss_fn(predicts,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(predicts, 1)
        running_corrects += torch.sum((predicted == labels).squeeze())

    epoch_acc = running_corrects / len(data_loader.dataset)
    epoch_loss = running_loss / len(data_loader.dataset)
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss

for epoch in range(Epochs):
    print("-"*50)
    print("Epoch ",epoch)
    train(net,dataloader_train,loss_fn,optimizer,device)
    if epoch % 5 == 0 :
        test(net,dataloader_test,device)
    print("-"*50)

torch.save(net, "weights.pth")







