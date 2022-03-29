from torch import nn as  nn
from torchvision import models
from torchsummary import summary
import torch

class Network(nn.Module):
    def __init__(self,num_class):
        super(Network,self).__init__()
        self.model = models.resnet101(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,num_class)
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Network(4).to(device)
    # input =  torch.rand((3,256,256),dtype=torch.float32).to(device)
    summary(net,input_size=(3,256,256))

