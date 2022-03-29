
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader
import glob
class MyDataset(Dataset):
    def __init__(self,list_images):
        self.list_images = list_images
        self.labels = [0,1,2,3]
        self.angles = [0,90,180,270]
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        path = self.list_images[item]
        img = Image.open(path)
        label = random.choice(self.labels)
        img = img.rotate(self.angles[label],expand=1)
        img =  self.transform(img)
        # img = img.permute(2,0,1)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.list_images)

if __name__ == "__main__":
    images = glob.glob("images/*")[:1]
    data = MyDataset(images)
    dataloader = DataLoader(data,batch_size=1)
    for (img,label) in dataloader:
        print(img.shape)
        print(label)



