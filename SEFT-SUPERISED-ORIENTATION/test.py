import torch
import numpy as np
import glob
import random
from dataset import MyDataset
from net import Network
import  cv2
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = [0,1,2,3]
angles = [0,90,180,270]
transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# net = Network(num_class=4).to(device)
net = torch.load("weights.pth"
)
net.eval()
path_images = glob.glob("images/*")
for path in path_images :
    img = Image.open(path)
    label = random.choice(labels)
    img = img.rotate(angles[label],expand=1)
    img_show = img
    img =  transform(img)
    img = torch.unsqueeze(img,0)
    pre = net(img.to(device))
    _,label_id = torch.max(pre,1)
    print(label_id)
    cv2.imshow("img",np.array(img_show))
    cv2.waitKey(0)
cv2.destroyAllWindows()
