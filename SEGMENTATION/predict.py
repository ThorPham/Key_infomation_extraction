from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from SEGMENTATION.model import U2NET
from skimage import io, transform, color
import sys
import shutil
import os
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image
import glob
import time
from SEGMENTATION.data_loader import RescaleT, ToTensor, ToTensorLab, SalObjDataset
import imutils

class Inference:
    '''
    Inference model U square for segmentationq
    '''

    def __init__(self, size=320, weight="weights/u2net.pth", cuda=False):
        self.size = size
        self.model = U2NET(3, 1)
        self.model.load_state_dict(torch.load(weight, map_location="cpu"))
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.model.eval()

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    def inference_remove_item_background(self, image_path):
        img = cv2.imread(image_path)
        h, w, c = img.shape
        image = transform.resize(img, (self.size, self.size), mode='constant')
        image = image / np.max(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        if torch.cuda.is_available():
            image = Variable(image.cuda())
        else:
            image = Variable(image)

        d1, d2, d3, d4, d5, d6, d7 = self.model(image)
        pred = d1[:, 0, :, :]
        pred = self.normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        imo = Image.fromarray(predict_np * 255).convert('RGB')
        imo = cv2.resize(np.array(imo), (img.shape[1], img.shape[0]), interpolation=Image.BILINEAR)
        result =  np.append(img, np.array(imo)[...,:1], axis=2)
        predict_np = cv2.resize(predict_np, (img.shape[1], img.shape[0]), interpolation=Image.BILINEAR)

        return predict_np,result


if __name__ == "__main__":
    import glob
    import pandas as pd
    net = Inference(size=320, weight="weights/u2net_bce.pth", cuda=True)
    paths = glob.glob("/home/thorpham/Desktop/OCR/processing_data/mcocr2021_private_test_data/mcocr_private_test_data/test_images/*")
    # df = pd.read_csv("/home/thorpham/Desktop/OCR/processing_data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df.csv")
    for path in paths :
        image = cv2.imread(path)
        name = os.path.basename(path)
        predict_np,predict = net.inference_remove_item_background(path)
        gray = predict_np*255
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(np.array(thresh,np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # # cv2.drawContours(predict,[box],0,(0,0,255),2)
        angle = rect[-1]
        print("angle",angle)
        p=np.array(rect[1])
        if angle <-45 :
            angle = 90+angle
        # (h, w) = predict.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # # rotated = cv2.warpAffine(predict, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        image_crop = imutils.rotate(image, angle)
        rotated_py = imutils.rotate(gray, angle)
        ret, rotated_py = cv2.threshold(rotated_py, 127, 255, 0)
        contours, hierarchy = cv2.findContours(np.array(rotated_py,np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        crop = image[y:y+h,x:x+w]


        # crop image

        # cv2.imshow("im",predict)
        # # cv2.imshow("rotate",rotated)
        # cv2.imshow("crop",crop)
        cv2.imwrite(f"/home/thorpham/Desktop/OCR/processing_data/private_test/images/{name}",crop)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
