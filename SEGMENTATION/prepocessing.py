import torch
import cv2
from SEGMENTATION.predict import *
import glob
import pandas as pd
import ast
import os
import imutils
import tqdm


# load path image and pretrained

net = Inference(size=320, weight="SEGMENTATION/weights/u2net_bce.pth", cuda=True)

def preprocessing(image_path) :
    image = cv2.imread(image_path)
    image_copy= image.copy()
    # predict image
    predict_np,predict = net.inference_remove_item_background(image_path)
    # contour
    gray = predict_np*255
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    _,contours1, hierarchy = cv2.findContours(np.array(thresh,np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt1 = max(contours1, key = cv2.contourArea)
    x1,y1,w1,h1 = cv2.boundingRect(cnt1)
    # 
    rect1 = cv2.minAreaRect(cnt1)
    angle = rect1[-1]
    if angle <-45 :
	    angle = 90+angle
    (h, w) = predict.shape[:2]
    center = (w // 2, h // 2)
    # rotate image
    image_crop = imutils.rotate(image, angle)
    rotated_py = imutils.rotate(gray, angle)
    ret, rotated_py = cv2.threshold(rotated_py, 127, 255, 0)
    _,contours2, hierarchy = cv2.findContours(np.array(rotated_py,np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt2 = max(contours2, key = cv2.contourArea)
    x2,y2,w2,h2 = cv2.boundingRect(cnt2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    crop = image_copy[y2:y2+h2,x2:x2+w2]
    return crop
