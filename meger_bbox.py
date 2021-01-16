import cv2
import numpy as np
import csv
import os,glob
import itertools
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from argparse import Namespace

import sys
import os
import time
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

# convert bb 8 point to 4 point
def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)

def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)

# merger bbox
def convert_box_to_rectangle(path):
    ''' Convert 8 point to 4 point'''
    bbox = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:-1]))
            #bbox.append([x1, y1, x2, y2, x3, y3, x4, y4])
            polys = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
            new  = rectangle_from_parallelogram(polys)
            new = np.array(new,dtype =np.int32)
            rectangle = new[[0,2]].flatten()
            bbox.append(rectangle)
    return bbox
def calc_distance(bbox1, bbox2):
    mean_y_1 = (bbox1[1] + bbox1[3]) / 2
    mean_y_2 = (bbox2[1] + bbox2[3]) / 2
    
    return abs(mean_y_1 - mean_y_2)

def filter_box(bboxes,threshold=10):
    '''Filter box same horizontal with threshold defaul 10
        return list bb '''
    group = {}
    group_id = 1
    threshold = 10
    bboxes_set = set(tuple(bbox) for bbox in bboxes)


    while bboxes_set:
        pivot = bboxes_set.pop()
        group[group_id] = [pivot]

        toremove = set()
        for bbox in bboxes_set:
            if calc_distance(pivot, bbox) < threshold:
                group[group_id] += [bbox]
                toremove.add(bbox)
        bboxes_set -= toremove
        group_id += 1
    return group

def merger_bbox(group):
    '''Transform box
        Return meger box'''
    final_box = []
    for i,k in group.items() :
        x1= []
        y1 = []
        x2 = []
        y2 = []
        for j in k:
            value =  j
            x1.append(value[0])
            y1.append(value[1])
            x2.append(value[2])
            y2.append(value[3])
        xmin = min(x1)
        ymin = min(y1)
        xmax = max(x2)
        ymax = max(y2)
        final_box.append([xmin,ymin,xmax,ymax])
    return final_box


if __name__ == '__main__':
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    import numpy as np
    import tqdm

    # config['weights'] = './weights/transformerocr.pth'
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = '/media/thorpham/PROJECT/OCR-challenge/transformer_ocr/transformerocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    detector = Predictor(config)

    paths = glob.glob("/media/thorpham/PROJECT/OCR-challenge/ctpn/data/crop_rotate/*.txt")
    for pth in tqdm.tqdm(paths) :
        name = os.path.basename(pth).split(".")[0]
        bboxs =  convert_box_to_rectangle(f"/media/thorpham/PROJECT/OCR-challenge/ctpn/data/crop_rotate/{name}.txt")
        image = cv2.imread(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_train_processing/images/{name}.jpg")
        # merger =  filter_box(bbox)
        # bboxs = merger_bbox(merger)
        if len(bboxs) <1:
            continue
        with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_train_processing/CTPN_OCR/{name}.txt","w") as f:
            for box in bboxs:
                (xmin,ymin,xmax,ymax) =  box
                img_crop = image[ymin:ymax,xmin:xmax]

                img_ocr = Image.fromarray(img_crop)
                s = detector.predict(img_ocr)
                if len(s) < 2 :
                    continue
                text = [xmin,ymin,xmax,ymax,s.strip()]
                save = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,s,"other"]
                f.write("\t".join([str(i) for i in save]) + "\n")


    #     for box in bboxs:
    #         (xmin,ymin,xmax,ymax) =  box
    #         cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255,0.6),1)
    #         img_crop = image[ymin:ymax,xmin:xmax]

    #         img_ocr = Image.fromarray(img_crop)
    #         s = detector.predict(img_ocr)
    #         print(s)
    #     cv2.imshow("image",image)
    #     cv2.waitKey(0)
    # # cv2.imwrite("out1.jpg",image)
    # cv2.destroyAllWindows()
