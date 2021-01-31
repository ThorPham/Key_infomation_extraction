import glob
import os
import numpy  as np
import cv2
import pandas as pd
import json
import ast

# index2label = {
# 15: "company",
# 16: "address",
# 17: "date",
# 18: "total"
# }
index2label ={
	"SELLER" : "company",
	"ADDRESS":"address",
	"TIMESTAMP":"date",
	"TOTAL_COST":"total"
}

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

paths = glob.glob("/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_train_processing/data_crop_and_rotate/*.txt")
for i in range(len(paths)) :
    name = os.path.basename(paths[i]).split(".")[0]
    path_img = f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_train_processing/images/{name}.jpg"
    with open(paths[i],"r") as f :
        data_lines = f.readlines()

    image = cv2.imread(path_img)
    data_label = []
    for line in data_lines:
        tmp = line.strip().split("\t")
        bbox = [int(i) for i in tmp[:4]]
        text = tmp[4]
        class_text = tmp[5]
        xmin_1,ymin_1,w,h = bbox
        xmax_1,ymax_1 = xmin_1 + w, ymin_1 + h
        class_id = index2label[class_text]
        tmp1 = [xmin_1,ymin_1,xmax_1,ymax_1,text,class_id]
        data_label.append(tmp1)
        # cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,255,0.4),1)

    with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_train_processing/CTPN_OCR/{name}.txt","r") as file :
        data_ocr = file.readlines()
    label_ocr = []
    for d in data_ocr :
        tmp3 = d.strip().split("\t")
        [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax] = [int(i) for i in tmp3[:8]] 
        ocr_label = tmp3[8]
        label_ocr.append([xmin,ymin,xmax,ymax,ocr_label,"other"])
        # cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,255,0.4),1)
    remove = []
    for i in label_ocr :
        x11,y11,x12,y12 = i[:4]
        for j in data_label :
            x21,y21,x22,y22  = j[:4]
            iou = bb_intersection_over_union(np.array([x11,y11,x12,y12]),np.array([x21,y21,x22,y22]))
            if iou > 0.2 :
                remove.append(i)

    label_ocr.extend(data_label)
  
    results = []
    for d in label_ocr :
        if d not in remove :
            results.append(d)
    if results is None :
        continue
    with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_train_processing/train/{name}.txt","w") as f :
        for b in results :
            (xmin,ymin,xmax,ymax,text,class_id) =  b
            if len(text) <2 :
                continue
            ttp = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,text,class_id]
            f.write("\t".join(str(i) for i in ttp) +"\n")
    # for b in results :
    #     (xmin,ymin,xmax,ymax,text,class_id) =  b
    #     cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,255,0.4),1)
    #     cv2.putText(image, str(class_id) , (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0,0.3), 1, cv2.LINE_AA)
    # cv2.imshow("im",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()