import glob
import os
import numpy  as np
import cv2
import pandas as pd
import json
import ast
path_image = "/media/thorpham/PROJECT/OCR-challenge/preprocessing/graph_model/visual_test"
path = "/home/thorpham/snap/skype/common/results.csv"
data = pd.read_csv(path)
for i in range(len(data)):
    img = data["img_id"].iloc[i]
    labels = data["anno_texts"].iloc[i]
    image = cv2.imread(os.path.join(path_image,img))
    print("="*50,os.path.join(path_image,img))

    print(labels)
    cv2.imshow("image",cv2.resize(image,(400,800)))
    cv2.waitKey(0)
cv2.destroyAllWindows()