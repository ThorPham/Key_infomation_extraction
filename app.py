from SEGMENTATION.prepocessing import preprocessing
import numpy as np 
import cv2
import os
import torch
import PIL
from PIL import Image
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import numpy as np
import cv2

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from GRAPH_MODEL.graph_predict import *


config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'OCR-challenge/weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

# load model graph
node_labels = ['other', 'company', 'address', 'date', 'total']
alphabet = ' "$(),-./0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZ_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
weight = 'weights/graph_weight.pkl'

device = 'cuda'
graph_model = GRAPH_MODEL(node_labels,alphabet,weight,device)


def rectify_poly(img, poly):
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32([[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]])
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32([[width_step, 0], [width_step + w - 1, height - 1], [width_step, height - 1]])
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1)
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
    return output_img

path_image = "/home/thorpham/Downloads/z1992150214667_157026289e86eb457a50058f03f93671_iksx.jpg"
image = preprocessing(path_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_visualize = image.copy()

prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )
regions=prediction_result["boxes"]
data_graph = []
for ind, region in enumerate(regions):
    result = rectify_poly(image, region)
    img_ocr = Image.fromarray(result)
    s = detector.predict(img_ocr)
    poly = np.array(region).astype(np.int32).reshape((-1))
    if len(s)<2 :
        continue
    box = np.array(region,np.int32)
    box = box.reshape((-1, 1, 2)) 
    poly = np.array(region).astype(np.int32).reshape((-1))
    cv2.polylines(image_visualize, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    strResult = '\t'.join([str(p) for p in poly]) +  "\t" + str(s) + "\t" + "other"
    data_graph.append(strResult)


pre = graph_model.predict(data_graph)
for i in pre :
    print(i)
cv2.imshow("img",image_visualize)
cv2.waitKey(0)
cv2.destroyAllWindows()
















# if __name__ == "__main__":
#     path_image = "/media/thorpham/PROJECT/OCR-challenge/data_train/mcocr_public_train_test_shared_data/mcocr_train_data/train_images/mcocr_public_145013aaprl.jpg"
#     image = preprocessing(path_image)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     cv2.imshow("img",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()