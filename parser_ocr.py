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
import numpy as np
import tqdm
import os
# config['weights'] = './weights/transformerocr.pth'
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = '/media/thorpham/PROJECT/OCR-challenge/transformer_ocr/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

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

import glob
import tqdm
paths = glob.glob("/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/private_test/images/*")
for image_path in tqdm.tqdm(paths) :
    name = os.path.basename(image_path)
    base_name = os.path.basename(image_path).replace("jpg","txt")
    image = read_image(image_path)
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
    with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/private_test/OCR_PARSER/{base_name}","w") as f :
        for ind, region in enumerate(regions):
            result = rectify_poly(image, region)
            img_ocr = Image.fromarray(result)
            s = detector.predict(img_ocr)
            poly = np.array(region).astype(np.int32).reshape((-1))
            if len(s)<2 :
                continue
            strResult = '\t'.join([str(p) for p in poly]) +  "\t" + str(s) + "\t" + "other" + '\n'
            f.write(strResult)

            box = np.array(region,np.int32)
            box = box.reshape((-1, 1, 2)) 
            poly = np.array(region).astype(np.int32).reshape((-1))
            cv2.polylines(image_visualize, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        cv2.imwrite(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/test-craft-visualize/{name}",image_visualize)
#         print(s)
#     cv2.imshow("image",cv2.resize(image,(600,800)))
#     cv2.waitKey(0)
# cv2.destroyAllWindows()