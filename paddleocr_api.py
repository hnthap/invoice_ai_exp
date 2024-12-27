'''
⛔⛔⛔

UNUSABLE CODE: DO NOT ATTEMPT

⛔⛔⛔
'''

import os

import cv2
import numpy as np

from ndcuong19_mcocr.mc_ocr.text_detector.PaddleOCR.tools.infer import (
    predict_det,
    utility as infer_utility,
)
from ndcuong19_mcocr.mc_ocr.text_detector.PaddleOCR.ppocr.utils import utility


def detect(
    image_file: str, 
    *, 
    model_dir='weights/ch_ppocr_server_v2.0_det_infer',
    visualize=True,
    db_thresh=0.3,
    db_box_thresh=0.3,
    out_viz_dir='out/detect/viz_imgs',
    out_txt_dir='out/detect/txt',
):
    if visualize:
        os.makedirs(out_viz_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)
    args = infer_utility.parse_args()
    args.det_model_dir = model_dir
    args.det_db_thresh = db_thresh
    args.det_db_box_thresh = db_box_thresh
    args.use_gpu = False
    detector = predict_det.TextDetector(args)
    image, flag = utility.check_and_read_gif(image_file)
    if not flag:
        image = cv2.imread(image_file)
    if image is None:
        return { 'success': False, 'message': 'Failed to load image' }
    dt_boxes, elapse = detector(image)
    pruned_image_name = os.path.split(image_file)[-1]
    output_txt_path = os.path.join(
        out_txt_dir, 
        pruned_image_name.replace('.jpg', '.txt'),
    )
    source_image = infer_utility.draw_text_det_res(
        dt_boxes,
        image_file,
        save_path=output_txt_path,
    )
    if visualize:
        image_path = os.path.join(out_viz_dir, pruned_image_name)
        cv2.imwrite(image_path, source_image)
    return { 'success': True, 'boxes': dt_boxes }
        

if __name__ == '__main__':
    detect('assets/sample-003.jpg')
