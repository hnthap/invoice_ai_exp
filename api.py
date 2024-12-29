print('😳 Import dependencies')

import base64
import gc
from io import BytesIO
from typing import Annotated
import warnings

from PIL import Image
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from torch.backends import cudnn

from clovaai_craft.craft import CRAFT
from clovaai_craft import imgproc
from clovaai_craft.refine_net import RefineNet
from layoutlm_api import (
    load_layoutlm_v3,
    load_layoutlm_v3_processor,
    tag,
)
from phobert import (
    infer_phobert,
    load_phobert_model,
    load_phobert_tokenizer,
)
from scene_text import (
    load_weights,
    detect_text,
    detect_text2,
    crop_boxes,
)
from vietocr_api import load_vietocr_detector, Predictor


warnings.filterwarnings('ignore')


# def base64_to_image(raw: str) -> np.ndarray:
#     '''
#     https://stackoverflow.com/a/48056396
#     '''
#     raw = raw.lstrip('data:image/jpeg;base64,').strip()
#     print(raw[-10:])
#     padding_len = 4 - (len(raw) % 4)
#     if padding_len != 4:
#         print('len(raw) = %d\tpadding_len = %d' % (len(raw), padding_len))
#         raw += '=' * padding_len
#     print('len(raw) = %d' % (len(raw)))
#     print('🍮')
#     data = base64.b64decode(raw)
#     with open('temp.txt', 'wt') as f: f.write(raw)
#     print(raw[:10])
#     print(raw[-10:])
#     print('⭐')
#     with open('temp.jpg', 'wb') as f: f.write(data)
#     with Image.open('temp.jpg') as obj:
#         print('🦆')
#         return np.array(obj)
#         # return cv2.cvtColor(np.array(obj), cv2.COLOR_BGR2RGB)
    

def detect_text_wrapper(net, *, refine_net, image, device, enables_rotate):
    if enables_rotate:
        new_image, boxes = detect_text2(net, image, device=device,
                                        refine_net=refine_net)
        if len(boxes) > 0: # If detected some texts
            image = new_image
    else:
        boxes = detect_text(net, image, device=device, refine_net=refine_net)
    return image, boxes


def extract_texts(
    detector: Predictor,
    image: np.ndarray,
    boxes: list[np.ndarray],
):
    pieces = crop_boxes(image.astype(np.uint8), boxes)
    objs = [Image.fromarray(p.astype(np.uint8)) for p in pieces]
    texts = detector.predict_batch(objs)
    piece_images = [image_to_base64(obj) for obj in objs]
    for obj in objs:
        obj.close()
    return [
        { 'box': b.tolist(), 'piece': p, 'text': t }
        for b, p, t in
        zip(boxes, piece_images, texts)
    ]


def get_device() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def image_to_base64(image: Image.Image | np.ndarray) -> str:
    '''
    https://stackoverflow.com/a/16066722
    '''
    gets_pil_image = isinstance(image, Image.Image)
    if not gets_pil_image:
        image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, format='JPEG')
    data = output.getvalue()
    encoded_data = base64.b64encode(data)
    if not isinstance(encoded_data, str):
        encoded_data = encoded_data.decode()
    # data_url = 'data:image/jpg;base64,' + encoded_data
    data_url = encoded_data
    if not gets_pil_image:
        image.close()
    return data_url


def load_craft(*, device=None) -> CRAFT:
    # Load CRAFT
    net = CRAFT().to(device) # net = nn.DataParallel(net)
    cudnn.benchmark = False
    net = load_weights(net, 'weights/craft_mlt_25k.pth', device=device)
    # Load refine net
    refine_net = RefineNet().to(device)
    refine_net = load_weights(
        refine_net, 'weights/craft_refiner_CTW1500.pth', device=device)
    return net, refine_net


def load_detector(*, device: str) -> Predictor:
    '''
    This function loads a pre-trained Vietnamese OCR detector for text 
    extraction. It uses the provided device (either CPU or GPU) for inference.

    Args:
        device (str): The device to be used for inference.

    Returns:
        Predictor: An instance of the Vietnamese OCR detector, pre-trained and
            ready for text extraction.
    '''
    return load_vietocr_detector(device=device)


# def normalize_box_for_visualization(box, width, height):
#     x1, y1 = box[0]
#     x2, y2 = box[2]
#     return [
#         special_round(x1, 0, width),
#         special_round(y1, 0, height),
#         special_round(x2, 0, width),
#         special_round(y2, 0, height),
#     ]


# def visualize_image_with_boxes(image, boxes, color=(0, 255, 0)):
#     width, height = image.shape[1], image.shape[0]
#     t = image.copy().astype(np.int32)
#     for box in boxes:
#         x1, y1, x2, y2 = normalize_box_for_visualization(box, width, height)
#         points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)]
#         for i in range(1, len(points)):
#             t = cv2.line(t, points[i - 1], points[i], color,
#                      max(2, min(image.shape[:2]) // 200))
#     return t


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


flush()
device = get_device()

print('🍞 Load CRAFT')
net, refine_net = load_craft(device=device)

print('🍞 Load VietOCR')
detector = load_detector(device=device)

print('🍞 Load LayoutLMv3')
layoutlm3 = load_layoutlm_v3().to(device)
layoutlm3_processor = load_layoutlm_v3_processor()

print('🍞 Load PhoBERT')
phobert = load_phobert_model().to(device)
phobert_tokenizer = load_phobert_tokenizer(device=device)

print('🍞 Start API')
app = FastAPI(debug=True, title='VietnameseSceneText')
flush()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def root():
    return { 'message': 'Hello, this is a Vietnamese scene text API.' }


@app.post('/api/process-image')
async def process_image(image: Annotated[UploadFile, File()]):
    try:
        global net, refine_net, detector, device
        print('🚩 process_image: save the received image to file')
        with open('temp.jpg', 'wb') as f:
            f.write(await image.read())
        print('🚩 process_image: load the image as numpy array')
        image = imgproc.loadImage('temp.jpg')
        print('🚩 process_image: detect text in image')
        image, boxes = detect_text_wrapper(
            net,
            refine_net=refine_net,
            image=image, 
            device=device,
            enables_rotate=True,
        )
        print('🚩 process_image: extract text from image')
        data = extract_texts(detector, image, boxes)
        words = [x['text'] for x in data]
        print('🚩 process_image: layoutlmv3 inference')
        layoutlmv3_labels, true_boxes = tag(
            image=image,
            words=words, 
            boxes=boxes, 
            model=layoutlm3, 
            processor=layoutlm3_processor,
            device=device,
        )
        print('🚩 process_image: phobert inference')
        phobert_labels = infer_phobert(
            words, 
            phobert_tokenizer,
            phobert,
            device=device,
        )
        print('🚩 process_image: add bounding boxes to image for visualization')
        # vis = visualize_image_with_boxes(image, boxes).astype(np.uint8)
        print('🚩 process_image: send the data to express server')
        return {
            'success': True,
            'data': [
                { **old, 'tag': tag, 'pho_tag': pho_tag } 
                for old, tag, pho_tag in 
                zip(data, layoutlmv3_labels, phobert_labels)
            ],
            # 'visualization': image_to_base64(vis),
        }
    except Exception as e:
        print('🌋 %s' % (str(e)))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e),
        )
