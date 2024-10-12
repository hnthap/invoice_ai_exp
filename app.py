import base64
from io import BytesIO

from PIL import Image
import numpy as np
import pandas as pd
from pandas.io import clipboard
import streamlit as st
import torch
from torch.backends import cudnn
from vietocr.tool.predictor import Predictor

from clovaai_craft.craft import CRAFT
from clovaai_craft import imgproc
from clovaai_craft.refine_net import RefineNet
from scene_text import (load_weights, detect_text, detect_text2, crop_boxes,
                        visualize_boxes)
from vietocr_api import load_vietocr_detector


def main():
    title = 'Insert Title Here'
    icon = '🖼️'

    st.set_page_config(title, icon, 'wide')

    device = get_device()    
    net, refine_net = load_craft(device=device)
    detector = load_detector(device=device)

    [left, right] = st.columns([3, 5], gap='medium')

    with right:
        # enables_sharpen = st.checkbox('Sharpen image before inferencing',
        #                               value=False)
        enables_rotate = st.checkbox(
            'Enable rotating for better results, but it is as twice as '
            'slower', value=False,
        )
    with left:
        image = upload_image()
        # if enables_sharpen:
        #     image = sharpen_image(image)

    image, boxes = detect_text_wrapper(
        net, refine_net, image, device, enables_rotate)
    toast_box_count(len(boxes))

    with left:
        visualize_image_with_boxes(image, boxes)
    
    with right:
        df = extract_texts(detector, image, boxes)
        st.dataframe(
            df,
            column_config={
                'image': st.column_config.ImageColumn('Image', width='medium'),
                'text': st.column_config.TextColumn('Text', width='medium'),
            },
            use_container_width=True,
            hide_index=False,
            column_order=('image', 'text'),
        )


def upload_image():
    supported_file_suffixes = ['jpg', 'png', 'jpeg']
    default_image_file = 'assets/sample-003.jpg'

    image_file = st.file_uploader('Upload an image', supported_file_suffixes)
    if image_file is None:
        image_file = default_image_file
    image = imgproc.loadImage(image_file)
    return image
                    

@st.cache_resource
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


@st.cache_resource
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


@st.cache_data
def extract_texts(_detector: Predictor, image: np.ndarray,
                  boxes: list[np.ndarray]) -> str:
    pieces = crop_boxes(np.int32(image), boxes)
    piece_objs = [Image.fromarray(np.uint8(piece), 'RGB') 
                    for piece in pieces]
    texts = _detector.predict_batch(piece_objs)
    df = pd.DataFrame({ 
        'ID': list(map(lambda v: '{}/{}'.format(v, len(pieces)),
                        range(1, len(pieces) + 1))),
        'image': list(map(pil_image_to_base36, piece_objs)),
        'text': texts,
    })
    df.set_index('ID', inplace=True)
    return df
    

@st.cache_data
def pil_image_to_base36(image: Image.Image) -> str:
    # https://stackoverflow.com/a/16066722
    output = BytesIO()
    image.save(output, format='JPEG')
    data = output.getvalue()
    encoded_data = base64.b64encode(data)
    if not isinstance(encoded_data, str):
        encoded_data = encoded_data.decode()
    data_url = 'data:image/jpg;base64,' + encoded_data
    return data_url


@st.cache_data
def detect_text_wrapper(_net, _refine_net, image, device, enables_rotate):
    if enables_rotate:
        new_image, boxes = detect_text2(_net, image, device=device,
                                        refine_net=_refine_net)
        if len(boxes) > 0: # If detected some texts
            image = new_image
    else:
        boxes = detect_text(_net, image, device=device, refine_net=_refine_net)
    return image, boxes


# @st.cache_data
# def sharpen_image_wrapper(image: np.ndarray) -> np.ndarray:
#     return sharpen_image(image)


def copy_to_clipboard(text: str):
    '''
    This function copies the provided text to the clipboard.
    It uses the 'clipboard' module from the pandas.io library to perform the
    copy operation.

    Args:
        text (str): The text to be copied to the clipboard.
    '''
    clipboard.copy(text)
    st.success('Copied text to clipboard.')


def get_device() -> str:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        st.warning('WARNING: No GPU detected. Using CPU for OCR. '
                   'It may take a long time.')
    return device


def visualize_image_with_boxes(image: np.ndarray, boxes: list[np.ndarray]):
    vis = visualize_boxes(image, boxes)
    st.image(vis, use_column_width=True)


def toast_box_count(box_count: int) -> None:
    if box_count == 0:
        toast = 'No text detected.'
    elif box_count == 1:
        toast = 'Detected 1 box.'
    else:
        toast = 'Detected {} boxes.'.format(box_count)
    if box_count == 0:
        icon = '❄️'
    elif box_count <= 30:
        icon = '📦'
    else:
        icon = '💣'
    st.toast(toast, icon=icon)


if __name__ == '__main__':
    main()
