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
from scene_text import load_weights, detect_text, crop_boxes, visualize_boxes
from vietocr_api import load_vietocr_detector


def main():
    title = 'Insert Title Here'
    icon = '🖼️'

    st.set_page_config(title, icon, 'wide')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        st.warning('WARNING: No GPU detected. Using CPU for OCR. '
                   'It may take a long time.')
        
    net, refine_net = load_craft(device=device)
    detector = load_detector(device=device)

    [left, right] = st.columns([1, 1.2], gap='medium')

    with right:
        image = upload_image()
        boxes = detect_text(net, image, device=device, refine_net=refine_net)

    with st.empty():
        st.toast('Detected {} boxes.'.format(len(boxes)), icon='📦')

    with left:
        st.image(visualize_boxes(image, boxes), use_column_width=True)
    
    with right:
        pieces = crop_boxes(np.int32(image), boxes)
        piece_objs = [Image.fromarray(np.uint8(piece), 'RGB') for piece in pieces]
        texts = detector.predict_batch(piece_objs)
        df = pd.DataFrame({ 
            'id': range(1, len(pieces) + 1),
            'image': list(map(pil_image_to_base36, piece_objs)),
            'text': texts,
        })
        df.set_index('id')
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
    supported_file_suffixes = ['jpg', 'png', 'jpeg', 'gif', 'webm']
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
    
    net = CRAFT().to(device) # net = nn.DataParallel(net)
    cudnn.benchmark = False
    net = load_weights(net, 'weights/craft_mlt_25k.pth', device=device)

    refine_net = RefineNet().to(device)
    refine_net = load_weights(
        refine_net, 'weights/craft_refiner_CTW1500.pth', device=device)
    
    return net, refine_net


@st.cache_data
def extract_text(_detector: Predictor, image_file) -> str:
    '''
    This function extracts text from an input image using a pre-trained
    Vietnamese OCR detector.

    Args:
        _detector (Predictor): An instance of the Vietnamese OCR detector. 
            It should be pre-trained and loaded.
        image (Image.Image): The input image from which text needs to be 
            extracted.

    Returns:
        str: The extracted text from the input image.
    '''
    with Image.open(image_file) as image:
       return _detector.predict(image)
    

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


if __name__ == '__main__':
    main()
