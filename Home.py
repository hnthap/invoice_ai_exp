from PIL import Image
from pandas.io import clipboard
import streamlit as st
import torch
from vietocr.tool.predictor import Predictor

from vietocr_api import load_vietocr_detector


def main():
    title = 'Vietnamese Text Extractor'
    icon = '🖼️'
    supported_file_suffixes = ['jpg', 'png', 'jpeg', 'gif', 'webm']
    default_image_file = 'assets/sample.png'

    st.set_page_config(title, icon, 'centered')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        st.warning('WARNING: No GPU detected. Using CPU for OCR. '
                   'It may take a long time.')
    detector = load_detector(device=device)

    image_file = st.file_uploader('Upload an image', supported_file_suffixes)
    if image_file is None:
        image_file = default_image_file

    result = extract_text(detector, image_file)
    
    with st.container(border=True):
        st.text(result)
    if st.button('Copy Result', help='Copy the result to clipboard'):
        copy_to_clipboard(result)
    st.image(image_file, width=500)


@st.cache_resource
def load_detector(*, device: str) -> Predictor:
    '''
    This function loads a pre-trained Vietnamese OCR detector for text extraction.
    It uses the provided device (either CPU or GPU) for inference.

    Parameters:
    device (str): The device to be used for inference. It can be either 'cuda:0' for GPU or 'cpu' for CPU.

    Returns:
    Predictor: An instance of the Vietnamese OCR detector, pre-trained and ready for text extraction.
    '''
    return load_vietocr_detector(device=device)


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


def copy_to_clipboard(text: str) -> None:
    '''
    This function copies the provided text to the clipboard.
    It uses the 'clipboard' module from the pandas.io library to perform the copy operation.
    Additionally, it uses the 'st' module from the Streamlit library to display a success message.

    Parameters:
    text (str): The text to be copied to the clipboard.

    Returns:
    None: This function does not return any value.
    '''
    clipboard.copy(text)
    st.success('Copied text to clipboard.')


if __name__ == '__main__':
    main()
