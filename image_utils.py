import cv2
import numpy as np
from scipy import ndimage


_SHARPEN_FILTER = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]], dtype=np.float16)


def rotate_image(image: np.ndarray, angle_deg: float,
                 cval: float | int | None = None):
    '''
    Rotates an image by a given angle in degrees.

    Args:
        image (np.ndarray): The input image.
        angle_deg (float): The angle in degrees to rotate the image by.
        cval (float | int | None): The constant value to fill the empty parts
            created after the rotation.
    
    Returns:
        np.ndarray: The rotated image.
    '''
    if cval is None:
        cval = np.average(image).astype(np.int32)
    rotated_image = ndimage.rotate(image, angle=angle_deg, cval=cval)
    rotated_image = np.clip(rotated_image, 0, 255)
    return rotated_image


def get_vector_angle(vector: np.ndarray):
    '''
    Calculates the angle in degrees between the x-axis and the given vector.
    '''
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def sharpen_image(image: np.ndarray) -> np.ndarray:
    '''
    Applies a sharpening filter to the input image using the given filter
    kernel.
    '''
    sharpened_image = cv2.filter2D(image, -1, _SHARPEN_FILTER)
    return sharpened_image
