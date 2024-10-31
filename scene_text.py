import os
import shutil

from PIL import Image
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from clovaai_craft.craft import CRAFT
from clovaai_craft import craft_utils
from clovaai_craft import imgproc
from clovaai_craft.refine_net import RefineNet
from image_utils import rotate_image, get_vector_angle
from vietocr_api import load_vietocr_detector


def main():
    input_image_file = 'assets/sample-003.jpg'

    trained_model = 'weights/craft_mlt_25k.pth'
    refiner_model = 'weights/craft_refiner_CTW1500.pth'
    output_dir = 'out'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert os.path.exists(trained_model)
    assert os.path.exists(refiner_model)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    net = CRAFT().to(device) # net = nn.DataParallel(net)
    cudnn.benchmark = False
    net = load_weights(net, trained_model, device=device)

    refine_net = RefineNet().to(device)
    refine_net = load_weights(refine_net, refiner_model, device=device)

    image = imgproc.loadImage(input_image_file)
    boxes = detect_text(net, image, device=device, refine_net=refine_net)
    
    pieces = crop_boxes(np.int32(image), boxes)
    piece_objs = [Image.fromarray(np.uint8(arr), 'RGB')
                  for arr in pieces]
    for i, obj in enumerate(piece_objs):
        save_image(obj, os.path.join(output_dir, f'{i}.jpg'))

    vietocr_detector = load_vietocr_detector(device=device)
    texts = vietocr_detector.predict_batch(piece_objs)

    with open(os.path.join(output_dir, 'texts.txt'), 'w') as f:
        for i, text in enumerate(texts):
            f.write(f'{i}\t{text}\n')

    for obj in piece_objs:
        obj.close()


def load_weights(model, filename, *, device):
    '''
    This function loads pre-trained weights into a given model.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        filename (str):
            The path to the file containing the pre-trained weights.
        device (torch.device): 
            The device (CPU or GPU) to load the weights onto.

    Returns:
        torch.nn.Module: The model with the loaded weights.
    '''
    weights = torch.load(filename, map_location=device, weights_only=True)
    weights = { key.replace('module.', ''): value 
                for key, value in weights.items() }
    model.load_state_dict(weights)
    model.eval()
    return model


def save_image(image: np.ndarray | Image.Image, filename):
    '''
    This function saves an image to a file.

    Args:
        image (np.ndarray | Image.Image):
            The image to be saved. It can be either a NumPy array or an 
            instance of the PIL.Image class.
        filename (str):
            The path and name of the file where the image will be saved.

    Returns:
        None:
    '''
    # if isinstance(image, str):
    #     image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image), 'RGB')
    assert isinstance(image, Image.Image)
    image.save(filename)


def detect_text(model, image: np.ndarray, *, device, refine_net = None,
                canvas_size = 1280, mag_ratio = 1.5, text_threshold = 0.7,
                low_text = 0.4, link_threshold = 0.4, poly = False
                ) -> list[np.ndarray]:
    '''
    This function performs text detection using a pre-trained CRAFT model and 
    an optional refiner model.

    Args:
        model: The pre-trained CRAFT model for text detection.
        image: The image as a 3D NumPy array.
        device: The device to run the model on (CPU or GPU).
        refine_net: (Optional) The refiner model for improving text detection 
            accuracy.
        canvas_size: The size of the canvas for resizing the image. 
            Default is 1280.
        mag_ratio: The magnification ratio for resizing the image. 
            Default is 1.5.
        text_threshold: The threshold for text detection. Default is 0.7.
        low_text: The threshold for low-confidence text detection. 
            Default is 0.4.
        link_threshold: The threshold for linking text detection components. 
            Default is 0.4.
        poly: A boolean flag indicating whether to return polygons instead of
            bounding boxes. Default is False.

    Returns:
        boxes: A list of bounding boxes or polygons around detected text in the
            image.
    '''
    t = imgproc.resize_aspect_ratio(image, canvas_size, 
                                    interpolation=cv2.INTER_LINEAR,
                                    mag_ratio=mag_ratio)
    img_resized, target_ratio, _ = t
    ratio_w = 1 / target_ratio
    ratio_h = ratio_w
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0), requires_grad=False).to(device)
    with torch.no_grad():
        y, feature = model(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    return boxes


def detect_text2(model, image: np.ndarray, *, device, refine_net = None,
                 canvas_size = 1280, mag_ratio = 1.5, text_threshold = 0.7,
                 low_text = 0.4, link_threshold = 0.4, poly = False,
                 rotate_if_less_than: int | None = None
                 ) -> tuple[np.ndarray, list[np.ndarray]]:
    boxes = detect_text(model, image, device=device, refine_net=refine_net,
                        canvas_size=canvas_size, mag_ratio=mag_ratio,
                        text_threshold=text_threshold, low_text=low_text,
                        link_threshold=link_threshold, poly=poly)
    if len(boxes) != 0 and (rotate_if_less_than is None or 
                            len(boxes) < rotate_if_less_than):
        vector = np.average(boxes[:, 1] - boxes[:, 0], axis=0)
        angle = get_vector_angle(vector)
        if abs(angle) > 0.5:
            image = rotate_image(image, angle)
            boxes = detect_text(
                model, image, device=device, refine_net=refine_net,
                canvas_size=canvas_size, mag_ratio=mag_ratio,
                text_threshold=text_threshold, low_text=low_text,
                link_threshold=link_threshold, poly=poly,
            )
    return image, boxes


def crop_boxes(image: np.ndarray, boxes):
    crops = []
    for box in boxes:
        box = np.int32(box)
        x_values = box[:, 0]
        y_values = box[:, 1]
        x1, x2 = min(x_values), max(x_values)
        y1, y2 = min(y_values), max(y_values)
        assert x1 + 1 < x2, '{}, {}'.format(x1, x2)
        assert y1 + 1 < y2, '{}, {}'.format(y1, y2)
        crops.append(image[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1].copy())
    return crops


def visualize_boxes(image: np.ndarray, boxes, *, color=(0, 255, 0)):
    t = np.int32(image.copy())
    for box in boxes:
        points = [np.int32(point) for point in box]
        points.append(points[0])
        for i in range(0, len(points)):
            cv2.line(t, points[i - 1], points[i], color,
                     max(2, min(image.shape[:2]) // 200))
    return t


if __name__ == "__main__":
    main()
