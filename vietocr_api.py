from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def load_vietocr_detector(*, device: str) -> Predictor:
    '''
    Load and initialize a VietOCR detector with the specified device.

    Args:
        device (str): The device to run the detector on.

    Returns:
        Predictor: An instance of the VietOCR detector with the given 
            configuration.
    '''
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['device'] = device
    detector = Predictor(config)
    return detector
