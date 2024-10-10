from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def load_vietocr_detector(*, device) -> Predictor:
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = device
    detector = Predictor(config)
    return detector
