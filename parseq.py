'''
https://huggingface.co/spaces/baudm/PARSeq-OCR/blob/main/app.py.
'''

# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms as T
from PIL import Image


class ParseqApp:

    models = ['parseq', 'parseq_tiny', 'abinet', 'crnn', 'trba', 'vitstr']

    def __init__(self, *, device):
        self.device = device
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC).to(device),
            T.ToTensor(),
            T.Normalize(0.5, 0.5).to(device),
        ])


    def _get_model(self, name, *, verbose=True):
        if name in self._model_cache:
            return self._model_cache[name]
        model = torch.hub.load('baudm/parseq', name, pretrained=True,
                               trust_repo=True, verbose=verbose)
        model = model.eval().to(self.device)
        self._model_cache[name] = model
        return model


    @torch.inference_mode()
    def __call__(self, model_name: str, image: dict | Image.Image, *,
                 return_confidence=False, verbose=True):
        if isinstance(image, dict):
            image = image['composite']
        assert isinstance(image, Image.Image)
        model = self._get_model(model_name, verbose=verbose)
        image = self._preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        # Greedy decoding
        pred = model(image).softmax(-1)
        label, _ = model.tokenizer.decode(pred)
        # Format confidence values
        if return_confidence:
            raw_label, raw_confidence = model.tokenizer.decode(pred, raw=True)
            max_len = 25 if model_name == 'crnn' else len(label[0]) + 1
            conf = list(map('{:0.1f}'.format, raw_confidence[0][:max_len].tolist()))
            return label[0], [raw_label[0][:max_len], conf]
        return label[0], None
    