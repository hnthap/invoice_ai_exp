import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


id2label={
    0: 'SELLER',
    1: 'ADDRESS',
    2: 'TIMESTAMP',
    3: 'TOTAL_COST',
    4: 'TOTAL_TOTAL_COST',
}
label2id={
    'SELLER': 0,
    'ADDRESS': 1,
    'TIMESTAMP': 2,
    'TOTAL_COST': 3,
    'TOTAL_TOTAL_COST': 4,
}

labels = list(label2id.keys())

def convert_ner_tags_to_id(ner_tags):
  return [label2id[ner_tag] for ner_tag in ner_tags]


def load_layoutlm_v3(
    pretrained_path='/home/tekton/weights/kaggle/working/best_layout_LM3',
):
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        pretrained_path,
        id2label=id2label,
        label2id=label2id,
        # torch_dtype=torch.float16,
    )
    return model


def load_layoutlm_v3_processor():
    processor = LayoutLMv3Processor.from_pretrained(
        'microsoft/layoutlmv3-base',
        apply_ocr=False,
    )
    return processor


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def special_round(value, min_value, max_value):
    x = round(value)
    if x < min_value: return min_value
    if x > max_value: return max_value
    return int(x)


def tag(*, image, words, boxes, model, processor, device):
    width, height = image.shape[1], image.shape[0]
    print('🚩 layoutlm: normalize bounding boxes')
    print('width =', width)
    print('height =', height)
    bboxes = []
    for b in boxes:
        x1 = special_round(
            min(b[0][0], b[1][0], b[2][0], b[3][0]) * 1000.0 // width,
            0,
            width,
        )
        y1 = special_round(
            min(b[0][1], b[1][1], b[2][1], b[3][1]) * 1000.0 // height,
            0,
            height,
        )
        x2 = special_round(
            max(b[0][0], b[1][0], b[2][0], b[3][0]) * 1000.0 // width,
            0,
            width,
        )
        y2 = special_round(
            max(b[0][1], b[1][1], b[2][1], b[3][1]) * 1000.0 // height,
            0,
            height,
        )
        bboxes.append([x1, y1, x2, y2])
    print('🚩 layoutlm: encode')
    encoding = processor(
        image,
        words,
        boxes=torch.tensor(bboxes),
        return_token_type_ids=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    encoding.pop('offset_mapping')
    for k, v in encoding.items():
        encoding[k] = v.to(device)
        print('🔰🔰', k, v.shape, v.dtype, v.max(), v.min())
    print('🚩 layoutlm: infer')
    with torch.no_grad():
        outputs = model(**encoding)
    for k, v in outputs.items():
        print('🔰🔰', k, v.shape, v.dtype, v.max(), v.min())
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    print('🔰 predictions =', predictions)
    true_predictions = list(map(id2label.__getitem__, predictions))
    token_boxes = encoding.bbox.squeeze().tolist()
    print('🔰 len(token_boxes) =', len(token_boxes))
    true_boxes = list(map(
        lambda b: unnormalize_box(b, width, height), 
        token_boxes,
    ))
    print('🚩 layoutlm: all done')
    return true_predictions, token_boxes
