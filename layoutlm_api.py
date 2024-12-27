import torch
from transformers import AutoProcessor, LayoutLMv3ForSequenceClassification


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
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        pretrained_path,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def load_layoutlm_v3_processor():
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    return processor


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]


def tag(image, width, height, words, boxes, processor, model):
    bboxes = []
    for b in boxes:
        bboxes.append([
            min(b[0][0], b[1][0], b[2][0], b[3][0]) * 1000.0 // width,
            min(b[0][1], b[1][1], b[2][1], b[3][1]) * 1000.0 // height,
            max(b[0][0], b[1][0], b[2][0], b[3][0]) * 1000.0 // width,
            max(b[0][1], b[1][1], b[2][1], b[3][1]) * 1000.0 // height,
        ])
    encoding = processor(image, words, boxes=boxes, truncation=True, return_tensors='pt')
    for k, v in encoding.items():
        print(k, v.shape, v.dtype)
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    print('Logits shape:', logits.shape)
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
    # true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]

    return true_predictions, true_boxes
