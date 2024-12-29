import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


label_list = ['SELLER', 'ADDRESS', 'TIMESTAMP', 'TOTAL_COST', 'TOTAL_TOTAL_COST']

PHOBERT_PRETRAINED_PATH = 'weights/best_phobert_20241229'


def load_phobert_tokenizer(pretrained_path=PHOBERT_PRETRAINED_PATH, *, device):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, device_map=device)
    return tokenizer


def preprocess_phobert(texts, tokenizer):
    return tokenizer(
        texts, 
        padding='max_length', 
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )


def load_phobert_model(pretrained_path=PHOBERT_PRETRAINED_PATH):
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        num_labels=len(label_list),
        device_map='auto',
    )


def infer_phobert(texts, tokenizer, model, *, device):
    tokens = preprocess_phobert(texts, tokenizer).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().detach()
    return [label_list[int(p)] for p in predictions]
