# Loader - NAV OCR System

from utils import *
import torch
from tqdm import tqdm
from PIL import Image


class dataSet:
    def __init__(self, json_path, processor=None) -> None:
        self.json_data = train_data_format(read_json(json_path))
        self.processor = processor

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, index) -> dict:
        imgs   = []
        words  = []
        labels = []
        bboxes = []

        data = self.json_data[index]

        imgs.append(Image.open(data['img_path']).convert('RGB'))

        # Hent tokens, labels og bboxes
        words  = data.get('tokens',  ["[UNK]"])
        labels = data.get('ner_tag', [0] * len(words))
        bboxes = data.get('bboxes',  [[0, 0, 0, 0]] * len(words))

        # Sikre at bbox og words har samme lengde
        if len(words) != len(bboxes):
            if len(words) > len(bboxes):
                diff = len(words) - len(bboxes)
                bboxes.extend([[0, 0, 0, 0]] * diff)
            else:
                bboxes = bboxes[:len(words)]

        # Tokenisering og koding
        encoding = self.processor(
            imgs,
            words=[words],
            boxes=[bboxes],
            word_labels=[labels],
            max_length=512,
            padding="max_length",
            truncation="longest_first",
            return_tensors='pt'
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox":           encoding["bbox"].squeeze(0),
            "pixel_values":   encoding["pixel_values"].squeeze(0),
            "labels":         encoding["labels"].squeeze(0)
        }
