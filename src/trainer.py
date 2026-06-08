# Trainer - NAV OCR System

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
from transformers import LayoutLMv3ForTokenClassification
import torch.nn.functional as nnf
from config import MODEL_DIR, NUM_CLASSES


def loss_fn(pred, target):
    return nn.CrossEntropyLoss()(pred.view(-1, NUM_CLASSES), target.view(-1))


class ModelModule(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
        self.cls_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes)
        )

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )

        output = self.cls_layer(output.logits)

        prob = nnf.softmax(output, dim=1)
        top_p, top_class = prob.topk(1, dim=1)

        print("Sannsynlighetsscore:", prob)
        print("Top sannsynlighet / klasse:", top_p, top_class)

        loss = loss_fn(output, labels) if labels is not None else None

        return output, loss
