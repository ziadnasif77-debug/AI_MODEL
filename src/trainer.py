# Trainer - NAV OCR System

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
from transformers import LayoutLMv3Model
from config import MODEL_DIR


class ModelModule(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model = LayoutLMv3Model.from_pretrained(MODEL_DIR)
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )

        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.n_classes), labels.view(-1))

        return logits, loss
