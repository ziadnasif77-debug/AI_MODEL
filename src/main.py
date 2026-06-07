# Main - NAV OCR System

import torch
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from trainer import *
from loader import *
from torch.optim import AdamW
import numpy as np
from engine import *

# ── NAV Labels ──
# 0: O (other)
# 1: NAVN
# 2: FODSELSNUMMER
# 3: DATO
# 4: ADRESSE
# 5: SIGNATUR
NUM_CLASSES = 6

# ── Paths ──
MODEL_DIR     = r"E:\AI_MODOL\models\layoutlmv3"
TRAINING_JSON = r"E:\AI_MODOL\training_data\Training_layoutLMV3.json"
OUTPUT_MODEL  = r"E:\AI_MODOL\models\nav_layoutlmv3"

import os
os.makedirs(OUTPUT_MODEL, exist_ok=True)

featur_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokeniser = LayoutLMv3TokenizerFast.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
processor = LayoutLMv3Processor(tokenizer=tokeniser, feature_extractor=featur_extractor)


if __name__ == "__main__":
    print("="*60)
    print("NAV OCR - LayoutLMv3 Treningsprogram")
    print("="*60)

    ds = dataSet(TRAINING_JSON, processor)
    print(f"Treningsdata: {len(ds)} bilder")

    dataload = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)

    # Opprett modell
    model = ModelModule(NUM_CLASSES)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_loss = np.inf

    loss_list = []
    EPOCHS = 30

    print(f"\nStarter trening: {EPOCHS} epoker")
    print("="*60)

    for epoch in range(EPOCHS):
        # Trening
        train_loss = train_fn(dataload, model, optimizer)

        # Lagre beste modell
        if train_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL, 'model_best.bin'))
            best_loss = train_loss
            print(f"  ✅ Ny beste modell lagret! Loss: {train_loss:.4f}")

        # Lagre sjekkpunkter hvert 10. epoke
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL, f'model_epoke_{epoch}.bin'))

        print(f"  Epoke {epoch+1:2d}/{EPOCHS}  |  Loss: {train_loss:.4f}")
        loss_list.append(train_loss)

        # Evaluering
        eval_loss = eval_fn(dataload, model)
        print(f"  Evalueringstap: {eval_loss:.4f}")

    np.array(loss_list).dump(open(os.path.join(OUTPUT_MODEL, 'loss_list.npy'), 'wb'))

    print("="*60)
    print("Trening fullfort!")
    print(f"Beste loss:  {best_loss:.4f}")
    print(f"Modell lagret i: {OUTPUT_MODEL}")
    print("="*60)