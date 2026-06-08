# Inference - NAV OCR System

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor,
)
import numpy as np
import torch.nn.functional as nnf
from engine import *
from trainer import *
from loader import *
from utils import *
from config import MODEL_DIR, OUTPUT_MODEL_DIR, OUTPUT_IMAGES_DIR, NUM_CLASSES, NAV_LABELS

# ── Stier ──
TRAINED_MODEL = os.path.join(OUTPUT_MODEL_DIR, 'model_best.bin')
TEST_IMAGE    = os.path.join(OUTPUT_IMAGES_DIR, 'doc07_kontroll_rapport.png')

# ── Prosessor ──
featur_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokeniser        = LayoutLMv3TokenizerFast.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
processor        = LayoutLMv3Processor(tokenizer=tokeniser, feature_extractor=featur_extractor)

# ── Last inn bilde ──
image = Image.open(TEST_IMAGE)
image.show()
test_dict, width_scale, height_scale = dataSetFormat(image)

print("Bounding boxes:", test_dict['bboxes'])

# ── Opprett og last inn modell ──
model = ModelModule(NUM_CLASSES)
model.load_state_dict(torch.load(TRAINED_MODEL))
model.eval()

# ── Koding ──
encoding = processor(
    test_dict['img_path'].convert('RGB'),
    test_dict['tokens'],
    boxes=test_dict['bboxes'],
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors='pt',
    return_offsets_mapping=True
)

print("Encoding bbox:", encoding['bbox'])

inputs_ids    = torch.tensor(encoding['input_ids'],      dtype=torch.int64).flatten()
attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.int64).flatten()
bbox           = torch.tensor(encoding['bbox'],           dtype=torch.int64).flatten(end_dim=1)
pixel_values   = torch.tensor(encoding['pixel_values'],   dtype=torch.float32).flatten(end_dim=1)

print("bbox:", bbox)

# ── Inferens ──
with torch.no_grad():
    op, _ = model(
        input_ids=inputs_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        bbox=bbox.unsqueeze(0),
        pixel_values=pixel_values.unsqueeze(0)
    )
    predictions = op.argmax(-1).squeeze().tolist()

    prob      = nnf.softmax(op, dim=1)
    txt       = prob.squeeze().numpy() / np.sum(prob.squeeze().numpy(), axis=1).reshape(-1, 1)
    output_prob = np.max(txt, axis=1)

pred           = torch.tensor(predictions)
offset_mapping = encoding['offset_mapping']
is_subword     = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

true_predictions = torch.tensor(np.array([pred.item() for idx, pred in enumerate(pred) if not is_subword[idx]]))
true_prob        = torch.tensor(np.array([output_prob.item() for idx, output_prob in enumerate(output_prob) if not is_subword[idx]]))
true_boxes       = torch.tensor([box.tolist() for idx, box in enumerate(bbox) if not is_subword[idx]])

concat_torch = torch.column_stack((true_boxes, true_predictions, true_prob))

# ── Filtrer etter NAV-felt (1-5) ──
navn_class   = concat_torch[torch.where((concat_torch[:, 4] == 1) & (concat_torch[:, 3] == 0) & (concat_torch[:, 2] == 0))]
fnr_class    = concat_torch[torch.where((concat_torch[:, 4] == 2) & (concat_torch[:, 3] == 0) & (concat_torch[:, 2] == 0))]
dato_class   = concat_torch[torch.where((concat_torch[:, 4] == 3) & (concat_torch[:, 3] == 0) & (concat_torch[:, 2] == 0))]
adr_class    = concat_torch[torch.where((concat_torch[:, 4] == 4) & (concat_torch[:, 3] == 0) & (concat_torch[:, 2] == 0))]
sign_class   = concat_torch[torch.where((concat_torch[:, 4] == 5) & (concat_torch[:, 3] == 0) & (concat_torch[:, 2] == 0))]

finl     = torch.row_stack((navn_class, fnr_class, dato_class, adr_class, sign_class))
unique_  = torch.unique(finl, dim=0)

# ── Vis resultat ──
print("\n" + "="*50)
print("NAV OCR - Resultater:")
print("="*50)
for row in unique_:
    label_id   = int(row[4].item())
    label_name = NAV_LABELS.get(label_id, "O")
    confidence = round(row[5].item() * 100, 1)
    print(f"  {label_name}: {confidence}% sikkerhet")

plot_img(
    test_dict['img_path'],
    unique_[:, :4],
    unique_[:, 4].tolist(),
    unique_[:, 5].tolist(),
    width_scale,
    height_scale
)

print(unique_)
