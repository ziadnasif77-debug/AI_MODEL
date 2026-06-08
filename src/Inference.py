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
import json
import time
import torch.nn.functional as nnf
from trainer import ModelModule
from utils import dataSetFormat, plot_img, fix_norwegian
from engine import DEVICE
from config import (
    MODEL_DIR, OUTPUT_MODEL_DIR, OUTPUT_IMAGES_DIR,
    NUM_CLASSES, NAV_LABELS
)


def validate_fodselsnummer(text):
    digits = ''.join(c for c in text if c.isdigit())
    if len(digits) == 11:
        return digits
    return text


def validate_dato(text):
    digits = ''.join(c for c in text if c.isdigit() or c in './-')
    return digits if digits else text


def post_process(label_name, text):
    if label_name == "FODSELSNUMMER":
        return validate_fodselsnummer(text)
    if label_name == "DATO":
        return validate_dato(text)
    return text


def infer_single_image(image_path, model, processor):
    image = Image.open(image_path)
    test_dict, width_scale, height_scale = dataSetFormat(image)

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

    inputs_ids     = encoding['input_ids'].squeeze(0).to(DEVICE)
    attention_mask = encoding['attention_mask'].squeeze(0).to(DEVICE)
    bbox           = encoding['bbox'].squeeze(0).to(DEVICE)
    pixel_values   = encoding['pixel_values'].squeeze(0).to(DEVICE)

    with torch.no_grad():
        op, _ = model(
            input_ids=inputs_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            bbox=bbox.unsqueeze(0),
            pixel_values=pixel_values.unsqueeze(0)
        )
        predictions = op.argmax(-1).squeeze().tolist()

        prob = nnf.softmax(op, dim=1)
        txt = prob.squeeze().cpu().numpy()
        txt = txt / np.sum(txt, axis=1).reshape(-1, 1)
        output_prob = np.max(txt, axis=1)

    pred = torch.tensor(predictions)
    offset_mapping = encoding['offset_mapping']
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

    true_predictions = [pred[i].item() for i in range(len(pred)) if not is_subword[i]]
    true_prob_list = [output_prob[i] for i in range(len(output_prob)) if not is_subword[i]]
    true_boxes = [bbox.cpu()[i].tolist() for i in range(len(bbox)) if not is_subword[i]]
    true_tokens = []
    token_idx = 0
    for i in range(len(pred)):
        if not is_subword[i]:
            if token_idx < len(test_dict['tokens']):
                true_tokens.append(test_dict['tokens'][token_idx])
                token_idx += 1
            else:
                true_tokens.append("")

    results = []
    for i in range(len(true_predictions)):
        label_id = true_predictions[i]
        if label_id == 0:
            continue
        label_name = NAV_LABELS.get(label_id, "O")
        token_text = true_tokens[i] if i < len(true_tokens) else ""
        token_text = post_process(label_name, token_text)
        results.append({
            "label": label_name,
            "text": token_text,
            "confidence": round(float(true_prob_list[i]) * 100, 1),
            "bbox": true_boxes[i]
        })

    return results, test_dict, width_scale, height_scale


def run_batch_inference(image_folder, model, processor):
    images = [f for f in os.listdir(image_folder)
              if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

    if not images:
        print("Ingen bilder funnet!")
        return

    total = len(images)
    all_results = {}

    print(f"Starter inferens pa {total} bilder (enhet: {DEVICE})")
    print("=" * 60)

    start_total = time.time()

    for i, image_file in enumerate(images):
        image_path = os.path.join(image_folder, image_file)
        start = time.time()

        try:
            results, _, _, _ = infer_single_image(image_path, model, processor)
            elapsed = time.time() - start

            all_results[image_file] = results

            nav_fields = [r for r in results if r['label'] != 'O']
            print(f"  [{i+1}/{total}] {image_file} — {len(nav_fields)} felt — {elapsed:.1f}s")
            for field in nav_fields:
                print(f"    {field['label']}: {field['text']} ({field['confidence']}%)")

        except Exception as e:
            print(f"  [{i+1}/{total}] {image_file} — FEIL: {e}")
            all_results[image_file] = {"error": str(e)}

    total_time = time.time() - start_total
    print("=" * 60)
    print(f"Ferdig! {total} bilder — {total_time:.1f}s totalt — {total_time/total:.1f}s/bilde")

    results_path = os.path.join(OUTPUT_MODEL_DIR, 'inference_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"Resultater lagret: {results_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NAV OCR Inferens")
    parser.add_argument('--image', type=str, help="Sti til enkeltbilde")
    parser.add_argument('--folder', type=str, help="Mappe med bilder for batch-inferens")
    parser.add_argument('--show', action='store_true', help="Vis bilde med resultater")
    args = parser.parse_args()

    # Last inn modell
    featur_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokeniser = LayoutLMv3TokenizerFast.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
    processor = LayoutLMv3Processor(tokenizer=tokeniser, feature_extractor=featur_extractor)

    trained_model_path = os.path.join(OUTPUT_MODEL_DIR, 'model_best.bin')
    model = ModelModule(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(trained_model_path, map_location=DEVICE))
    model.eval()

    print("=" * 60)
    print(f"NAV OCR - Inferens (enhet: {DEVICE})")
    print("=" * 60)

    if args.folder:
        run_batch_inference(args.folder, model, processor)

    elif args.image:
        results, test_dict, w, h = infer_single_image(args.image, model, processor)

        print("\nResultater:")
        print("-" * 40)
        for r in results:
            print(f"  {r['label']}: {r['text']} ({r['confidence']}%)")

        if args.show:
            from utils import plot_img
            boxes = torch.tensor([r['bbox'] for r in results])
            labels = [list(NAV_LABELS.values()).index(r['label']) for r in results]
            probs = [r['confidence'] for r in results]
            plot_img(test_dict['img_path'], boxes, labels, probs, w, h)

    else:
        default_folder = OUTPUT_IMAGES_DIR
        print(f"Ingen argumenter — kjorer batch pa: {default_folder}")
        run_batch_inference(default_folder, model, processor)
