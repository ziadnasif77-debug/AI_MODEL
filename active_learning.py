# Active Learning - NAV OCR System

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch
import json
import shutil
import time
import base64
import numpy as np
from PIL import Image
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor,
)
from trainer import ModelModule
from engine import DEVICE
from config import (
    MODEL_DIR, OUTPUT_MODEL_DIR, OUTPUT_IMAGES_DIR,
    OUTPUT_JSON_DIR, PENDING_DIR,
    NUM_CLASSES, NAV_LABELS, NAV_LABEL_MAP,
    fix_norwegian
)

CONFIDENCE_THRESHOLD = 90.0

REVIEW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'active_learning')
APPROVED_DIR = os.path.join(REVIEW_DIR, 'approved')
LOW_CONFIDENCE_DIR = os.path.join(REVIEW_DIR, 'needs_review')
HISTORY_FILE = os.path.join(REVIEW_DIR, 'history.json')


def setup_dirs():
    for d in [REVIEW_DIR, APPROVED_DIR, LOW_CONFIDENCE_DIR]:
        os.makedirs(d, exist_ok=True)


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = image_path.split('.')[-1].lower()
    return f"data:image/{ext};base64,{data}"


def results_to_label_studio(image_path, results, img_width, img_height):
    from uuid import uuid4

    output_json = {}
    output_json['data'] = {"ocr": image_to_base64(image_path)}

    annotation_result = []

    for r in results:
        bbox = r['bbox']
        x_pct = bbox[0] / 10.0
        y_pct = bbox[1] / 10.0
        w_pct = (bbox[2] - bbox[0]) / 10.0
        h_pct = (bbox[3] - bbox[1]) / 10.0

        region_id = str(uuid4())[:10]

        bbox_val = {
            'x': x_pct, 'y': y_pct,
            'width': w_pct, 'height': h_pct,
            'rotation': 0
        }

        bbox_result = {
            'id': region_id,
            'from_name': 'bbox',
            'to_name': 'image',
            'type': 'rectangle',
            'value': bbox_val,
            'original_width': img_width,
            'original_height': img_height,
        }

        transcription_result = {
            'id': region_id,
            'from_name': 'transcription',
            'to_name': 'image',
            'type': 'textarea',
            'value': dict(text=[r['text']], **bbox_val),
            'score': r['confidence'] / 100.0
        }

        label_result = {
            'id': region_id,
            'from_name': 'label',
            'to_name': 'image',
            'type': 'rectanglelabels',
            'value': dict(rectanglelabels=[r['label']], **bbox_val),
            'original_width': img_width,
            'original_height': img_height,
        }

        annotation_result.extend([bbox_result, transcription_result, label_result])

    output_json['predictions'] = [{"result": annotation_result, "score": 0.5}]
    return output_json


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"rounds": [], "processed_files": []}


def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def run_active_learning(image_folder=None, threshold=None):
    from src.Inference import infer_single_image

    if threshold is not None:
        conf_threshold = threshold
    else:
        conf_threshold = CONFIDENCE_THRESHOLD

    if image_folder is None:
        image_folder = OUTPUT_IMAGES_DIR

    setup_dirs()
    history = load_history()

    already_processed = set(history.get("processed_files", []))

    images = [f for f in os.listdir(image_folder)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))
              and f not in already_processed]

    if not images:
        print("Ingen nye bilder a behandle!")
        if already_processed:
            print(f"  ({len(already_processed)} bilder allerede behandlet i tidligere runder)")
        return

    featur_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokeniser = LayoutLMv3TokenizerFast.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
    processor = LayoutLMv3Processor(tokenizer=tokeniser, feature_extractor=featur_extractor)

    trained_model_path = os.path.join(OUTPUT_MODEL_DIR, 'model_best.bin')
    if not os.path.exists(trained_model_path):
        print(f"FEIL: Ingen trent modell funnet: {trained_model_path}")
        print("Kjor main.py forst for a trene modellen.")
        return

    model = ModelModule(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(trained_model_path, map_location=DEVICE))
    model.eval()

    total = len(images)
    approved = []
    needs_review = []
    errors = []

    round_num = len(history["rounds"]) + 1

    print("=" * 60)
    print(f"Active Learning — Runde {round_num}")
    print(f"Terskel: {conf_threshold}% | Bilder: {total} | Enhet: {DEVICE}")
    print("=" * 60)

    start_total = time.time()

    for i, image_file in enumerate(images):
        image_path = os.path.join(image_folder, image_file)
        start = time.time()

        try:
            results, test_dict, w, h = infer_single_image(image_path, model, processor)
            elapsed = time.time() - start

            nav_fields = [r for r in results if r['label'] != 'O']

            if not nav_fields:
                avg_conf = 0.0
            else:
                avg_conf = sum(r['confidence'] for r in nav_fields) / len(nav_fields)

            img = Image.open(image_path)
            img_width, img_height = img.size

            if avg_conf >= conf_threshold:
                approved.append({
                    "file": image_file,
                    "fields": nav_fields,
                    "avg_confidence": round(avg_conf, 1)
                })

                result_path = os.path.join(APPROVED_DIR, f'{image_file}.json')
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "file": image_file,
                        "avg_confidence": round(avg_conf, 1),
                        "fields": nav_fields
                    }, f, indent=2, ensure_ascii=False)

                status = "OK"

            else:
                needs_review.append({
                    "file": image_file,
                    "fields": nav_fields,
                    "avg_confidence": round(avg_conf, 1)
                })

                ls_json = results_to_label_studio(image_path, results, img_width, img_height)
                review_path = os.path.join(LOW_CONFIDENCE_DIR, f'{image_file[:-4]}_review.json')
                with open(review_path, 'w', encoding='utf-8') as f:
                    json.dump(ls_json, f, indent=2, ensure_ascii=False)

                status = "REVIEW"

            field_summary = ", ".join(f"{r['label']}:{r['confidence']}%" for r in nav_fields[:4])
            print(f"  [{i+1}/{total}] {image_file} — {status} ({avg_conf:.0f}%) — {elapsed:.1f}s")
            if field_summary:
                print(f"           {field_summary}")

        except Exception as e:
            errors.append({"file": image_file, "error": str(e)})
            print(f"  [{i+1}/{total}] {image_file} — FEIL: {e}")

    total_time = time.time() - start_total

    round_info = {
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": conf_threshold,
        "total_images": total,
        "approved": len(approved),
        "needs_review": len(needs_review),
        "errors": len(errors),
        "time_seconds": round(total_time, 1)
    }

    history["rounds"].append(round_info)
    history["processed_files"].extend([img for img in images])
    save_history(history)

    print("\n" + "=" * 60)
    print(f"Runde {round_num} ferdig! ({total_time:.1f}s)")
    print(f"  Godkjent (>{conf_threshold}%):  {len(approved)}/{total}")
    print(f"  Trenger review (<{conf_threshold}%): {len(needs_review)}/{total}")
    if errors:
        print(f"  Feil:                    {len(errors)}/{total}")

    if len(approved) + len(needs_review) > 0:
        approval_rate = len(approved) / (len(approved) + len(needs_review)) * 100
        print(f"  Godkjenningsrate:        {approval_rate:.0f}%")

    if needs_review:
        print(f"\n  Filer som trenger review ligger i:")
        print(f"    {LOW_CONFIDENCE_DIR}")
        print(f"\n  Neste steg:")
        print(f"    1. Importer JSON-filene i Label Studio")
        print(f"    2. Korriger merkingen")
        print(f"    3. Eksporter til training_data/pending/")
        print(f"    4. Kjor Label_studio_to_layoutLMV3.py")
        print(f"    5. Kjor main.py for a retrenne")
        print(f"    6. Kjor active_learning.py igjen!")
    else:
        print(f"\n  Alle bilder godkjent! Modellen er bra nok.")

    print("=" * 60)

    if history["rounds"] and len(history["rounds"]) > 1:
        print("\n  Historikk:")
        print(f"  {'Runde':<8}{'Godkjent':<12}{'Review':<12}{'Rate':<10}")
        print(f"  {'-'*42}")
        for r in history["rounds"]:
            total_r = r["approved"] + r["needs_review"]
            rate = r["approved"] / total_r * 100 if total_r > 0 else 0
            print(f"  {r['round']:<8}{r['approved']:<12}{r['needs_review']:<12}{rate:.0f}%")

    return {
        "approved": approved,
        "needs_review": needs_review,
        "errors": errors,
        "round_info": round_info
    }


def send_to_pending():
    review_files = [f for f in os.listdir(LOW_CONFIDENCE_DIR) if f.endswith('.json')]
    if not review_files:
        print("Ingen filer i needs_review!")
        return

    os.makedirs(PENDING_DIR, exist_ok=True)

    for f in review_files:
        src = os.path.join(LOW_CONFIDENCE_DIR, f)
        dst = os.path.join(PENDING_DIR, f)
        shutil.copy2(src, dst)

    print(f"Kopiert {len(review_files)} filer til {PENDING_DIR}")
    print("Neste: Importer i Label Studio, korriger, og eksporter tilbake.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NAV OCR Active Learning")
    parser.add_argument('--folder', type=str, help="Mappe med bilder")
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Konfidensterskel i prosent (standard: {CONFIDENCE_THRESHOLD})")
    parser.add_argument('--send-to-pending', action='store_true',
                        help="Kopier review-filer til training_data/pending/")
    parser.add_argument('--reset', action='store_true',
                        help="Nullstill historikk og start pa nytt")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            print("Historikk nullstilt!")
        else:
            print("Ingen historikk a nullstille.")

    elif args.send_to_pending:
        send_to_pending()

    else:
        run_active_learning(
            image_folder=args.folder,
            threshold=args.threshold
        )
