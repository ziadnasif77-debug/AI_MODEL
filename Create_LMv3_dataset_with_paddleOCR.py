import os
import base64
from multiprocessing import Pool, cpu_count
from paddleocr import PaddleOCR
from PIL import Image
import json
from uuid import uuid4
import numpy as np
import time
from config import (
    OUTPUT_IMAGES_DIR, OUTPUT_JSON_DIR,
    OCR_LANG, OCR_USE_GPU, OCR_GPU_MEM, NUM_WORKERS,
    fix_norwegian
)

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR,   exist_ok=True)


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = image_path.split('.')[-1].lower()
    return f"data:image/{ext};base64,{data}"


def process_single_image(args):
    image_file, images_folder_path = args

    ocr = PaddleOCR(
        use_angle_cls=False,
        lang=OCR_LANG,
        use_gpu=OCR_USE_GPU,
        gpu_mem=OCR_GPU_MEM,
        show_log=False
    )

    output_json = {}
    annotation_result = []

    img_path = os.path.join(images_folder_path, image_file)

    output_json['data'] = {"ocr": image_to_base64(img_path)}

    img = Image.open(img_path)
    img_arr = np.asarray(img)
    image_height, image_width = img_arr.shape[:2]

    start = time.time()
    result = ocr.ocr(img_arr, cls=False)
    elapsed = time.time() - start

    for output in result:
        if output is None:
            continue
        for item in output:
            co_ord = item[0]
            text = fix_norwegian(item[1][0])

            four_co_ord = [
                co_ord[0][0],
                co_ord[1][1],
                co_ord[2][0] - co_ord[0][0],
                co_ord[2][1] - co_ord[1][1]
            ]

            bbox = {
                'x': 100 * four_co_ord[0] / image_width,
                'y': 100 * four_co_ord[1] / image_height,
                'width': 100 * four_co_ord[2] / image_width,
                'height': 100 * four_co_ord[3] / image_height,
                'rotation': 0
            }

            if not text:
                continue

            region_id = str(uuid4())[:10]
            score = 0.5
            bbox_result = {
                'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
                'value': bbox
            }
            transcription_result = {
                'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
                'value': dict(text=[text], **bbox), 'score': score
            }
            annotation_result.extend([bbox_result, transcription_result])

    output_json['predictions'] = [{"result": annotation_result, "score": 0.97}]

    json_filename = os.path.join(OUTPUT_JSON_DIR, f'{image_file[:-4]}_label_studio.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    return image_file, len(annotation_result) // 2, elapsed


def run_ocr_pipeline():
    images = [f for f in os.listdir(OUTPUT_IMAGES_DIR)
              if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

    if not images:
        print("Ingen bilder funnet i output_images!")
        return

    total = len(images)
    workers = min(NUM_WORKERS, total, cpu_count())

    print(f"Fant {total} bilder i output_images")
    print(f"Starter {workers} parallelle prosesser (GPU: {'Ja' if OCR_USE_GPU else 'Nei'})")
    print("=" * 60)

    start_total = time.time()

    args_list = [(img, OUTPUT_IMAGES_DIR) for img in images]

    if workers > 1 and not OCR_USE_GPU:
        with Pool(processes=workers) as pool:
            for i, (image_file, num_texts, elapsed) in enumerate(pool.imap_unordered(process_single_image, args_list)):
                print(f"  [{i+1}/{total}] {image_file} — {num_texts} tekster — {elapsed:.1f}s")
    else:
        for i, args in enumerate(args_list):
            image_file, num_texts, elapsed = process_single_image(args)
            print(f"  [{i+1}/{total}] {image_file} — {num_texts} tekster — {elapsed:.1f}s")

    total_time = time.time() - start_total
    avg_time = total_time / total if total > 0 else 0

    print("=" * 60)
    print(f"Ferdig! {total} bilder behandlet")
    print(f"Total tid: {total_time:.1f}s | Gjennomsnitt: {avg_time:.1f}s/bilde")


if __name__ == "__main__":
    run_ocr_pipeline()
