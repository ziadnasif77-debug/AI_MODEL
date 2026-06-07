import os
import base64
from paddleocr import PaddleOCR
from PIL import Image
import json
from uuid import uuid4
import numpy as np

# ── Grunnsti ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialiser OCR-motor
ocr = PaddleOCR(use_angle_cls=False, lang='latin', use_gpu=False)


def fix_norwegian(text):
    replacements = {
        'ae': 'æ', 'AE': 'Æ',
        'oe': 'ø', 'OE': 'Ø', 'o/': 'ø',
        'aa': 'å', 'AA': 'Å',
        'Fodselsnummer': 'Fødselsnummer',
        'fodselsnummer': 'fødselsnummer',
        'Fodselsdato': 'Fødselsdato',
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

images_folder_path = os.path.join(BASE_DIR, 'output_images')
output_json_folder = os.path.join(BASE_DIR, 'output_json')

os.makedirs(images_folder_path, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = image_path.split('.')[-1].lower()
    return f"data:image/{ext};base64,{data}"


def extracted_tables_to_label_studio_json_file_with_paddleOCR(images_folder_path):
    images = [f for f in os.listdir(images_folder_path)
              if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

    if not images:
        print("Ingen bilder funnet i input_images!")
        return

    print(f"Fant {len(images)} bilder i input_images")
    print("="*50)

    for image_file in images:
        output_json = {}
        annotation_result = []

        print(f"\nBehandler: {image_file}")

        img_path = os.path.join(images_folder_path, image_file)

        output_json['data'] = {"ocr": image_to_base64(img_path)}

        img = Image.open(img_path)
        img_arr = np.asarray(img)
        image_height, image_width = img_arr.shape[:2]

        result = ocr.ocr(img_arr, cls=False)

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

        json_filename = os.path.join(output_json_folder, f'{image_file[:-4]}_label_studio.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        print(f"  JSON lagret: {json_filename}")

    print("\n" + "="*50)
    print("Behandling fullfort!")


extracted_tables_to_label_studio_json_file_with_paddleOCR(images_folder_path)
