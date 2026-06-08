import json
import os
import shutil
import tempfile
from config import PENDING_DIR, TRAINED_DIR, TRAINING_JSON, OUTPUT_IMAGES_DIR


def convert_bounding_box(x, y, width, height):
    x1 = int(x * 10)
    y1 = int(y * 10)
    x2 = int((x + width) * 10)
    y2 = int((y + height) * 10)
    return [x1, y1, x2, y2]


os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)

json_files = [f for f in os.listdir(PENDING_DIR)
              if f.endswith('.json')]

print(f"Fant {len(json_files)} JSON-filer i training_data/pending")
print("="*50)

if os.path.exists(TRAINING_JSON):
    with open(TRAINING_JSON, 'r', encoding='utf-8') as f:
        output = json.load(f)
    print(f"Lastet {len(output)} eksisterende bilder fra Training_layoutLMV3.json")
else:
    output = []
    print("Ingen eksisterende data — starter fra scratch")

existing_files = {item['file_name'] for item in output}
new_count = 0

for json_file in json_files:
    json_path = os.path.join(PENDING_DIR, json_file)
    print(f"\nLeser: {json_file}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for task in data:
        data_item = {}
        ann_list  = []

        file_upload = task.get('file_upload', '')
        image_name  = os.path.basename(file_upload)
        if '-' in image_name:
            parts = image_name.split('-', 1)
            if len(parts[0]) == 8 and parts[0].isalnum():
                image_name = parts[1]
        image_name = image_name.replace('_label_studio.json', '.png')

        data_item["file_name"] = os.path.join(OUTPUT_IMAGES_DIR, image_name)
        print(f"  Fil: {image_name}")

        img_path = data_item["file_name"]
        if os.path.exists(img_path):
            from PIL import Image
            img = Image.open(img_path)
            data_item["width"], data_item["height"] = img.size
        else:
            data_item["width"] = 880
            data_item["height"] = 700

        annotations = task.get('annotations', [])
        if annotations and annotations[0].get('result'):
            first = annotations[0]['result'][0]
            data_item["width"]  = first.get('original_width', data_item["width"])
            data_item["height"] = first.get('original_height', data_item["height"])

        if annotations:
            results = annotations[0].get('result', [])

            bbox_dict  = {}
            text_dict  = {}
            label_dict = {}

            for r in results:
                rid   = r.get('id', '')
                rtype = r.get('type', '')
                val   = r.get('value', {})

                if rtype == 'rectangle':
                    bbox_dict[rid] = val

                elif rtype == 'textarea':
                    text_dict[rid] = val.get('text', [''])[0]

                elif rtype == 'rectanglelabels':
                    label_dict[rid] = val.get('rectanglelabels', ['O'])[0]

                if val.get('rectanglelabels'):
                    label_dict[rid] = val['rectanglelabels'][0]

            for rid, bbox in bbox_dict.items():
                text  = text_dict.get(rid, '')
                label = label_dict.get(rid, 'O')
                if text:
                    ann_dict = {
                        "box":   convert_bounding_box(
                            bbox['x'], bbox['y'],
                            bbox['width'], bbox['height']
                        ),
                        "text":  text,
                        "label": label
                    }
                    ann_list.append(ann_dict)
                    print(f"    {label}: {text}")

        data_item["annotations"] = ann_list

        if data_item["file_name"] in existing_files:
            print(f"  Finnes allerede — oppdaterer")
            output = [item for item in output if item['file_name'] != data_item['file_name']]

        output.append(data_item)
        new_count += 1

    shutil.move(json_path, os.path.join(TRAINED_DIR, json_file))
    print(f"  Flyttet til training_data/trained")

tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(TRAINING_JSON), suffix='.json')
with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
os.replace(tmp_path, TRAINING_JSON)

print(f"\n{'='*50}")
print(f"Nye bilder lagt til:    {new_count}")
print(f"Totalt i treningsdata:  {len(output)}")
print(f"Lagret til: {TRAINING_JSON}")
