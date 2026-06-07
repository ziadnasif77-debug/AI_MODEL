import json
import os
import shutil

# ── Grunnsti ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_bounding_box(x, y, width, height):
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return [x1, y1, x2, y2]


# ── Stier ──
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')
DONE_DIR          = os.path.join(BASE_DIR, 'training_data', 'trained')
OUTPUT_JSON       = os.path.join(BASE_DIR, 'training_data', 'Training_layoutLMV3.json')
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, 'output_images')

os.makedirs(DONE_DIR, exist_ok=True)

json_files = [f for f in os.listdir(TRAINING_DATA_DIR)
              if f.endswith('.json') and f != 'Training_layoutLMV3.json']

print(f"Fant {len(json_files)} JSON-filer i training_data")
print("="*50)

output = []

for json_file in json_files:
    json_path = os.path.join(TRAINING_DATA_DIR, json_file)
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

        data_item["file_name"] = os.path.join(OUTPUT_IMAGES_DIR, image_name)
        print(f"  Fil: {image_name}")

        annotations = task.get('annotations', [])
        if annotations and annotations[0].get('result'):
            first = annotations[0]['result'][0]
            data_item["width"]  = first.get('original_width', 880)
            data_item["height"] = first.get('original_height', 700)

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
        output.append(data_item)

    shutil.move(json_path, os.path.join(DONE_DIR, json_file))
    print(f"  Flyttet til trained")

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"Ferdig! {len(output)} bilder konvertert")
print(f"Lagret til: {OUTPUT_JSON}")
