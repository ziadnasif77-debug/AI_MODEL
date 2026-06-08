# Utils - NAV OCR System

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddleocr import PaddleOCR
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from config import (
    OUTPUT_IMAGES_DIR, OCR_LANG, OCR_USE_GPU,
    NAV_LABELS, NAV_LABEL_MAP, NAV_LABEL_COLORS,
    fix_norwegian
)


def read_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as fp:
        data = json.loads(fp.read())
    return data


def train_data_format(json_to_dict: list):
    final_list = []
    count = 0
    for item in json_to_dict:
        count += 1
        test_dict = {"id": int, "tokens": [], "bboxes": [], "ner_tag": []}
        test_dict["id"]       = count
        img_name = os.path.basename(item['file_name'])
        test_dict["img_path"] = os.path.join(OUTPUT_IMAGES_DIR, img_name)

        for cont in item['annotations']:
            test_dict['tokens'].append(cont['text'])
            test_dict['bboxes'].append(cont['box'])
            label_str = cont['label']
            test_dict['ner_tag'].append(NAV_LABEL_MAP.get(label_str, 0))

        final_list.append(test_dict)

    return final_list


_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_angle_cls=False,
            lang=OCR_LANG,
            use_gpu=OCR_USE_GPU,
        )
    return _ocr


def scale_bounding_box(box: list, width: float, height: float) -> list:
    return [
        100 * box[0] / width,
        100 * box[1] / height,
        100 * (box[0] + box[2]) / width,
        100 * (box[1] + box[3]) / height
    ]


def process_bbox(box: list):
    return [box[0][0], box[1][1], box[2][0] - box[0][0], box[2][1] - box[1][1]]


def dataSetFormat(img_file):
    width, height = img_file.size
    ocr = _get_ocr()
    ress = ocr.ocr(np.asarray(img_file))

    test_dict = {'tokens': [], "bboxes": []}
    test_dict['img_path'] = img_file

    if ress and ress[0]:
        for item in ress[0]:
            test_dict['tokens'].append(fix_norwegian(item[1][0]))
            test_dict['bboxes'].append(scale_bounding_box(process_bbox(item[0]), width, height))

    return test_dict, width, height


def plot_img(im, bbox_list, label_list, prob_list, width, height):
    plt.imshow(im)
    ax = plt.gca()

    for i, (item,) in enumerate(zip(bbox_list)):
        print("Element:", item)
        label_id = int(label_list[i]) if i < len(label_list) else 0
        label_name  = NAV_LABELS.get(label_id, "O")
        label_color = NAV_LABEL_COLORS.get(label_id, "gray")

        rect = Rectangle(
            (item[0] * width / 100, item[1] * height / 100),
            item[2] - item[0],
            item[3] - item[1],
            linewidth=1,
            edgecolor=label_color,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            item[0] * width / 100,
            item[1] * height / 100,
            f"{label_name}",
            bbox={'facecolor': [1, 1, 1], 'alpha': 0.5},
            clip_box=ax.clipbox,
            clip_on=True
        )

    plt.show()
    plt.savefig("nav_inference_result.jpg")
    print("Ferdig — lagret som nav_inference_result.jpg")
    plt.clf()
