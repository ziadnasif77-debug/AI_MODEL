# Utils - NAV OCR System

from paddleocr import PaddleOCR
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ── NAV Labels ──
NAV_LABELS = {
    0: "O",
    1: "NAVN",
    2: "FODSELSNUMMER",
    3: "DATO",
    4: "ADRESSE",
    5: "SIGNATUR"
}

NAV_LABEL_COLORS = {
    0: "gray",
    1: "red",
    2: "cyan",
    3: "blue",
    4: "green",
    5: "yellow"
}


def read_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as fp:
        data = json.loads(fp.read())
    return data


def train_data_format(json_to_dict: list):
    # ── NAV Label mapping ──
    label_map = {
        "O":              0,
        "NAVN":           1,
        "FODSELSNUMMER":  2,
        "DATO":           3,
        "ADRESSE":        4,
        "SIGNATUR":       5,
    }

    final_list = []
    count = 0
    for item in json_to_dict:
        count += 1
        test_dict = {"id": int, "tokens": [], "bboxes": [], "ner_tag": []}
        test_dict["id"]       = count
        test_dict["img_path"] = item['file_name']

        for cont in item['annotations']:
            test_dict['tokens'].append(cont['text'])
            test_dict['bboxes'].append(cont['box'])
            # Konverter label-streng til tall
            label_str = cont['label']
            test_dict['ner_tag'].append(label_map.get(label_str, 0))

        final_list.append(test_dict)

    return final_list


# OCR-motor for inferens
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='latin',
    use_gpu=False,
)


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


def scale_bounding_box(box: list, width: float, height: float) -> list:
    return [
        100 * box[0] / width,
        100 * box[1] / height,
        (100 * box[0] / width) + box[2],
        (100 * box[1] / height) + box[3]
    ]


def process_bbox(box: list):
    return [box[0][0], box[1][1], box[2][0] - box[0][0], box[2][1] - box[1][1]]


def dataSetFormat(img_file):
    width, height = img_file.size
    ress = ocr.ocr(np.asarray(img_file))

    test_dict = {'tokens': [], "bboxes": []}
    test_dict['img_path'] = img_file

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
