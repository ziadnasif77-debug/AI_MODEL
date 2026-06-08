# config.py - NAV OCR Konfigurasjon

import os

# ── Grunnsti (endre denne ved flytting til ny maskin) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Mapper ──
DATA_DIR          = os.path.join(BASE_DIR, 'Data')
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, 'output_images')
OUTPUT_JSON_DIR   = os.path.join(BASE_DIR, 'output_json')
PENDING_DIR       = os.path.join(BASE_DIR, 'training_data', 'pending')
TRAINED_DIR       = os.path.join(BASE_DIR, 'training_data', 'trained')
TRAINING_JSON     = os.path.join(BASE_DIR, 'training_data', 'Training_layoutLMV3.json')
MODEL_DIR         = os.path.join(BASE_DIR, 'models', 'layoutlmv3')
OUTPUT_MODEL_DIR  = os.path.join(BASE_DIR, 'models', 'nav_layoutlmv3')

# ── PaddleOCR ──
OCR_LANG    = 'latin'
OCR_USE_GPU = True
OCR_GPU_MEM = 500

# ── Parallellbehandling ──
NUM_WORKERS = 4

# ── LayoutLMv3 Trening ──
NUM_CLASSES   = 6
BATCH_SIZE    = 2
LEARNING_RATE = 5e-5
EPOCHS        = 30
MAX_LENGTH    = 512

# ── Treningskontroll ──
VALIDATION_SPLIT        = 0.2
SAVE_EVERY              = 5
EARLY_STOPPING_PATIENCE = 5
DATALOADER_WORKERS      = 2
RANDOM_SEED             = 42

# ── NAV Labels ──
NAV_LABELS = {
    0: "O",
    1: "NAVN",
    2: "FODSELSNUMMER",
    3: "DATO",
    4: "ADRESSE",
    5: "SIGNATUR"
}

NAV_LABEL_MAP = {
    "O":             0,
    "NAVN":          1,
    "FODSELSNUMMER": 2,
    "DATO":          3,
    "ADRESSE":       4,
    "SIGNATUR":      5,
}

NAV_LABEL_COLORS = {
    0: "gray",
    1: "red",
    2: "cyan",
    3: "blue",
    4: "green",
    5: "yellow"
}

# ── Norsk tekstkorrigering ──
NORWEGIAN_SAFE_WORDS = {
    'Fodselsnummer': 'Fødselsnummer',
    'fodselsnummer': 'fødselsnummer',
    'Fodselsdato': 'Fødselsdato',
    'fodselsdato': 'fødselsdato',
}

NORWEGIAN_REPLACEMENTS = {
    'ae': 'æ', 'AE': 'Æ',
    'oe': 'ø', 'OE': 'Ø',
    'aa': 'å', 'AA': 'Å',
}

_SAFE_EXCEPTIONS = {'michael', 'raphael', 'israel', 'joel', 'noel', 'poet', 'roer', 'aachen', 'baal', 'kraal'}


def fix_norwegian(text):
    for wrong, correct in NORWEGIAN_SAFE_WORDS.items():
        text = text.replace(wrong, correct)
    if text.lower() in _SAFE_EXCEPTIONS:
        return text
    for wrong, correct in NORWEGIAN_REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    return text


# ── PDF-konvertering ──
PDF_DPI = 300
