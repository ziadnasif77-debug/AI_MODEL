# NAV OCR System — LayoutLMv3 + PaddleOCR + Label Studio

## Oversikt
Dette systemet ekstraherer strukturerte data fra NAV-dokumenter ved hjelp av LayoutLMv3, PaddleOCR og Label Studio.

## Felt som ekstraheres
- NAVN
- FODSELSNUMMER
- DATO
- ADRESSE
- SIGNATUR

## Mappstruktur
```
E:\AI MODOL\
├── Data\                        ← PDF-filer hit
├── output_images\               ← Konverterte PNG-bilder
├── output_json\                 ← PaddleOCR JSON for Label Studio
├── images\                      ← Treningsbilder
├── models\
│   ├── layoutlmv3\              ← Forhåndstrent modell
│   └── nav_layoutlmv3\         ← Trent NAV-modell
├── training_data\
│   ├── Training_json.json       ← Eksportert fra Label Studio
│   └── Training_layoutLMV3.json ← Klar for trening
└── src\
    ├── main.py
    ├── engine.py
    ├── loader.py
    ├── trainer.py
    ├── utils.py
    └── Inference.py
```

## Installasjon
```powershell
pip install -r requirements.txt
```

## Slik bruker du systemet

### Steg 1 — Konverter PDF til bilder
```powershell
python Convert.py
```
Legg PDF-filer i `E:\AI MODOL\Data\` — bilder lagres i `output_images\`

### Steg 2 — Generer JSON for Label Studio
```powershell
python Create_LMv3_dataset_with_paddleOCR.py
```
JSON-filer lagres i `output_json\`

### Steg 3 — Importer i Label Studio
1. Åpne http://localhost:8081
2. Importer JSON-filene fra `output_json\`
3. Teamet merker feltene (NAVN, FODSELSNUMMER, DATO, ADRESSE, SIGNATUR)
4. Eksporter som `Training_json.json` til `training_data\`

### Steg 4 — Konverter til LayoutLMv3-format
```powershell
python Label_studio_to_layoutLMV3.py
```

### Steg 5 — Tren modellen
```powershell
python src\main.py
```

### Steg 6 — Kjør inferens på nye dokumenter
```powershell
python src\Inference.py
```

## Teknologi
- LayoutLMv3 (Microsoft) — forstår tekst + layout
- PaddleOCR 3.0 — tekstgjenkjenning
- Label Studio — teambasert merking
- PyTorch — trening
- Apache 2.0 — 100% åpen kildekode
