import pymupdf  # PyMuPDF
import os
from config import DATA_DIR, OUTPUT_IMAGES_DIR, PDF_DPI

# Parametere
zoom = PDF_DPI / 72
magnify = pymupdf.Matrix(zoom, zoom)

# Sikre at utdata-mappen finnes
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# Behandle hver PDF i data-mappen
for pdf_file in os.listdir(DATA_DIR):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        doc = pymupdf.open(pdf_path)

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=magnify)
            output_path = os.path.join(OUTPUT_IMAGES_DIR, f"{pdf_file[:-4]}_page_{page_num + 1}.png")
            pix.save(output_path)
            print(f"Lagret: {output_path}")

print("PDF til bilde konvertering fullfort.")
