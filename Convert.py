import pymupdf  # PyMuPDF
import os

# ── Grunnsti ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parametere
dpi = 300
zoom = dpi / 72
magnify = pymupdf.Matrix(zoom, zoom)

# Stier
input_folder  = os.path.join(BASE_DIR, 'Data')
output_folder = os.path.join(BASE_DIR, 'output_images')

# Sikre at utdata-mappen finnes
os.makedirs(output_folder, exist_ok=True)

# Behandle hver PDF i data-mappen
for pdf_file in os.listdir(input_folder):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(input_folder, pdf_file)
        doc = pymupdf.open(pdf_path)

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=magnify)
            output_path = os.path.join(output_folder, f"{pdf_file[:-4]}_page_{page_num + 1}.png")
            pix.save(output_path)
            print(f"Lagret: {output_path}")

print("PDF til bilde konvertering fullfort.")
