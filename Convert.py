import pymupdf  # PyMuPDF
import os

# Parameters
dpi = 300
zoom = dpi / 72
magnify = pymupdf.Matrix(zoom, zoom)

# Paths
input_folder  = r'E:\AI MODOL\Data'
output_folder = r'E:\AI MODOL\output_images'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each PDF in the data folder
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