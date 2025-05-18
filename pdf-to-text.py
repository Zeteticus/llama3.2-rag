import os
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import time

def extract_text_with_pdfminer(pdf_path):
    output_string = StringIO()
    with open(pdf_path, 'rb') as pdf_file:
        extract_text_to_fp(pdf_file, output_string, laparams=LAParams(), 
                           output_type='text', codec='utf-8')
    return output_string.getvalue().strip()

def extract_text_with_ocr(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n\n"
    return text.strip()

def pdf_to_text(pdf_path, output_path):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_file_path = os.path.join(output_path, f"{base_name}.txt")

    if os.path.exists(txt_file_path):
        print(f"Text file already exists for {base_name}. Skipping conversion.")
        return

    print(f"Processing {base_name}")

    try:
        # Try pdfminer first
        text = extract_text_with_pdfminer(pdf_path)
        
        # If pdfminer doesn't extract any text, use OCR
        if not text.strip():
            print(f"No text found with pdfminer for {base_name}. Trying OCR...")
            text = extract_text_with_ocr(pdf_path)

        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        print(f"Text extracted and saved to {txt_file_path}")

    except Exception as e:
        print(f"Error processing {base_name}: {str(e)}")

def process_directory(input_dir, output_dir, batch_size=15, delay=10):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {len(pdf_files)//batch_size + 1}")
        
        for filename in batch:
            pdf_path = os.path.join(input_dir, filename)
            pdf_to_text(pdf_path, output_dir)
        
        if i + batch_size < len(pdf_files):
            print(f"Batch complete. Waiting for {delay} seconds before next batch...")
            time.sleep(delay)

# Usage
input_directory = 'pdfs'
output_directory = 'textfiles'

process_directory(input_directory, output_directory, batch_size=15, delay=10)
