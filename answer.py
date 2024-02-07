import os
import PyPDF2

DATA_PATH = "C:/Users/Aditi Agarwal/Desktop/Pythia/data"  # Replace with your PDF directory

pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(DATA_PATH, pdf_file)
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"

    with open(pdf_path, 'rb') as pdf_file_object:
        pdf_reader = PyPDF2.PdfReader(pdf_file_object)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()  # Extract text from the page

            # Ensure consistent spacing:
            page_text = page_text.replace("\n", " ")  # Replace newlines with spaces
            page_text = " ".join(page_text.split())  # Remove redundant spaces

            text += page_text

    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
