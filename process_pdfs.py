import os
import fitz  # PyMuPDF
import pandas as pd
from collections import Counter

PDF_DIR = "./dataset/pdfs"
OUTPUT_CSV = "./dataset/outputs/combined_dataset.csv"

def extract_text_from_pdf(pdf_path, source_pdf):
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        extracted_data.append({
                            "text": text,
                            "font_size": round(span["size"], 2),
                            "page": page_num,
                            "x_position": round(span["bbox"][0], 2),
                            "y_position": round(span["bbox"][1], 2),
                            "source_pdf": source_pdf
                        })
    return extracted_data

def detect_heading_levels(df, max_heading_levels=5):
    size_counts = Counter(df['font_size'])
    sorted_sizes = sorted(size_counts.keys(), reverse=True)

    heading_labels = ['title'] + [f'H{i}' for i in range(1, max_heading_levels)]
    size_to_level = {}
    for i, size in enumerate(sorted_sizes):
        if i < len(heading_labels):
            size_to_level[size] = heading_labels[i]
        else:
            size_to_level[size] = 'body'

    df['level'] = df['font_size'].map(size_to_level)
    return df[df['level'] != 'body']  # Remove body text

def process_all_pdfs():
    all_data = []

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, filename)
            print(f"Processing: {filename}")
            pdf_data = extract_text_from_pdf(file_path, filename)
            if not pdf_data:
                continue

            df = pd.DataFrame(pdf_data)
            df = detect_heading_levels(df)
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Combined CSV saved at: {OUTPUT_CSV}")
    else:
        print("⚠️ No data extracted from PDFs.")

if __name__ == "__main__":
    process_all_pdfs()
