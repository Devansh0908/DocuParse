import os
import fitz  # PyMuPDF
import pandas as pd
import joblib
import json

MODEL_PATH = "./ml/model.pkl"
PDF_DIR = "./dataset/new_pdfs"  # Changed to new_pdfs
OUTPUT_DIR = "./dataset/outputs"

# Feature extraction (same as training)
def extract_features(df):
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["is_uppercase"] = df["text"].apply(lambda x: int(str(x).isupper()))
    df["is_titlecase"] = df["text"].apply(lambda x: int(str(x).istitle()))
    features = df[[
        "font_size", "x_position", "y_position", "page",
        "text_length", "word_count", "is_uppercase", "is_titlecase"
    ]].fillna(0)
    return features

def extract_text_from_pdf(pdf_path):
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
                            "y_position": round(span["bbox"][1], 2)
                        })
    return extracted_data

def predict_pdf(pdf_path, model, le):
    data = extract_text_from_pdf(pdf_path)
    if not data:
        return None
    df = pd.DataFrame(data)
    X = extract_features(df)
    y_pred = model.predict(X)
    levels = le.inverse_transform(y_pred)
    df["level"] = levels
    return df

def pdf_to_json(pdf_path, df):
    # Try to get the first text with level 'title'
    title_row = df[df["level"] == "title"]
    if not title_row.empty:
        title = title_row.iloc[0]["text"]
    else:
        title = os.path.splitext(os.path.basename(pdf_path))[0]
    outline = []
    for _, row in df.iterrows():
        outline.append({
            "level": row["level"],
            "text": row["text"],
            "page": int(row["page"])
        })
    return {
        "title": title,
        "outline": outline
    }

def main():
    model, le = joblib.load(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"Predicting headings for: {filename}")
            df = predict_pdf(pdf_path, model, le)
            if df is None or df.empty:
                print(f"No text found in {filename}")
                continue
            # Only keep rows with predicted heading levels
            df = df[df["level"].isin(["title", "H1", "H2", "H3", "H4"])]
            if df.empty:
                print(f"No headings detected in {filename}")
                continue
            json_data = pdf_to_json(pdf_path, df)
            json_path = os.path.join(OUTPUT_DIR, filename.replace('.pdf', '.json'))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"JSON outline saved to: {json_path}")

if __name__ == "__main__":
    main()
