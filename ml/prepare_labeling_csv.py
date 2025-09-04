import pandas as pd

# Input and output file paths
INPUT_CSV = 'dataset/outputs/labeling_ready_dataset.csv'
OUTPUT_CSV = 'dataset/outputs/sample_labeled.csv'

# Allowed heading levels
ALLOWED_LEVELS = ['title', 'H1', 'H2', 'H3', 'H4']

# Columns to keep
COLUMNS = ['text', 'level', 'font_size', 'page', 'x_position', 'y_position', 'source_pdf']

def main():
    df = pd.read_csv(INPUT_CSV)
    filtered = df[df['level'].isin(ALLOWED_LEVELS)]
    filtered = filtered[COLUMNS]
    filtered['label'] = filtered['level']  # Set label equal to level
    filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"Sample labeled CSV (with label=level) saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main() 