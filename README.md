# PDF Heading Extractor

A machine learning-based tool that automatically extracts and classifies headings from PDF documents. This project uses font size analysis and machine learning to identify different heading levels (Title, H1, H2, H3, H4) in PDF documents.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Docker Setup](#docker-setup)
- [Project Structure](#project-structure)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Input/Output Specifications](#inputoutput-specifications)
- [Troubleshooting](#troubleshooting)

## Overview

This project consists of four main components:

1. **PDF Processing** (`process_pdfs.py`) - Extracts text and metadata from PDFs
2. **Data Preparation** (`ml/prepare_labeling_csv.py`) - Prepares data for machine learning
3. **Model Training** (`ml/train_model.py`) - Trains a Random Forest classifier
4. **Prediction** (`ml/predict_headings.py`) - Predicts headings in new PDFs

## Prerequisites

### Local Installation
- Python 3.7 or higher
- pip (Python package installer)

### Docker Installation
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose

## Installation

### Option 1: Local Installation

1. **Clone or download the project** to your local machine
2. **Install required dependencies**:

```bash
pip install pandas scikit-learn joblib PyMuPDF
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation (Recommended)

1. **Clone or download the project** to your local machine
2. **Build the Docker image**:

```bash
# Windows
run.bat build

# Linux/Mac
./run.sh build
```

3. **Start the container**:

```bash
# Windows
run.bat start

# Linux/Mac
./run.sh start
```

## Docker Setup

### Quick Start with Docker

The easiest way to run the PDF Heading Extractor is using Docker. This ensures consistent behavior across different environments.

#### Available Commands

**Windows Users:**
```bash
run.bat build      # Build Docker image
run.bat start      # Start container
run.bat stop       # Stop container
run.bat restart    # Restart container
run.bat shell      # Open shell in container
run.bat process    # Run PDF processing
run.bat prepare    # Run data preparation
run.bat train      # Train model
run.bat predict    # Run predictions
run.bat logs       # Show container logs
run.bat help       # Show all commands
```

**Linux/Mac Users:**
```bash
./run.sh build     # Build Docker image
./run.sh start     # Start container
./run.sh stop      # Stop container
./run.sh restart   # Restart container
./run.sh shell     # Open shell in container
./run.sh process   # Run PDF processing
./run.sh prepare   # Run data preparation
./run.sh train     # Train model
./run.sh predict   # Run predictions
./run.sh logs      # Show container logs
./run.sh help      # Show all commands
```

#### Complete Docker Workflow

1. **Build and start the container:**
   ```bash
   run.bat build    # or ./run.sh build
   run.bat start    # or ./run.sh start
   ```

2. **Run the complete pipeline:**
   ```bash
   run.bat process  # Extract data from PDFs
   run.bat prepare  # Prepare data for training
   run.bat train    # Train the model
   run.bat predict  # Predict on new PDFs
   ```

3. **Access the container shell for manual operations:**
   ```bash
   run.bat shell    # or ./run.sh shell
   ```

#### Docker Benefits

- **Consistency**: Same environment across different machines
- **Isolation**: No conflicts with system Python packages
- **Portability**: Easy to deploy on any system with Docker
- **Data Persistence**: Volumes ensure data survives container restarts
- **Easy Management**: Simple commands to build, run, and manage

## Project Structure

```
Challenge_1a/
├── dataset/
│   ├── pdfs/                    # Training PDFs (input)
│   ├── new_pdfs/               # New PDFs for prediction (input)
│   └── outputs/                # All generated outputs
│       ├── combined_dataset.csv
│       ├── labeling_ready_dataset.csv
│       └── sample_labeled.csv
├── ml/
│   ├── model.pkl              # Trained model (output)
│   ├── train_model.py         # Model training script
│   ├── predict_headings.py    # Prediction script
│   └── prepare_labeling_csv.py # Data preparation script
├── process_pdfs.py            # PDF processing script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore              # Docker build exclusions
├── run.bat                    # Windows batch runner script
├── run.sh                     # Linux/Mac shell runner script
└── README.md                  # This file
```

## Step-by-Step Workflow

### Step 1: Prepare Training Data

**Input**: Place your training PDF files in the `dataset/pdfs/` directory

**Command**:
```bash
# Local installation
python process_pdfs.py

# Docker installation
run.bat process    # Windows
./run.sh process   # Linux/Mac
```

**What it does**:
- Extracts text, font size, position, and page number from all PDFs
- Automatically detects heading levels based on font size hierarchy
- Filters out body text, keeping only potential headings
- Saves results to `dataset/outputs/combined_dataset.csv`

**Output**: `dataset/outputs/combined_dataset.csv`
- Columns: `text`, `level`, `font_size`, `page`, `x_position`, `y_position`, `source_pdf`
- Contains all extracted headings with automatically assigned levels

**Sample Output**:
```csv
text,level,font_size,page,x_position,y_position,source_pdf
"Introduction",H1,16.0,1,72.0,144.0,document1.pdf
"Background",H2,14.0,1,72.0,180.0,document1.pdf
"Methods",H2,14.0,2,72.0,144.0,document1.pdf
```

### Step 2: Prepare Labeling Dataset

**Input**: Uses `dataset/outputs/combined_dataset.csv` from Step 1

**Command**:
```bash
# Local installation
python ml/prepare_labeling_csv.py

# Docker installation
run.bat prepare    # Windows
./run.sh prepare   # Linux/Mac
```

**What it does**:
- Filters data to include only allowed heading levels: `title`, `H1`, `H2`, `H3`, `H4`
- Adds a `label` column equal to the detected `level`
- Creates a clean dataset ready for machine learning

**Output**: `dataset/outputs/sample_labeled.csv`
- Columns: `text`, `level`, `font_size`, `page`, `x_position`, `y_position`, `source_pdf`, `label`
- Ready for model training

### Step 3: Train the Machine Learning Model

**Input**: Uses `dataset/outputs/sample_labeled.csv` from Step 2

**Command**:
```bash
# Local installation
python ml/train_model.py

# Docker installation
run.bat train      # Windows
./run.sh train     # Linux/Mac
```

**What it does**:
- Extracts features from the text data:
  - `text_length`: Number of characters
  - `word_count`: Number of words
  - `is_uppercase`: Whether text is all uppercase
  - `is_titlecase`: Whether text is title case
  - `font_size`, `x_position`, `y_position`, `page`: Original metadata
- Trains a Random Forest classifier
- Splits data into training (80%) and validation (20%) sets
- Evaluates model performance and displays metrics
- Saves the trained model and label encoder

**Output**: `ml/model.pkl`
- Contains the trained Random Forest model and LabelEncoder
- Used for making predictions on new PDFs

**Sample Console Output**:
```
| Level  | Precision | Recall | Frequency |
|--------|-----------|--------|-----------|
| title  |   0.95    |  0.92 |     15    |
| H1     |   0.88    |  0.85 |     25    |
| H2     |   0.82    |  0.78 |     30    |
| H3     |   0.75    |  0.70 |     20    |
| H4     |   0.68    |  0.65 |     10    |
|--------|-----------|--------|-----------|

Accuracy: 82.50%
```

### Step 4: Predict Headings in New PDFs

**Input**: Place new PDF files in the `dataset/new_pdfs/` directory

**Command**:
```bash
# Local installation
python ml/predict_headings.py

# Docker installation
run.bat predict    # Windows
./run.sh predict   # Linux/Mac
```

**What it does**:
- Loads the trained model from `ml/model.pkl`
- Processes each PDF in the `new_pdfs/` directory
- Extracts text and metadata (same as training)
- Applies the trained model to predict heading levels
- Generates JSON outlines for each PDF

**Output**: JSON files in `dataset/outputs/` directory
- One JSON file per PDF with the same name (`.json` extension)
- Contains structured outline with title and heading hierarchy

**Sample JSON Output** (`document1.json`):
```json
{
  "title": "Research Paper Title",
  "outline": [
    {
      "level": "title",
      "text": "Research Paper Title",
      "page": 1
    },
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Background",
      "page": 1
    },
    {
      "level": "H1",
      "text": "Methods",
      "page": 2
    }
  ]
}
```

## Input/Output Specifications

### Input Requirements

**Training PDFs** (`dataset/pdfs/`):
- Format: PDF files
- Content: Documents with clear heading hierarchy
- Font sizes should vary between heading levels
- Recommended: 5-20 PDFs for good training

**New PDFs** (`dataset/new_pdfs/`):
- Format: PDF files
- Content: Any PDF document you want to extract headings from
- Should have similar structure to training PDFs for best results

### Output Files

1. **`combined_dataset.csv`**: Raw extracted data from training PDFs
2. **`labeling_ready_dataset.csv`**: Cleaned data ready for labeling
3. **`sample_labeled.csv`**: Final training dataset with labels
4. **`model.pkl`**: Trained machine learning model
5. **`*.json`**: Structured outlines for each processed PDF

### Data Columns

**Extracted Data Columns**:
- `text`: The actual text content
- `font_size`: Font size in points
- `page`: Page number where text appears
- `x_position`: X-coordinate on the page
- `y_position`: Y-coordinate on the page
- `source_pdf`: Original PDF filename
- `level`: Detected heading level (title, H1, H2, H3, H4)
- `label`: Training label (same as level)

## Troubleshooting

### Docker Issues

**1. "Docker command not found"**
- Install Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Ensure Docker is running and accessible from command line

**2. "Permission denied" (Linux/Mac)**
```bash
sudo chmod +x run.sh
```

**3. "Container not starting"**
- Check if Docker has enough resources allocated
- Ensure ports are not already in use
- Check Docker logs: `run.bat logs` or `./run.sh logs`

**4. "Volume mount issues"**
- Ensure directories exist: `dataset/pdfs/`, `dataset/new_pdfs/`, `dataset/outputs/`, `ml/`
- Check file permissions on mounted directories

### Common Issues

**1. "No module named 'fitz'"**
```bash
# Local installation
pip install PyMuPDF

# Docker installation
run.bat build    # Rebuild the image
```

**2. "No PDFs found in directory"**
- Ensure PDF files are in the correct directory (`dataset/pdfs/` or `dataset/new_pdfs/`)
- Check file extensions are `.pdf` (case-sensitive)
- For Docker: Ensure volumes are properly mounted

**3. "No headings detected"**
- PDF may not have clear font size differences
- Try with PDFs that have distinct heading styles
- Check if PDF contains selectable text (not scanned images)

**4. "Poor prediction accuracy"**
- Increase training data with more diverse PDFs
- Ensure training PDFs have similar structure to target PDFs
- Check font size hierarchy in training documents

### Performance Tips

1. **Training Data Quality**: Use PDFs with clear, consistent heading styles
2. **Font Size Variation**: Ensure different heading levels use different font sizes
3. **Data Quantity**: More training PDFs generally improve accuracy
4. **Consistent Format**: Similar document types work better than mixed formats

## Model Performance

The Random Forest classifier typically achieves:
- **Accuracy**: 75-90% depending on data quality
- **Precision**: Higher for common heading levels (title, H1)
- **Recall**: Varies by heading level frequency

Performance metrics are displayed during training, showing precision, recall, and frequency for each heading level.

## Workflow Summary

### Local Installation
1. **Setup**: Install dependencies, organize PDF files
2. **Extract**: Run `python process_pdfs.py` to extract training data
3. **Prepare**: Run `python ml/prepare_labeling_csv.py` to clean data
4. **Train**: Run `python ml/train_model.py` to train the classifier
5. **Predict**: Run `python ml/predict_headings.py` on new PDFs
6. **Use**: JSON outlines are ready for further processing

### Docker Installation (Recommended)
1. **Setup**: Build Docker image and start container
   ```bash
   run.bat build && run.bat start    # Windows
   ./run.sh build && ./run.sh start  # Linux/Mac
   ```
2. **Extract**: Run `run.bat process` to extract training data
3. **Prepare**: Run `run.bat prepare` to clean data
4. **Train**: Run `run.bat train` to train the classifier
5. **Predict**: Run `run.bat predict` on new PDFs
6. **Use**: JSON outlines are ready for further processing

This workflow creates a complete pipeline from raw PDFs to structured heading outlines using machine learning. 
This workflow creates a complete pipeline from raw PDFs to structured heading outlines using machine learning. 