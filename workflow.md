# PDF Heading Extractor - Complete Workflow Documentation

## Project Overview

The **PDF Heading Extractor** is a machine learning-based tool that automatically extracts and classifies headings from PDF documents. This project uses font size analysis and machine learning to identify different heading levels (Title, H1, H2, H3, H4) in PDF documents, converting unstructured PDF content into structured outlines.

### Key Features
- **Automatic Heading Detection**: Uses font size hierarchy to identify heading levels
- **Machine Learning Classification**: Random Forest model for accurate heading classification
- **Structured Output**: Generates JSON outlines with hierarchical heading structure
- **Docker Support**: Containerized solution for consistent deployment
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Project Architecture

### Technology Stack
- **Python 3.9**: Core programming language
- **PyMuPDF (fitz)**: PDF text extraction and analysis
- **scikit-learn**: Machine learning framework (Random Forest)
- **pandas**: Data manipulation and processing
- **Docker**: Containerization for deployment
- **joblib**: Model serialization and persistence

### Project Structure
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
└── README.md                  # Project documentation
```

## Detailed Workflow

### Phase 1: Data Extraction and Processing

#### Step 1: PDF Text Extraction (`process_pdfs.py`)

**Purpose**: Extract text, font metadata, and position information from training PDFs

**Input**: PDF files placed in `dataset/pdfs/` directory

**Process**:
1. **PDF Parsing**: Uses PyMuPDF to open and parse each PDF file
2. **Text Extraction**: Extracts text spans with their metadata:
   - Text content
   - Font size (in points)
   - Page number
   - X and Y coordinates
   - Source PDF filename
3. **Heading Detection**: Automatically detects heading levels based on font size hierarchy:
   - Largest font size → "title"
   - Second largest → "H1"
   - Third largest → "H2"
   - And so on...
4. **Filtering**: Removes body text, keeping only potential headings

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

### Phase 2: Data Preparation

#### Step 2: Data Cleaning (`ml/prepare_labeling_csv.py`)

**Purpose**: Prepare extracted data for machine learning training

**Input**: `dataset/outputs/combined_dataset.csv` from Step 1

**Process**:
1. **Level Filtering**: Filters data to include only allowed heading levels:
   - `title`, `H1`, `H2`, `H3`, `H4`
2. **Column Selection**: Keeps relevant columns for training
3. **Label Assignment**: Sets the `label` column equal to the detected `level`

**Output**: `dataset/outputs/sample_labeled.csv`
- Columns: `text`, `level`, `font_size`, `page`, `x_position`, `y_position`, `source_pdf`, `label`
- Ready for machine learning model training

### Phase 3: Machine Learning Model Training

#### Step 3: Feature Engineering and Model Training (`ml/train_model.py`)

**Purpose**: Train a Random Forest classifier to predict heading levels

**Input**: `dataset/outputs/sample_labeled.csv` from Step 2

**Process**:

**Feature Extraction**:
1. **Text Features**:
   - `text_length`: Number of characters in the text
   - `word_count`: Number of words in the text
   - `is_uppercase`: Binary flag (1 if text is all uppercase)
   - `is_titlecase`: Binary flag (1 if text is title case)
2. **Positional Features**:
   - `font_size`: Font size in points
   - `x_position`: X-coordinate on the page
   - `y_position`: Y-coordinate on the page
   - `page`: Page number

**Model Training**:
1. **Data Split**: 80% training, 20% validation
2. **Label Encoding**: Converts text labels to numerical values
3. **Random Forest Training**: 
   - 100 estimators
   - Maximum depth of 10
   - Random state for reproducibility
4. **Model Evaluation**: Calculates precision, recall, and accuracy for each heading level
5. **Model Persistence**: Saves trained model and label encoder to `ml/model.pkl`

**Output**: `ml/model.pkl`
- Contains the trained Random Forest model and LabelEncoder
- Used for making predictions on new PDFs

**Sample Performance Metrics**:
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

### Phase 4: Prediction and Output Generation

#### Step 4: Heading Prediction (`ml/predict_headings.py`)

**Purpose**: Apply the trained model to new PDFs and generate structured outlines

**Input**: 
- Trained model from `ml/model.pkl`
- New PDF files in `dataset/new_pdfs/` directory

**Process**:
1. **Model Loading**: Loads the trained Random Forest model and LabelEncoder
2. **PDF Processing**: For each PDF in the new_pdfs directory:
   - Extracts text and metadata (same process as training)
   - Applies feature engineering (same features as training)
   - Uses the trained model to predict heading levels
3. **Output Generation**: Creates structured JSON outlines for each PDF

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

## Deployment Options

### Option 1: Local Installation

**Prerequisites**:
- Python 3.7 or higher
- pip (Python package installer)

**Setup**:
```bash
# Install dependencies
pip install -r requirements.txt
```

**Execution**:
```bash
# Step 1: Extract data from training PDFs
python process_pdfs.py

# Step 2: Prepare data for training
python ml/prepare_labeling_csv.py

# Step 3: Train the model
python ml/train_model.py

# Step 4: Predict on new PDFs
python ml/predict_headings.py
```

### Option 2: Docker Installation (Recommended)

**Prerequisites**:
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose

**Setup and Execution**:

**Windows Users**:
```bash
# Build and start
run.bat build
run.bat start

# Run complete pipeline
run.bat process  # Extract data from PDFs
run.bat prepare  # Prepare data for training
run.bat train    # Train the model
run.bat predict  # Predict on new PDFs
```

**Linux/Mac Users**:
```bash
# Build and start
./run.sh build
./run.sh start

# Run complete pipeline
./run.sh process  # Extract data from PDFs
./run.sh prepare  # Prepare data for training
./run.sh train    # Train the model
./run.sh predict  # Predict on new PDFs
```

## Data Flow and File Dependencies

### Input Files
1. **Training PDFs** (`dataset/pdfs/`):
   - Format: PDF files
   - Content: Documents with clear heading hierarchy
   - Requirements: Font sizes should vary between heading levels
   - Recommended: 5-20 PDFs for good training

2. **New PDFs** (`dataset/new_pdfs/`):
   - Format: PDF files
   - Content: Any PDF document you want to extract headings from
   - Should have similar structure to training PDFs for best results

### Intermediate Files
1. **`combined_dataset.csv`**: Raw extracted data from training PDFs
2. **`labeling_ready_dataset.csv`**: Cleaned data ready for labeling
3. **`sample_labeled.csv`**: Final training dataset with labels

### Output Files
1. **`model.pkl`**: Trained machine learning model
2. **`*.json`**: Structured outlines for each processed PDF

### File Dependencies
```
dataset/pdfs/*.pdf
    ↓
process_pdfs.py
    ↓
dataset/outputs/combined_dataset.csv
    ↓
ml/prepare_labeling_csv.py
    ↓
dataset/outputs/sample_labeled.csv
    ↓
ml/train_model.py
    ↓
ml/model.pkl
    ↓
ml/predict_headings.py (with dataset/new_pdfs/*.pdf)
    ↓
dataset/outputs/*.json
```

## Technical Implementation Details

### PDF Processing Algorithm

**Text Extraction Process**:
1. **Document Parsing**: Uses PyMuPDF's `fitz.open()` to parse PDF structure
2. **Block Analysis**: Extracts text blocks with their formatting information
3. **Span Processing**: Processes individual text spans to get:
   - Text content
   - Font properties (size, family)
   - Positional coordinates
   - Page information

**Heading Detection Algorithm**:
1. **Font Size Analysis**: Counts frequency of each font size
2. **Size Ranking**: Sorts font sizes in descending order
3. **Level Assignment**: Maps font sizes to heading levels:
   - Largest font → "title"
   - Second largest → "H1"
   - Third largest → "H2"
   - etc.
4. **Body Text Filtering**: Removes text with smaller font sizes

### Machine Learning Model

**Random Forest Classifier**:
- **Algorithm**: Ensemble learning method using multiple decision trees
- **Parameters**:
  - `n_estimators=100`: Number of trees in the forest
  - `max_depth=10`: Maximum depth of each tree
  - `random_state=42`: For reproducible results

**Feature Engineering**:
- **Text Features**: Length, word count, case analysis
- **Positional Features**: Font size, coordinates, page number
- **Feature Scaling**: No scaling required for Random Forest

**Model Performance**:
- **Accuracy**: Typically 75-90% depending on data quality
- **Precision**: Higher for common heading levels (title, H1)
- **Recall**: Varies by heading level frequency

### Docker Configuration

**Container Setup**:
- **Base Image**: Python 3.9 slim
- **System Dependencies**: build-essential for PyMuPDF compilation
- **Python Dependencies**: Installed from requirements.txt
- **Volume Mounts**: 
  - `./dataset/pdfs:/app/dataset/pdfs`
  - `./dataset/new_pdfs:/app/dataset/new_pdfs`
  - `./dataset/outputs:/app/dataset/outputs`
  - `./ml:/app/ml`

**Management Scripts**:
- **Windows**: `run.bat` with commands for build, start, stop, process, etc.
- **Linux/Mac**: `run.sh` with equivalent functionality

## Use Cases and Applications

### Primary Use Cases
1. **Document Structure Analysis**: Extract hierarchical structure from PDFs
2. **Content Indexing**: Create automatic table of contents
3. **Document Classification**: Categorize documents based on heading structure
4. **Content Extraction**: Extract specific sections based on heading levels
5. **Document Comparison**: Compare structure across multiple documents

### Industry Applications
- **Legal Documents**: Extract section headings from contracts and legal papers
- **Academic Papers**: Create structured outlines from research papers
- **Technical Manuals**: Extract chapter and section headings
- **Business Reports**: Analyze document structure and organization
- **Government Documents**: Process forms and official documents

## Performance Considerations

### Training Data Quality
- **Font Size Variation**: Ensure different heading levels use different font sizes
- **Consistent Formatting**: Similar document types work better than mixed formats
- **Data Quantity**: More training PDFs generally improve accuracy
- **Label Consistency**: Manual review of auto-detected labels may improve results

### Model Performance Optimization
- **Feature Engineering**: Additional features like text patterns, special characters
- **Hyperparameter Tuning**: Grid search for optimal Random Forest parameters
- **Ensemble Methods**: Combining multiple models for better accuracy
- **Data Augmentation**: Creating synthetic training data

### Scalability
- **Batch Processing**: Process multiple PDFs simultaneously
- **Parallel Processing**: Use multiple CPU cores for feature extraction
- **Memory Management**: Handle large PDF files efficiently
- **Caching**: Cache extracted features for repeated processing

## Troubleshooting and Common Issues

### PDF Processing Issues
- **No Text Extracted**: PDF may be scanned images without OCR
- **Poor Heading Detection**: Font sizes may be too similar
- **Missing Content**: PDF structure may be complex or non-standard

### Model Performance Issues
- **Low Accuracy**: Insufficient or poor quality training data
- **Overfitting**: Model performs well on training but poorly on new data
- **Class Imbalance**: Some heading levels may be underrepresented

### Docker Issues
- **Permission Errors**: File permission issues on mounted volumes
- **Resource Constraints**: Insufficient memory or CPU allocation
- **Network Issues**: Problems with Docker networking

## Future Enhancements

### Potential Improvements
1. **Advanced NLP Features**: Use word embeddings and semantic analysis
2. **Layout Analysis**: Consider document layout and positioning
3. **Multi-language Support**: Extend to non-English documents
4. **Web Interface**: Add GUI for easier interaction
5. **API Integration**: REST API for programmatic access
6. **Real-time Processing**: Stream processing for large document sets

### Model Enhancements
1. **Deep Learning**: Neural networks for better feature learning
2. **Transfer Learning**: Pre-trained models for document understanding
3. **Active Learning**: Interactive model improvement
4. **Ensemble Methods**: Multiple model combination

## Conclusion

The PDF Heading Extractor provides a complete end-to-end solution for automatically extracting and classifying headings from PDF documents. The workflow combines traditional PDF processing techniques with modern machine learning to create a robust and scalable system.

The project demonstrates best practices in:
- **Modular Design**: Separate components for different stages
- **Containerization**: Docker for consistent deployment
- **Data Pipeline**: Clear data flow between components
- **Documentation**: Comprehensive setup and usage instructions
- **Error Handling**: Robust processing with fallback options

This system can be easily extended and customized for specific use cases while maintaining the core functionality of automatic heading extraction and classification.
