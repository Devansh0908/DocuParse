# PDF Heading Extractor - File Descriptions

This document provides a comprehensive description of each file in the PDF Heading Extractor project, including their purpose, functionality, code explanation, and technical details.

## Project Overview

The PDF Heading Extractor is a machine learning-based tool that automatically extracts and classifies headings from PDF documents. It uses font size analysis and machine learning to identify different heading levels (Title, H1, H2, H3, H4) and converts unstructured PDF content into structured outlines.

## Core Python Scripts

### 1. `process_pdfs.py` - PDF Text Extraction Engine

**Purpose**: Main script for extracting text, font metadata, and position information from training PDFs.

**Key Functions**:
- `extract_text_from_pdf(pdf_path, source_pdf)`: Extracts text spans with metadata from a single PDF
- `detect_heading_levels(df, max_heading_levels=5)`: Automatically detects heading levels based on font size hierarchy
- `process_all_pdfs()`: Orchestrates the processing of all PDFs in the dataset

**Technical Details**:
- Uses PyMuPDF (fitz) for PDF parsing and text extraction
- Extracts text spans with font size, position, and page information
- Implements font size-based heading detection algorithm
- Filters out body text, keeping only potential headings
- Outputs structured CSV data for machine learning training

**Input**: PDF files in `dataset/pdfs/` directory
**Output**: `dataset/outputs/combined_dataset.csv` with columns: text, level, font_size, page, x_position, y_position, source_pdf

**Code Flow**:
1. Iterates through all PDF files in the input directory
2. For each PDF, extracts text blocks and their formatting metadata
3. Analyzes font size distribution to determine heading hierarchy
4. Maps font sizes to heading levels (title, H1, H2, H3, H4)
5. Filters out body text and saves results to CSV

### 2. `ml/prepare_labeling_csv.py` - Data Preparation Script

**Purpose**: Prepares extracted data for machine learning training by cleaning and filtering the dataset.

**Key Functions**:
- `main()`: Orchestrates the data preparation process

**Technical Details**:
- Filters data to include only allowed heading levels: title, H1, H2, H3, H4
- Selects relevant columns for training
- Sets the label column equal to the detected level for supervised learning
- Creates a clean, labeled dataset ready for model training

**Input**: `dataset/outputs/labeling_ready_dataset.csv`
**Output**: `dataset/outputs/sample_labeled.csv`

**Code Flow**:
1. Reads the combined dataset CSV
2. Filters rows to include only valid heading levels
3. Selects relevant columns for training
4. Adds label column for supervised learning
5. Saves cleaned dataset

### 3. `ml/train_model.py` - Machine Learning Model Trainer

**Purpose**: Trains a Random Forest classifier to predict heading levels based on extracted features.

**Key Functions**:
- `extract_features(df)`: Extracts numerical features from text data
- `train_model()`: Orchestrates the complete training pipeline

**Technical Details**:
- **Feature Engineering**:
  - `text_length`: Number of characters in text
  - `word_count`: Number of words in text
  - `is_uppercase`: Binary flag for all uppercase text
  - `is_titlecase`: Binary flag for title case text
  - `font_size`, `x_position`, `y_position`, `page`: Original metadata
- **Model**: Random Forest Classifier with 100 estimators, max depth 10
- **Data Split**: 80% training, 20% validation
- **Evaluation**: Calculates precision, recall, and accuracy for each heading level
- **Persistence**: Saves trained model and label encoder using joblib

**Input**: `dataset/outputs/sample_labeled.csv`
**Output**: `ml/model.pkl` (trained model and label encoder)

**Code Flow**:
1. Loads labeled training data
2. Extracts numerical features from text and metadata
3. Encodes text labels to numerical values
4. Splits data into training and validation sets
5. Trains Random Forest classifier
6. Evaluates model performance and displays metrics
7. Saves trained model and encoder

### 4. `ml/predict_headings.py` - Prediction and Output Generator

**Purpose**: Applies the trained model to new PDFs and generates structured JSON outlines.

**Key Functions**:
- `extract_features(df)`: Same feature extraction as training
- `extract_text_from_pdf(pdf_path)`: Extracts text and metadata from PDF
- `predict_pdf(pdf_path, model, le)`: Predicts heading levels for a PDF
- `pdf_to_json(pdf_path, df)`: Converts predictions to structured JSON
- `main()`: Orchestrates the prediction pipeline

**Technical Details**:
- Uses the same feature extraction process as training for consistency
- Applies trained Random Forest model to predict heading levels
- Filters predictions to include only valid heading levels
- Generates structured JSON outlines with hierarchical organization
- Creates one JSON file per PDF with title and outline structure

**Input**: 
- Trained model from `ml/model.pkl`
- New PDF files in `dataset/new_pdfs/` directory

**Output**: JSON files in `dataset/outputs/` directory (one per PDF)

**Code Flow**:
1. Loads trained model and label encoder
2. Iterates through PDFs in new_pdfs directory
3. Extracts text and metadata from each PDF
4. Applies feature engineering
5. Uses trained model to predict heading levels
6. Filters predictions to valid heading levels
7. Generates structured JSON outline
8. Saves JSON file for each PDF

## Configuration and Dependency Files

### 5. `requirements.txt` - Python Dependencies

**Purpose**: Defines all Python package dependencies required for the project.

**Dependencies**:
- `pandas>=1.3.0`: Data manipulation and analysis
- `scikit-learn>=1.0.0`: Machine learning framework
- `joblib>=1.1.0`: Model serialization and persistence
- `PyMuPDF>=1.20.0`: PDF text extraction and analysis
- `numpy>=1.21.0`: Numerical computing (dependency of scikit-learn)

**Usage**: Install with `pip install -r requirements.txt`

### 6. `Dockerfile` - Container Configuration

**Purpose**: Defines the Docker container environment for consistent deployment.

**Technical Details**:
- **Base Image**: Python 3.9 slim for lightweight container
- **System Dependencies**: build-essential for PyMuPDF compilation
- **Python Dependencies**: Installed from requirements.txt
- **Directory Structure**: Creates necessary directories for data and models
- **Environment Variables**: Sets Python path and unbuffered output
- **Default Command**: Runs PDF processing script

**Key Sections**:
1. **Base Setup**: Python 3.9 slim image with build tools
2. **Dependency Installation**: System and Python packages
3. **Project Setup**: Copy project files and create directories
4. **Environment Configuration**: Set Python environment variables
5. **Default Behavior**: Run PDF processing on container start

### 7. `docker-compose.yml` - Container Orchestration

**Purpose**: Defines the Docker Compose configuration for easy container management.

**Technical Details**:
- **Service Name**: pdf-heading-extractor
- **Volume Mounts**: 
  - `./dataset/pdfs:/app/dataset/pdfs` - Training PDFs
  - `./dataset/new_pdfs:/app/dataset/new_pdfs` - New PDFs for prediction
  - `./dataset/outputs:/app/dataset/outputs` - Output files
  - `./ml:/app/ml` - Model files
- **Environment**: Sets Python unbuffered output
- **Default Command**: Keeps container running with tail command

**Benefits**:
- Persistent data storage through volume mounts
- Easy access to input/output directories
- Consistent environment across different systems

### 8. `.dockerignore` - Docker Build Exclusions

**Purpose**: Specifies files and directories to exclude from Docker build context.

**Excluded Items**:
- Git files (.git, .gitignore)
- Python cache files (__pycache__, *.pyc)
- Virtual environments (venv/, env/)
- IDE files (.vscode/, .idea/)
- OS files (.DS_Store, Thumbs.db)
- Logs and temporary files
- Documentation (README.md)
- Docker files (Dockerfile, .dockerignore)
- Large data files (*.csv, *.pkl, outputs/, model.pkl)

**Benefits**:
- Faster Docker builds by excluding unnecessary files
- Smaller build context
- Prevents sensitive or large files from being included

## Management Scripts

### 9. `run.bat` - Windows Batch Runner Script

**Purpose**: Provides easy command-line interface for Windows users to manage the Docker container and run project operations.

**Available Commands**:
- `build`: Build Docker image
- `start`: Start container
- `stop`: Stop container
- `restart`: Restart container
- `logs`: Show container logs
- `shell`: Open shell in running container
- `process`: Run PDF processing
- `prepare`: Run data preparation
- `train`: Train model
- `predict`: Run predictions on new PDFs
- `help`: Show usage information

**Technical Details**:
- Uses Windows batch script syntax
- Implements command routing with goto statements
- Provides clear error messages and usage instructions
- Executes Docker commands through docker-compose
- Supports interactive operations (shell, logs)

**Usage Examples**:
```batch
run.bat build      # Build the Docker image
run.bat start      # Start the container
run.bat process    # Run PDF processing
run.bat shell      # Open interactive shell
```

### 10. `run.sh` - Linux/Mac Shell Runner Script

**Purpose**: Provides easy command-line interface for Linux/Mac users to manage the Docker container and run project operations.

**Available Commands**: Same as run.bat but for Unix-like systems

**Technical Details**:
- Uses bash script syntax with shebang (`#!/bin/bash`)
- Implements function-based command routing
- Provides clear error messages and usage instructions
- Executes Docker commands through docker-compose
- Supports interactive operations (shell, logs)

**Key Functions**:
- `show_usage()`: Displays help information
- `build_image()`: Builds Docker image
- `start_container()`: Starts container
- `stop_container()`: Stops container
- `open_shell()`: Opens interactive shell
- `run_process()`, `run_prepare()`, `run_train()`, `run_predict()`: Run specific operations

**Usage Examples**:
```bash
./run.sh build     # Build the Docker image
./run.sh start     # Start the container
./run.sh process   # Run PDF processing
./run.sh shell     # Open interactive shell
```

## Documentation Files

### 11. `README.md` - Project Documentation

**Purpose**: Comprehensive project documentation with setup instructions, usage examples, and troubleshooting guide.

**Key Sections**:
- **Overview**: Project description and key features
- **Prerequisites**: System requirements for local and Docker installation
- **Installation**: Step-by-step setup instructions
- **Docker Setup**: Detailed Docker configuration and usage
- **Project Structure**: File organization and purpose
- **Step-by-Step Workflow**: Complete pipeline explanation
- **Input/Output Specifications**: Data format requirements
- **Troubleshooting**: Common issues and solutions
- **Performance**: Model performance expectations

**Technical Details**:
- 444 lines of comprehensive documentation
- Includes code examples and command references
- Provides both local and Docker installation paths
- Contains troubleshooting section for common issues
- Shows expected output formats and performance metrics

### 12. `workflow.md` - Detailed Workflow Documentation

**Purpose**: In-depth technical documentation of the complete workflow and architecture.

**Key Sections**:
- **Project Overview**: Detailed project description and features
- **Project Architecture**: Technology stack and structure
- **Detailed Workflow**: Step-by-step technical process
- **Deployment Options**: Local vs Docker installation
- **Data Flow**: File dependencies and relationships
- **Technical Implementation**: Algorithm details and model specifications
- **Use Cases**: Applications and industry examples
- **Performance Considerations**: Optimization and scalability
- **Troubleshooting**: Technical issues and solutions
- **Future Enhancements**: Potential improvements and extensions

**Technical Details**:
- 450 lines of technical documentation
- Includes algorithm explanations and code flow
- Provides detailed architecture information
- Contains performance optimization guidelines
- Shows data flow diagrams and file relationships

## Data Directory Structure

### 13. `dataset/` - Data Organization Directory

**Purpose**: Organizes all input and output data files in a structured manner.

**Subdirectories**:
- **`pdfs/`**: Training PDF files (input for model training)
- **`new_pdfs/`**: New PDF files for prediction (input for inference)
- **`outputs/`**: All generated output files (CSV data, JSON outlines)

**Data Flow**:
```
dataset/pdfs/*.pdf → process_pdfs.py → dataset/outputs/combined_dataset.csv
dataset/outputs/combined_dataset.csv → prepare_labeling_csv.py → dataset/outputs/sample_labeled.csv
dataset/outputs/sample_labeled.csv → train_model.py → ml/model.pkl
dataset/new_pdfs/*.pdf + ml/model.pkl → predict_headings.py → dataset/outputs/*.json
```

### 14. `ml/` - Machine Learning Directory

**Purpose**: Contains all machine learning related files and trained models.

**Files**:
- **`model.pkl`**: Trained Random Forest model and LabelEncoder (507KB)
- **`train_model.py`**: Model training script
- **`predict_headings.py`**: Prediction script
- **`prepare_labeling_csv.py`**: Data preparation script

**Model Details**:
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, max depth 10, random state 42
- **Features**: 8 numerical features (text length, word count, case flags, position, font size)
- **Classes**: 5 heading levels (title, H1, H2, H3, H4)
- **Performance**: Typically 75-90% accuracy depending on data quality

## File Dependencies and Data Flow

### Complete Pipeline Flow

```
1. Input PDFs (dataset/pdfs/*.pdf)
   ↓
2. process_pdfs.py
   ↓
3. combined_dataset.csv (dataset/outputs/)
   ↓
4. prepare_labeling_csv.py
   ↓
5. sample_labeled.csv (dataset/outputs/)
   ↓
6. train_model.py
   ↓
7. model.pkl (ml/)
   ↓
8. predict_headings.py + new PDFs (dataset/new_pdfs/*.pdf)
   ↓
9. JSON outlines (dataset/outputs/*.json)
```

### Key Dependencies

- **PyMuPDF**: Required for PDF text extraction in `process_pdfs.py` and `predict_headings.py`
- **scikit-learn**: Required for machine learning in `train_model.py`
- **pandas**: Required for data manipulation in all scripts
- **joblib**: Required for model persistence in `train_model.py` and `predict_headings.py`
- **Docker**: Required for containerized deployment (Dockerfile, docker-compose.yml)

## Technical Architecture Summary

### Core Components

1. **PDF Processing Engine** (`process_pdfs.py`): Extracts text and metadata from PDFs
2. **Data Pipeline** (`prepare_labeling_csv.py`): Prepares data for machine learning
3. **ML Training** (`train_model.py`): Trains Random Forest classifier
4. **Prediction Engine** (`predict_headings.py`): Applies model to new PDFs
5. **Containerization** (Dockerfile, docker-compose.yml): Ensures consistent deployment
6. **Management Scripts** (run.bat, run.sh): Simplifies operations

### Technology Stack

- **Language**: Python 3.9
- **PDF Processing**: PyMuPDF (fitz)
- **Machine Learning**: scikit-learn (Random Forest)
- **Data Processing**: pandas
- **Model Persistence**: joblib
- **Containerization**: Docker & Docker Compose
- **Cross-Platform**: Windows (batch) and Unix (bash) scripts

### Key Features

- **Automatic Heading Detection**: Font size-based hierarchy detection
- **Machine Learning Classification**: Random Forest for accurate predictions
- **Structured Output**: JSON outlines with hierarchical organization
- **Docker Support**: Containerized for consistent deployment
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Modular Design**: Separate components for different stages
- **Comprehensive Documentation**: Detailed setup and usage instructions

This project demonstrates a complete end-to-end machine learning pipeline for document structure analysis, combining traditional PDF processing techniques with modern machine learning approaches to create a robust and scalable solution for automatic heading extraction and classification.
