# Text Processing and Classification Toolkit

A comprehensive toolkit for text processing, organization, and multi-label classification using transformer models.

## Overview

This repository contains three main components for text processing and classification:

1. **Text Translation Tool** - Translates text files from Bulgarian to English using OpenAI GPT
2. **Text Data Organizer** - Organizes text files into hierarchical classification structures
3. **Multi-Label Text Classifier** - Trains transformer models for multi-label text classification

## Project Structure

```
├── translator.py                 # Bulgarian to English text translation
├── text_organizer.py            # Text data organization and classification
├── multilabel_classifier.py     # Multi-label text classification training
├── text_classifier.py           # Hierarchical text classification system
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## Features

### 🌐 Text Translation (`translator.py`)
- Translates long Bulgarian documents to English using OpenAI GPT
- Handles oversized documents by processing them in segments
- Comprehensive error handling and retry logic
- Batch processing of multiple files

### 📁 Text Data Organization (`text_organizer.py`)
- Processes TSV annotation files with hierarchical labels
- Organizes text files into category-based directory structures
- Generates label files for each top-level category
- Statistics tracking and comprehensive logging

### 🧠 Multi-Label Classification (`multilabel_classifier.py`)
- PyTorch Lightning implementation for multi-label text classification
- Support for any HuggingFace transformer model
- Comprehensive training pipeline with checkpointing
- CSV and TensorBoard logging

### 🎯 Hierarchical Classification (`text_classifier.py`)
- Trains separate models for each top-level category
- Automatic memory management and cleanup
- Support for prediction with confidence scores
- Configurable training parameters

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text-processing-toolkit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (for translation):
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

### Text Translation
```python
from translator import TextTranslator

# Initialize translator
translator = TextTranslator()

# Set custom folders
translator.set_folders("input/", "output/", "failed/")

# Run translation
translator.run()
```

### Text Organization
```python
from text_organizer import TextDataOrganizer

# Initialize organizer
organizer = TextDataOrganizer(
    annotation_file="annotations.txt",
    text_dir="documents/",
    output_root="classified_data/"
)

# Organize data
organizer.organize_data()
```

### Multi-Label Classification
```python
from multilabel_classifier import MultiLabelTextClassifier

# Initialize classifier
classifier = MultiLabelTextClassifier(
    raw_documents_dir='documents/',
    annotations_file='annotations.txt'
)

# Run training
classifier.run_training_pipeline()
```

### Hierarchical Classification
```python
from text_classifier import TextClassificationTrainer

# Initialize trainer
trainer = TextClassificationTrainer()

# Train all categories
trainer.train_all_categories("classified_by_top_label")
```

## Data Formats

### Translation Input
- **Input**: Directory containing `.txt` files in Bulgarian
- **Output**: Translated English `.txt` files

### Organization Input
- **Annotation File**: TSV format with columns:
  ```
  filename.txt    top_label1;top_label2    sub_label1;sub_label2
  ```
- **Text Directory**: Directory containing text files referenced in annotations

### Classification Input
- **Documents**: Text files in specified directory
- **Annotations**: TSV file with article IDs and semicolon-separated labels

## Configuration

### Environment Variables
```bash
# Required for translation
export OPENAI_API_KEY='your-openai-api-key'

# Optional: Custom model cache directory
export TRANSFORMERS_CACHE='/path/to/cache'
```

### Model Configuration
All components support customizable parameters:

- **Model Selection**: Any HuggingFace transformer model
- **Training Parameters**: Learning rate, batch size, epochs
- **Processing Options**: Max length, threshold values
- **Output Paths**: Customizable save locations

## Output Structure

### Organized Data Structure
```
classified_by_top_label/
├── Category1/
│   ├── article1.txt
│   ├── article2.txt
│   └── Category1_labels.txt
├── Category2/
│   ├── article3.txt
│   └── Category2_labels.txt
└── ...
```

### Model Outputs
```
├── trained_model.pth         # Model weights
├── label_binarizer.pkl       # Label encoder
├── tokenizer/               # Tokenizer configuration
├── lightning_logs/          # Training logs
└── best-checkpoint.ckpt     # Best model checkpoint
```

## Advanced Usage

### Custom Model Training
```python
# Use custom transformer model
classifier = MultiLabelTextClassifier(
    model_name="roberta-base",
    max_length=256,
    batch_size=16,
    max_epochs=50
)
```

### Batch Processing
```python
# Process multiple translation jobs
translator = TextTranslator()
for input_dir in ["batch1/", "batch2/", "batch3/"]:
    translator.set_folders(input_dir, f"output_{input_dir}", f"failed_{input_dir}")
    translator.run()
```

### Model Prediction
```python
from text_classifier import TextClassificationPredictor

# Load trained model for prediction
predictor = TextClassificationPredictor("path/to/model")
label, confidence = predictor.predict("Your text here")
print(f"Predicted: {label} (confidence: {confidence:.3f})")
```

## Performance Considerations

- **GPU Support**: Automatic GPU detection and utilization
- **Memory Management**: Efficient cleanup and garbage collection
- **Batch Processing**: Optimized data loading and processing
- **Checkpointing**: Resume training from interruptions

## Logging and Monitoring

- **Translation**: File-level progress and error tracking
- **Organization**: Statistics on processed files and categories
- **Training**: Loss curves, metrics, and model checkpoints
- **Classification**: Confidence scores and prediction logs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **API Rate Limits**: Implement delays in translation calls
3. **File Encoding Issues**: Ensure UTF-8 encoding for all text files
4. **Missing Dependencies**: Install all requirements from `requirements.txt`

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers for model architectures
- PyTorch Lightning for training framework
- OpenAI for translation capabilities
- scikit-learn for preprocessing utilities

## Support

For questions and support:
- Check the documentation above
- Review the code comments and docstrings
- Create an issue in the repository
