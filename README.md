# Sentiment Classification

A sentiment analysis project demonstrating both traditional dictionary-based and modern transformer-based approaches for classifying review sentiments.

## Overview

This repository contains Jupyter notebooks for performing sentiment analysis on review data using different methodologies:

1. **Data Loading** (`01_Load_Data.ipynb`) - Loads and prepares review data
2. **Dictionary-Based Approach** (`02_Create_Dictionary_Based_Sentiment_Analyzer.ipynb`) - Traditional sentiment analysis using NLTK opinion lexicon
3. **Transformer-Based Approach** (`03_Transformer_Based_Sentiment_Analyzer.ipynb`) - **NEW!** State-of-the-art sentiment analysis using transformer models

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Transformer-Based Sentiment Analysis

The latest addition to this project uses state-of-the-art transformer models from Hugging Face's `transformers` library. 

### Key Features:

- **Latest Models**: Uses DistilBERT fine-tuned on SST-2, with options for RoBERTa and other cutting-edge models
- **Better Accuracy**: Significantly improved performance over dictionary-based methods
- **Contextual Understanding**: Handles negations, sarcasm, and complex sentiment expressions
- **GPU Support**: Automatically leverages GPU if available for faster processing
- **Easy to Extend**: Simple pipeline interface makes it easy to experiment with different models

### Supported Models:

The notebook supports multiple state-of-the-art models:
- **DistilBERT** (default): `distilbert-base-uncased-finetuned-sst-2-english` - Fast and efficient
- **RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - Latest model, great for social media
- **BERT**: `bert-base-uncased` - Original transformer model
- **Multilingual**: `nlptown/bert-base-multilingual-uncased-sentiment` - For non-English text

### Usage:

1. Open `03_Transformer_Based_Sentiment_Analyzer.ipynb`
2. Run the cells to perform sentiment analysis using transformers
3. Compare results with the dictionary-based approach
4. Experiment with different models by uncommenting alternative model sections

## Comparison: Dictionary vs Transformer

| Aspect | Dictionary-Based | Transformer-Based |
|--------|-----------------|-------------------|
| Accuracy | Moderate | High |
| Speed | Fast | Moderate (faster with GPU) |
| Context Understanding | Limited | Excellent |
| Negation Handling | Poor | Excellent |
| Setup Complexity | Simple | Moderate |
| Model Size | Minimal | Large (hundreds of MB) |

## Data Structure

- `DATA/raw/` - Original review data
- `DATA/processed/` - Processed results with sentiment scores
- `plots/` - Visualization outputs

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order:
   - Start with `01_Load_Data.ipynb` to prepare data
   - Run `02_Create_Dictionary_Based_Sentiment_Analyzer.ipynb` for baseline
   - Run `03_Transformer_Based_Sentiment_Analyzer.ipynb` for state-of-the-art results

## Notes

- The transformer models will be downloaded automatically on first run (~250MB for DistilBERT)
- GPU is recommended for processing large datasets but not required
- Results are saved in TSV format for easy comparison

## Contributing

Feel free to experiment with different transformer models and contribute improvements!
