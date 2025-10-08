#!/usr/bin/env python3
"""
Transformer-Based Sentiment Analysis Script

This script provides a command-line interface for sentiment analysis using
state-of-the-art transformer models from Hugging Face.

Usage:
    python transformer_sentiment_analyzer.py --input DATA/raw/review_corpus.tsv --output DATA/processed/transformer_sentiment.tsv

Requirements:
    - transformers
    - torch
    - pandas
    - numpy
"""

import argparse
import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')


def get_transformer_sentiment(review, sentiment_pipeline, max_length=512):
    """
    Get sentiment score using transformer model.
    Returns a score between -1 and 1.
    
    Args:
        review (str): The review text to analyze
        sentiment_pipeline: The transformer pipeline
        max_length (int): Maximum length of text to process
        
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    # Truncate review to max length to avoid errors
    truncated_review = review[:max_length]
    
    try:
        result = sentiment_pipeline(truncated_review)[0]
        
        # Convert to numerical score: POSITIVE=1, NEGATIVE=-1, weighted by confidence
        if result['label'] == 'POSITIVE':
            return result['score']
        else:  # NEGATIVE
            return -result['score']
    except Exception as e:
        print(f"Error processing review: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Perform sentiment analysis using transformer models'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='DATA/raw/review_corpus.tsv',
        help='Input TSV file with reviews (default: DATA/raw/review_corpus.tsv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='DATA/processed/transformer_based_sentiment.tsv',
        help='Output TSV file for results (default: DATA/processed/transformer_based_sentiment.tsv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='distilbert-base-uncased-finetuned-sst-2-english',
        help='Transformer model to use (default: distilbert-base-uncased-finetuned-sst-2-english)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Transformer-Based Sentiment Analysis")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    df = pd.read_csv(args.input, sep="\t")
    print(f"Loaded {len(df)} reviews")
    
    ratings = list(df['rating'])
    reviews = list(df['review'])
    
    # Initialize the sentiment analysis pipeline
    print(f"\nInitializing transformer model: {args.model}")
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=args.model,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Process all reviews
    print(f"\nProcessing {len(reviews)} reviews...")
    transformer_sentiments = []
    
    for i, review in enumerate(reviews):
        sentiment = get_transformer_sentiment(review, sentiment_pipeline)
        transformer_sentiments.append(sentiment)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(reviews)} reviews")
    
    print("Processing complete!")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'rating': ratings,
        'review': reviews,
        'transformer_sentiment': transformer_sentiments
    })
    
    # Display statistics
    print("\n" + "=" * 80)
    print("Sentiment Statistics:")
    print("=" * 80)
    print(f"Mean:     {np.mean(transformer_sentiments):.4f}")
    print(f"Median:   {np.median(transformer_sentiments):.4f}")
    print(f"Std Dev:  {np.std(transformer_sentiments):.4f}")
    print(f"Min:      {np.min(transformer_sentiments):.4f}")
    print(f"Max:      {np.max(transformer_sentiments):.4f}")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as outfile:
        outfile.write(results_df.to_csv(index=False, sep="\t"))
    
    print("\nDone! Results saved successfully.")
    print("=" * 80)
    
    # Show sample results
    print("\nSample Results (first 5 reviews):")
    print("-" * 80)
    for i in range(min(5, len(results_df))):
        print(f"\nReview {i+1}:")
        print(f"Rating: {results_df.iloc[i]['rating']}")
        print(f"Sentiment Score: {results_df.iloc[i]['transformer_sentiment']:.4f}")
        print(f"Text: {results_df.iloc[i]['review'][:100]}...")


if __name__ == "__main__":
    main()
