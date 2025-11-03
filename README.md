# Prime Number ML Classifier

An AutoML-based machine learning pipeline for classifying 7-digit numbers as prime or non-prime using digit-level features.

## Overview

This project generates a dataset of 200 7-digit numbers (100 primes and 100 non-primes), extracts individual digits as features, and trains multiple ML models to identify prime numbers. The best model is automatically selected using cross-validation.

## Features

- **Data Generation**: Automatically generates 100 7-digit prime and 100 non-prime numbers
- **Feature Engineering**: Converts each number into 7 features (one per digit position: `ten_power_0` through `ten_power_6`)
- **AutoML**: Trains and evaluates multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Neural Network (MLP)
- **Model Selection**: Automatically selects the best model based on 5-fold cross-validation AUC
- **Evaluation**: Generates ROC curves, AUC scores, confusion matrices, and classification reports

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete ML pipeline:

```bash
python prime_ml_classifier.py
```

## Output

The script generates:
1. **prime_dataset.csv** - The complete dataset with 200 samples
2. **model_evaluation.png** - Visualization of ROC curve and confusion matrix

## Dataset Structure

Each sample has the following features:
- `ten_power_0`: Ones digit (rightmost)
- `ten_power_1`: Tens digit
- `ten_power_2`: Hundreds digit
- `ten_power_3`: Thousands digit
- `ten_power_4`: Ten thousands digit
- `ten_power_5`: Hundred thousands digit
- `ten_power_6`: Millions digit (leftmost)
- `prime`: Label (1 for prime, 0 for non-prime)
- `number`: The original 7-digit number

## Results

The AutoML process evaluates multiple models and automatically selects the best performer. Results include:
- Cross-validation AUC scores for all models
- Test set AUC score
- Classification report with precision, recall, and F1-score
- ROC curve visualization
- Confusion matrix