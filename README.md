# Prime Number ML Classifier

An AutoML-based machine learning pipeline for classifying 7-digit numbers as prime or non-prime using digit-level features.

## Overview

This project generates a dataset of 200 7-digit numbers (100 primes and 100 non-primes), extracts individual digits as features, and trains multiple ML models to identify prime numbers. The best model is automatically selected using cross-validation.

## Features

- **Custom Dataset Generation**: Generate CSV datasets with any number of prime and non-prime samples using `generate_dataset.py`
- **Data Generation**: Automatically generates 100 7-digit prime and 100 non-prime numbers (default)
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

### Generate Dataset with Custom Sample Size

To generate a dataset CSV file with a custom number of samples:

```bash
# Generate default 200 samples (100 primes + 100 non-primes)
python generate_dataset.py

# Generate 1000 samples (500 primes + 500 non-primes)
python generate_dataset.py --primes 500 --non-primes 500

# Generate to custom output file
python generate_dataset.py --output my_dataset.csv --primes 200 --non-primes 200

# Generate 2000 primes and 2000 non-primes for larger training set
python generate_dataset.py --primes 2000 --non-primes 2000 --output prime_dataset_large.csv
```

Run `python generate_dataset.py --help` for all available options.

### Run Complete ML Pipeline

Run the complete ML pipeline with default dataset (200 samples):

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