# Prime Number ML Classifier

An AutoML-based machine learning pipeline for classifying 7-digit numbers as prime or non-prime using digit-level features.

## Overview

This project generates a dataset of 200 7-digit numbers (100 primes and 100 non-primes), extracts individual digits as features, and trains multiple ML models to identify prime numbers. The best model is automatically selected using cross-validation.

## Features

- **Custom Dataset Generation**: Generate CSV datasets with any number of prime and non-prime samples using `generate_dataset.py`
- **Data Generation**: Automatically generates 100 7-digit prime and 100 non-prime numbers (default)
- **Advanced Feature Engineering**: Converts each number into rich features:
  - 7 digit position features (one-hot encoded: `ten_power_0` through `ten_power_6`)
  - 4 mathematical features based on prime number properties:
    - Sum of digits (for divisibility rules)
    - Digital root (iterative digit sum)
    - Product of digits
    - Last two digits value
- **PCA Visualization**: Visualize primes and non-primes in 2D space using Principal Component Analysis with `pca_analysis.py`
- **AutoML**: Trains and evaluates multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Neural Network (MLP)
- **Model Selection**: Automatically selects the best model based on 5-fold cross-validation AUC
- **Neural Network Hyperparameter Search**: Advanced model evaluation with grid search to optimize F1-score using neural networks
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

You can either train with a pre-generated CSV file or let the script generate a new dataset:

```bash
# Train using a pre-generated CSV file (recommended workflow)
python generate_dataset.py --output my_dataset.csv
python prime_ml_classifier.py --input my_dataset.csv

# Generate new dataset and train (default behavior, generates 200 samples)
python prime_ml_classifier.py

# Train using a large pre-generated dataset
python generate_dataset.py --primes 1000 --non-primes 1000 --output large_dataset.csv
python prime_ml_classifier.py --input large_dataset.csv
```

Run `python prime_ml_classifier.py --help` for all available options.

### Visualize Data with PCA

To visualize the dataset in 2D space using Principal Component Analysis:

```bash
# Use default dataset (prime_dataset.csv)
python pca_analysis.py

# Use a custom dataset
python pca_analysis.py --input my_dataset.csv

# Specify custom output file
python pca_analysis.py --output my_pca_plot.png

# Use both custom input and output
python pca_analysis.py --input large_dataset.csv --output large_pca.png
```

Run `python pca_analysis.py --help` for all available options.

### Neural Network Model with Hyperparameter Search

For advanced model evaluation focusing on F1-score optimization:

```bash
# Train using a pre-generated CSV file with neural network hyperparameter search
python model_evaluation_nn.py --input prime_dataset.csv

# Generate new dataset and train with neural network (default behavior)
python model_evaluation_nn.py

# Train with more cross-validation folds for better hyperparameter search
python model_evaluation_nn.py --input prime_dataset.csv --cv 10

# Train using a large dataset with custom CV folds
python generate_dataset.py --primes 1000 --non-primes 1000 --output large_dataset.csv
python model_evaluation_nn.py --input large_dataset.csv --cv 10
```

This script performs grid search over neural network hyperparameters to find the best F1-score:
- Hidden layer sizes: (50,), (100,), (50, 30), (100, 50), (100, 50, 25), (150, 100, 50)
- Activation functions: ReLU, tanh
- Regularization (alpha): 0.0001, 0.001, 0.01
- Learning rates: constant, adaptive

Run `python model_evaluation_nn.py --help` for all available options.

## Output

The scripts generate:
1. **prime_dataset.csv** - The complete dataset with 200 samples
2. **model_evaluation.png** - Visualization of ROC curve and confusion matrix (from `prime_ml_classifier.py`)
3. **model_evaluation_nn.png** - Confusion matrix with F1-score from neural network hyperparameter search (from `model_evaluation_nn.py`)
4. **pca_analysis.png** - 2D PCA visualization showing primes and non-primes side-by-side

## Dataset Structure

Each sample has the following features:
- `ten_power_0`: Ones digit (rightmost)
- `ten_power_1`: Tens digit
- `ten_power_2`: Hundreds digit
- `ten_power_3`: Thousands digit
- `ten_power_4`: Ten thousands digit
- `ten_power_5`: Hundred thousands digit
- `ten_power_6`: Millions digit (leftmost)
- `sum_digits`: Sum of all digits (useful for divisibility by 3 and 9)
- `digital_root`: Digital root of the number (iterative sum until single digit)
- `product_digits`: Product of all digits
- `last_two_digits`: Value of the last two digits (0-99)
- `prime`: Label (1 for prime, 0 for non-prime)
- `number`: The original 7-digit number

**Note:** The classifier uses 70 one-hot encoded features for the digits plus 4 scaled mathematical features, totaling 74 features for training.

## Results

The AutoML process evaluates multiple models and automatically selects the best performer. Results include:
- Cross-validation AUC scores for all models
- Test set AUC score
- Classification report with precision, recall, and F1-score
- ROC curve visualization
- Confusion matrix