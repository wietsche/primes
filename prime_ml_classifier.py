"""
Prime Number ML Classifier
This script generates a dataset of 7-digit prime and non-prime numbers,
extracts digit features, and trains an AutoML model to classify them.
"""

import argparse
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Constants
MAX_GENERATION_ATTEMPTS_MULTIPLIER = 1000  # Max attempts = count * multiplier


def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_prime_numbers(count, min_val=1000000, max_val=9999999):
    """Generate a specified number of 7-digit prime numbers.
    Only generates numbers that don't end in 0, 2, or 5."""
    primes = []
    attempts = 0
    max_attempts = count * MAX_GENERATION_ATTEMPTS_MULTIPLIER  # Prevent infinite loops
    valid_endings = [1, 3, 7, 9]  # Numbers ending in 0, 2, 5 are excluded
    
    # Calculate the range for random number generation (excluding the last digit)
    min_prefix = (min_val + 9) // 10  # Calculate minimum prefix to ensure candidates >= min_val
    max_prefix = max_val // 10
    
    while len(primes) < count and attempts < max_attempts:
        candidate = random.randint(min_prefix, max_prefix) * 10 + random.choice(valid_endings)
        # Bounds check needed because candidate may fall outside [min_val, max_val]
        # due to the random ending (e.g., min_prefix*10+9 might exceed max_val)
        if candidate < min_val or candidate > max_val:
            attempts += 1
            continue
        if is_prime(candidate) and candidate not in primes:
            primes.append(candidate)
        attempts += 1
    
    if len(primes) < count:
        raise ValueError(f"Could only generate {len(primes)} primes after {attempts} attempts")
    
    return primes


def generate_non_prime_numbers(count, min_val=1000000, max_val=9999999):
    """Generate a specified number of 7-digit non-prime numbers.
    Only generates numbers that don't end in 0, 2, or 5."""
    non_primes = []
    attempts = 0
    max_attempts = count * MAX_GENERATION_ATTEMPTS_MULTIPLIER
    valid_endings = [1, 3, 7, 9]  # Numbers ending in 0, 2, 5 are excluded
    
    # Calculate the range for random number generation (excluding the last digit)
    min_prefix = (min_val + 9) // 10  # Calculate minimum prefix to ensure candidates >= min_val
    max_prefix = max_val // 10
    
    while len(non_primes) < count and attempts < max_attempts:
        candidate = random.randint(min_prefix, max_prefix) * 10 + random.choice(valid_endings)
        # Bounds check needed because candidate may fall outside [min_val, max_val]
        # due to the random ending (e.g., min_prefix*10+9 might exceed max_val)
        if candidate < min_val or candidate > max_val:
            attempts += 1
            continue
        if not is_prime(candidate) and candidate not in non_primes:
            non_primes.append(candidate)
        attempts += 1
    
    if len(non_primes) < count:
        raise ValueError(f"Could only generate {len(non_primes)} non-primes after {attempts} attempts")
    
    return non_primes


def number_to_features(number):
    """Convert a 7-digit number to features (individual digits plus mathematical properties).
    
    Returns a dictionary with keys:
    - ten_power_0 through ten_power_6: individual digits
    - sum_digits: sum of all digits (useful for divisibility by 3, 9)
    - digital_root: digital root of the number (single digit 0-9, typically 1-9 for 7-digit numbers)
    - product_digits: product of all digits
    - last_two_digits: value of last two digits (0-99)
    """
    digits = str(number).zfill(7)  # Ensure 7 digits
    features = {}
    
    # Individual digits
    digit_values = []
    for i in range(7):
        # ten_power_0 is rightmost digit (ones place)
        # ten_power_6 is leftmost digit (millions place)
        digit_val = int(digits[6 - i])
        features[f'ten_power_{i}'] = digit_val
        digit_values.append(digit_val)
    
    # Mathematical features
    # Sum of digits (for divisibility by 3 and 9)
    features['sum_digits'] = sum(digit_values)
    
    # Digital root (iteratively sum digits until single digit)
    dr = features['sum_digits']
    while dr >= 10:
        dr = sum(int(d) for d in str(dr))
    features['digital_root'] = dr
    
    # Product of digits (patterns in composite numbers)
    product = 1
    for d in digit_values:
        product *= d
    features['product_digits'] = product
    
    # Last two digits value (patterns in primes)
    features['last_two_digits'] = digit_values[1] * 10 + digit_values[0]
    
    return features


def create_dataset(prime_numbers, non_prime_numbers):
    """Create a dataset from prime and non-prime numbers."""
    data = []
    
    # Add prime numbers
    for number in prime_numbers:
        features = number_to_features(number)
        features['prime'] = 1
        features['number'] = number
        data.append(features)
    
    # Add non-prime numbers
    for number in non_prime_numbers:
        features = number_to_features(number)
        features['prime'] = 0
        features['number'] = number
        data.append(features)
    
    df = pd.DataFrame(data)
    return df


def one_hot_encode_features(X):
    """
    Transform each feature to one-hot encoding.
    
    Each feature (value 0-9) is transformed to 10 binary features.
    For example, ten_power_0 with value 3 becomes:
    ten_power_0_is_0=0, ten_power_0_is_1=0, ten_power_0_is_2=0, ten_power_0_is_3=1, ..., ten_power_0_is_9=0
    
    Args:
        X: numpy array of shape (n_samples, n_features) with values 0-9
        
    Returns:
        numpy array of shape (n_samples, n_features * 10) with binary values
    """
    n_samples, n_features = X.shape
    n_values = 10  # Each digit can be 0-9
    
    # Create output array for one-hot encoded features
    X_encoded = np.zeros((n_samples, n_features * n_values), dtype=int)
    
    # For each feature, create one-hot encoding
    for feature_idx in range(n_features):
        for value in range(n_values):
            # Column index for this one-hot feature
            col_idx = feature_idx * n_values + value
            # Set to 1 where the feature equals this value
            X_encoded[:, col_idx] = (X[:, feature_idx] == value).astype(int)
    
    return X_encoded


def prepare_features(df, scaler=None):
    """
    Prepare features for training, combining one-hot encoded digits with mathematical features.
    
    Args:
        df: DataFrame with digit features and mathematical features
        scaler: Optional pre-fitted StandardScaler for test set transformation
        
    Returns:
        X: numpy array with combined features
        feature_info: dict with information about feature preparation
    """
    # One-hot encode the digit features
    digit_columns = [f'ten_power_{i}' for i in range(7)]
    X_digits = df[digit_columns].values
    X_digits_encoded = one_hot_encode_features(X_digits)
    
    # Get mathematical features (if they exist)
    math_feature_columns = ['sum_digits', 'digital_root', 'product_digits', 'last_two_digits']
    available_math_features = [col for col in math_feature_columns if col in df.columns]
    
    if available_math_features:
        X_math = df[available_math_features].values
        # Normalize mathematical features to prevent dominance
        if scaler is None:
            # Training phase - fit new scaler
            scaler = StandardScaler()
            X_math_scaled = scaler.fit_transform(X_math)
        else:
            # Test phase - use existing scaler
            X_math_scaled = scaler.transform(X_math)
        
        # Combine one-hot encoded digits with scaled mathematical features
        X = np.hstack([X_digits_encoded, X_math_scaled])
        feature_info = {
            'digit_features': 70,  # 7 digits × 10 values
            'math_features': len(available_math_features),
            'total_features': X.shape[1],
            'scaler': scaler,
            'math_feature_names': available_math_features
        }
    else:
        # Only digit features available (backward compatibility)
        X = X_digits_encoded
        feature_info = {
            'digit_features': 70,
            'math_features': 0,
            'total_features': 70,
            'scaler': None,
            'math_feature_names': []
        }
    
    return X, feature_info


def simple_automl(X_train, y_train, X_test, y_test):
    """
    Simple AutoML: Train multiple models and select the best one based on cross-validation.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    print("\n" + "="*60)
    print("AutoML Model Selection (5-fold Cross-Validation)")
    print("="*60)
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        results[name] = {
            'mean_cv_auc': mean_score,
            'std_cv_auc': std_score
        }
        
        print(f"{name:25s}: AUC = {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    print("="*60)
    print(f"Best Model: {best_name} with CV AUC = {best_score:.4f}")
    print("="*60)
    
    # Train the best model on full training data
    best_model.fit(X_train, y_train)
    
    return best_model, best_name, results


def plot_evaluation_metrics(y_test, y_pred, y_pred_proba, model_name, output_dir='.'):
    """Plot ROC curve and AUC."""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {auc_score:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].figure.colorbar(im, ax=axes[1])
    axes[1].set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=['Non-Prime', 'Prime'],
                yticklabels=['Non-Prime', 'Prime'],
                title=f'Confusion Matrix - {model_name}',
                ylabel='True label',
                xlabel='Predicted label')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_evaluation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plot saved as '{output_path}'")
    plt.close()


def load_dataset_from_csv(csv_path):
    """Load dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the dataset
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['ten_power_0', 'ten_power_1', 'ten_power_2', 'ten_power_3',
                           'ten_power_4', 'ten_power_5', 'ten_power_6', 'prime', 'number']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse CSV file '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to run the complete ML pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Prime Number ML Classifier - Train models to classify prime numbers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using a pre-generated CSV file
  python prime_ml_classifier.py --input prime_dataset.csv
  
  # Generate new dataset and train (default behavior)
  python prime_ml_classifier.py
  
  # Train using a large dataset
  python prime_ml_classifier.py --input prime_dataset_large.csv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file (generated by generate_dataset.py). If not provided, generates a new dataset.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory for output files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Prime Number ML Classifier")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Get output directory
    output_dir = args.output_dir
    
    # Step 1: Load or generate dataset
    if args.input:
        print(f"\nStep 1: Loading dataset from '{args.input}'...")
        print("-" * 60)
        df = load_dataset_from_csv(args.input)
        print(f"✓ Dataset loaded successfully")
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few samples:")
        print(df.head(10))
    else:
        print("\nStep 1: Generating 7-digit numbers...")
        print("-" * 60)
        prime_numbers = generate_prime_numbers(100)
        print(f"Generated {len(prime_numbers)} prime numbers")
        print(f"Sample primes: {prime_numbers[:5]}")
        
        non_prime_numbers = generate_non_prime_numbers(100)
        print(f"Generated {len(non_prime_numbers)} non-prime numbers")
        print(f"Sample non-primes: {non_prime_numbers[:5]}")
        
        # Step 2: Create dataset
        print("\nStep 2: Creating dataset with digit features...")
        print("-" * 60)
        df = create_dataset(prime_numbers, non_prime_numbers)
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few samples:")
        print(df.head(10))
        
        # Save dataset
        dataset_path = os.path.join(output_dir, 'prime_dataset.csv')
        df.to_csv(dataset_path, index=False)
        print(f"\nDataset saved as '{dataset_path}'")
    
    # Step 2: Prepare features and labels
    print("\nStep 2: Preparing features and labels...")
    print("-" * 60)
    y = df['prime'].values
    
    # Step 3: Split data (80/20) - split before feature preparation to avoid data leakage
    print("\nStep 3: Splitting data (80% train, 20% test)...")
    print("-" * 60)
    
    # Split indices to preserve all columns in both sets
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    y_train = df_train['prime'].values
    y_test = df_test['prime'].values
    
    print(f"Training set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    # Step 4: Prepare features with one-hot encoding and mathematical features
    print("\nStep 4: Preparing features with enhanced feature engineering...")
    print("-" * 60)
    X_train, feature_info = prepare_features(df_train)
    X_test, _ = prepare_features(df_test, scaler=feature_info['scaler'])
    
    print(f"Feature preparation complete:")
    print(f"  - Digit features (one-hot encoded): {feature_info['digit_features']}")
    print(f"  - Mathematical features: {feature_info['math_features']}")
    if feature_info['math_feature_names']:
        print(f"    ({', '.join(feature_info['math_feature_names'])})")
    print(f"  - Total features: {feature_info['total_features']}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 5: AutoML - Train and select best model
    print("\nStep 5: Training models with AutoML...")
    best_model, best_name, cv_results = simple_automl(X_train, y_train, X_test, y_test)
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating best model on test set...")
    print("="*60)
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Set Performance ({best_name}):")
    print("-" * 60)
    print(f"AUC Score: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non-Prime', 'Prime'],
                                digits=4))
    
    # Step 7: Plot evaluation metrics
    print("\nStep 7: Generating evaluation plots...")
    print("-" * 60)
    plot_evaluation_metrics(y_test, y_pred, y_pred_proba, best_name, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total samples: {len(df)} ({df['prime'].sum()} primes + {len(df) - df['prime'].sum()} non-primes)")
    print(f"Training samples: {len(df_train)} (80%)")
    print(f"Test samples: {len(df_test)} (20%)")
    print(f"Features: {feature_info['total_features']} ({feature_info['digit_features']} digit + {feature_info['math_features']} mathematical)")
    print(f"Best model: {best_name}")
    print(f"Test AUC: {test_auc:.4f}")
    print("="*60)
    
    return df, best_model, cv_results, test_auc


if __name__ == "__main__":
    main()
