"""
Prime Number Neural Network Classifier with Hyperparameter Search
This script uses neural networks with hyperparameter search to find the best F-score
for classifying 7-digit numbers as prime or non-prime.
"""

import argparse
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    make_scorer, precision_recall_fscore_support
)
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Import functions from prime_ml_classifier
from prime_ml_classifier import (
    generate_prime_numbers,
    generate_non_prime_numbers,
    create_dataset,
    load_dataset_from_csv,
    one_hot_encode_features
)


def neural_network_hyperparameter_search(X_train, y_train, X_test, y_test, cv=5):
    """
    Perform hyperparameter search on neural network to find the best F-score.
    
    Args:
        X_train: Training features (already one-hot encoded)
        y_train: Training labels
        X_test: Test features (already one-hot encoded)
        y_test: Test labels
        cv: Number of cross-validation folds
        
    Returns:
        best_model: The best neural network model
        best_params: Best hyperparameters found
        cv_results: Cross-validation results
        best_f1_score: Best F1 score achieved
    """
    # Define hyperparameter grid for neural networks
    param_grid = {
        'hidden_layer_sizes': [
            (50,),
            (100,),
            (50, 30),
            (100, 50),
            (100, 50, 25),
            (150, 100, 50)
        ],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000]
    }
    
    # Create MLPClassifier with fixed random state
    mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    
    # Use F1 score as the scoring metric
    f1_scorer = make_scorer(f1_score)
    
    print("\n" + "="*60)
    print("Neural Network Hyperparameter Search")
    print("="*60)
    # Calculate total combinations dynamically
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    print(f"Parameter grid size: {total_combinations} combinations")
    print(f"Cross-validation folds: {cv}")
    print(f"Scoring metric: F1 Score")
    print("="*60)
    
    # Perform grid search
    grid_search = GridSearchCV(
        mlp,
        param_grid,
        cv=cv,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    print("\nStarting grid search... This may take a few minutes...")
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_
    
    print("\n" + "="*60)
    print("Hyperparameter Search Results")
    print("="*60)
    print(f"Best Cross-Validation F1 Score: {best_cv_f1:.4f}")
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Show top 5 models
    print("\nTop 5 Model Configurations:")
    print("-" * 60)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    for idx, row in results_df.head(5).iterrows():
        print(f"Rank {int(row['rank_test_score'])}: F1={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        print(f"  Params: {row['params']}")
    
    print("="*60)
    
    return best_model, best_params, grid_search.cv_results_, best_cv_f1


def evaluate_model(model, X_test, y_test, model_name, output_dir='.'):
    """
    Evaluate the model on test set and generate visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        output_dir: Directory to save output files
        
    Returns:
        test_f1: F1 score on test set
        y_pred: Predictions on test set
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    
    print("\n" + "="*60)
    print(f"Test Set Evaluation - {model_name}")
    print("="*60)
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non-Prime', 'Prime'],
                                digits=4))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot Confusion Matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Non-Prime', 'Prime'],
           yticklabels=['Non-Prime', 'Prime'],
           title=f'Confusion Matrix - {model_name}\nF1 Score: {f1:.4f}',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_evaluation_nn.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plot saved as '{output_path}'")
    plt.close()
    
    return f1, y_pred


def main():
    """Main function to run the neural network hyperparameter search pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Prime Number Neural Network Classifier with Hyperparameter Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using a pre-generated CSV file
  python model_evaluation_nn.py --input prime_dataset.csv
  
  # Generate new dataset and train (default behavior)
  python model_evaluation_nn.py
  
  # Train using a large dataset with more CV folds
  python model_evaluation_nn.py --input prime_dataset_large.csv --cv 10
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
    
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Prime Number Neural Network Classifier")
    print("With Hyperparameter Search for Best F-Score")
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
    feature_columns = [f'ten_power_{i}' for i in range(7)]
    X = df[feature_columns].values
    y = df['prime'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature columns: {feature_columns}")
    
    # Step 3: Split data (80/20)
    print("\nStep 3: Splitting data (80% train, 20% test)...")
    print("-" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    # Step 4: Apply one-hot encoding transformation
    print("\nStep 4: Applying one-hot encoding transformation...")
    print("-" * 60)
    print(f"Original features shape: {X_train.shape}")
    X_train_encoded = one_hot_encode_features(X_train)
    X_test_encoded = one_hot_encode_features(X_test)
    print(f"One-hot encoded features shape: {X_train_encoded.shape}")
    print(f"Each of the 7 features (ten_power_0 to ten_power_6) is now represented by 10 binary features")
    print(f"Total features: 7 × 10 = 70 binary features")
    
    # Step 5: Neural Network Hyperparameter Search
    print("\nStep 5: Neural Network with Hyperparameter Search...")
    best_model, best_params, cv_results, best_cv_f1 = neural_network_hyperparameter_search(
        X_train_encoded, y_train, X_test_encoded, y_test, cv=args.cv
    )
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating best model on test set...")
    test_f1, y_pred = evaluate_model(
        best_model, X_test_encoded, y_test, 
        "Neural Network (Best Hyperparameters)", 
        output_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total samples: {len(df)} ({df['prime'].sum()} primes + {len(df) - df['prime'].sum()} non-primes)")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Test samples: {len(X_test)} (20%)")
    print(f"Features: 7 digit positions transformed to 70 one-hot encoded features")
    print(f"Model: Neural Network (MLPClassifier)")
    print(f"Hyperparameter optimization: Grid Search with {args.cv}-fold CV")
    print(f"Best CV F1 Score: {best_cv_f1:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("="*60)
    
    return df, best_model, best_params, test_f1


if __name__ == "__main__":
    main()
