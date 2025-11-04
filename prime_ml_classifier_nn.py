"""
Prime Number Neural Network Classifier with Hyperparameter Search
This script uses neural networks with hyperparameter search to find the best F-score
for classifying 11-digit numbers as prime or non-prime.
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
    prepare_features,
    NUM_DIGITS
)


def neural_network_hyperparameter_search(X_train, y_train, X_test, y_test, cv=5):
    """
    Perform hyperparameter search on neural network to find the best F-score.
    
    Args:
        X_train: Training features (full feature set from CSV)
        y_train: Training labels
        X_test: Test features (full feature set from CSV)
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


def evaluate_model(model, X_test, y_test, model_name, best_params, output_dir='.'):
    """
    Evaluate the model on test set and generate visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        best_params: Best hyperparameters from grid search
        output_dir: Directory to save output files
        
    Returns:
        test_f1: F1 score on test set
        y_pred: Predictions on test set
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, average=None
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
    
    # Plot confusion matrix and additional visualizations
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a figure with 2x2 subplots for more interesting visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{model_name} - Comprehensive Evaluation\nF1 Score: {f1:.4f}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Subplot 1: Confusion Matrix
    im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 0].figure.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=['Non-Prime', 'Prime'],
                   yticklabels=['Non-Prime', 'Prime'],
                   title='Confusion Matrix',
                   ylabel='True label',
                   xlabel='Predicted label')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=16)
    
    # Subplot 2: Per-Class Metrics Bar Chart
    metrics_labels = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    non_prime_metrics = [precision_per_class[0], recall_per_class[0], f1_per_class[0]]
    prime_metrics = [precision_per_class[1], recall_per_class[1], f1_per_class[1]]
    
    bars1 = axes[0, 1].bar(x - width/2, non_prime_metrics, width, label='Non-Prime', 
                           color='skyblue', edgecolor='black')
    bars2 = axes[0, 1].bar(x + width/2, prime_metrics, width, label='Prime',
                           color='lightcoral', edgecolor='black')
    
    axes[0, 1].set_xlabel('Metrics', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_labels)
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Prediction Distribution (Histogram)
    axes[1, 0].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='Non-Prime (True)',
                    color='skyblue', edgecolor='black')
    axes[1, 0].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='Prime (True)',
                    color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Predicted Probability (Prime)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='upper center')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    
    # Subplot 4: Model Configuration Summary
    axes[1, 1].axis('off')
    config_text = "Best Hyperparameters:\n" + "="*40 + "\n"
    for param, value in best_params.items():
        config_text += f"{param}: {value}\n"
    
    config_text += "\n" + "="*40 + "\n"
    config_text += f"Test Set Size: {len(y_test)} samples\n"
    config_text += f"  • Non-Prime: {support_per_class[0]:.0f} samples\n"
    config_text += f"  • Prime: {support_per_class[1]:.0f} samples\n"
    config_text += "\n" + "="*40 + "\n"
    config_text += "Overall Metrics:\n"
    config_text += f"  • Accuracy: {(y_pred == y_test).mean():.4f}\n"
    config_text += f"  • Precision: {precision:.4f}\n"
    config_text += f"  • Recall: {recall:.4f}\n"
    config_text += f"  • F1-Score: {f1:.4f}\n"
    
    axes[1, 1].text(0.1, 0.95, config_text, 
                   transform=axes[1, 1].transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1, 1].set_title('Model Summary', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prime_ml_classifier_nn.png')
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
  python prime_ml_classifier_nn.py --input prime_dataset.csv
  
  # Generate new dataset and train (default behavior)
  python prime_ml_classifier_nn.py
  
  # Train using a large dataset with more CV folds
  python prime_ml_classifier_nn.py --input prime_dataset_large.csv --cv 10
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
        print("\nStep 1: Generating 11-digit numbers...")
        print("-" * 60)
        prime_numbers = generate_prime_numbers(100)
        print(f"Generated {len(prime_numbers)} prime numbers")
        print(f"Sample primes: {prime_numbers[:5]}")
        
        non_prime_numbers = generate_non_prime_numbers(100)
        print(f"Generated {len(non_prime_numbers)} non-prime numbers")
        print(f"Sample non-primes: {non_prime_numbers[:5]}")
        
        # Step 2: Create dataset
        print("\nStep 2: Creating dataset with all features...")
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
    
    # Step 4: Extract features (all preprocessing already done in CSV)
    print("\nStep 4: Extracting features from dataset...")
    print("-" * 60)
    X_train, feature_info = prepare_features(df_train)
    X_test, _ = prepare_features(df_test)
    
    print(f"Feature extraction complete:")
    print(f"  - One-hot encoded digit features: {feature_info['onehot_features']}")
    print(f"  - Mathematical features: {feature_info['math_features']}")
    print(f"  - Total features: {feature_info['total_features']}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 5: Neural Network Hyperparameter Search
    print("\nStep 5: Neural Network with Hyperparameter Search...")
    best_model, best_params, cv_results, best_cv_f1 = neural_network_hyperparameter_search(
        X_train, y_train, X_test, y_test, cv=args.cv
    )
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating best model on test set...")
    test_f1, y_pred = evaluate_model(
        best_model, X_test, y_test, 
        "Neural Network (Best Hyperparameters)",
        best_params,
        output_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total samples: {len(df)} ({df['prime'].sum()} primes + {len(df) - df['prime'].sum()} non-primes)")
    print(f"Training samples: {len(df_train)} (80%)")
    print(f"Test samples: {len(df_test)} (20%)")
    print(f"Features: {feature_info['total_features']} ({feature_info['onehot_features']} one-hot + {feature_info['math_features']} mathematical)")
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
