#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) on Prime Dataset

This script performs PCA on the 7-digit prime dataset to visualize
primes and non-primes in 2D space using two side-by-side plots.
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path):
    """Load dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the dataset
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = [f'ten_power_{i}' for i in range(7)] + ['prime']
        
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


def perform_pca_analysis(df):
    """Perform PCA on the dataset features.
    
    Args:
        df: DataFrame with features and prime labels
        
    Returns:
        tuple: (pca_components, prime_mask, pca_model, scaler)
    """
    # Extract features (7 digit positions)
    feature_columns = [f'ten_power_{i}' for i in range(7)]
    X = df[feature_columns].values
    
    # Get prime labels (1 for prime, 0 for non-prime)
    prime_mask = df['prime'].values == 1
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, prime_mask, pca, scaler


def plot_pca_results(X_pca, prime_mask, pca, output_path='pca_analysis.png'):
    """Create side-by-side plots of primes and non-primes in 2D PCA space.
    
    Args:
        X_pca: PCA-transformed data (Nx2 array)
        prime_mask: Boolean mask indicating primes (True) vs non-primes (False)
        pca: Fitted PCA model
        output_path: Path to save the output image
    """
    # Split data into primes and non-primes
    primes_pca = X_pca[prime_mask]
    non_primes_pca = X_pca[~prime_mask]
    
    # Create figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate explained variance for axis labels
    var_explained = pca.explained_variance_ratio_
    pc1_var = var_explained[0] * 100
    pc2_var = var_explained[1] * 100
    
    # Left plot: Primes
    axes[0].scatter(primes_pca[:, 0], primes_pca[:, 1], 
                   c='royalblue', alpha=0.6, s=50, edgecolors='darkblue', linewidth=0.5)
    axes[0].set_xlabel(f'First Principal Component ({pc1_var:.1f}% variance)', fontsize=12)
    axes[0].set_ylabel(f'Second Principal Component ({pc2_var:.1f}% variance)', fontsize=12)
    axes[0].set_title('Prime Numbers in 2D PCA Space', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    axes[0].axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Add text annotation with count
    axes[0].text(0.02, 0.98, f'n = {len(primes_pca)}', 
                transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right plot: Non-primes
    axes[1].scatter(non_primes_pca[:, 0], non_primes_pca[:, 1], 
                   c='coral', alpha=0.6, s=50, edgecolors='darkred', linewidth=0.5)
    axes[1].set_xlabel(f'First Principal Component ({pc1_var:.1f}% variance)', fontsize=12)
    axes[1].set_ylabel(f'Second Principal Component ({pc2_var:.1f}% variance)', fontsize=12)
    axes[1].set_title('Non-Prime Numbers in 2D PCA Space', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    axes[1].axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Add text annotation with count
    axes[1].text(0.02, 0.98, f'n = {len(non_primes_pca)}', 
                transform=axes[1].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Principal Component Analysis of 7-Digit Numbers\n(Based on Individual Digit Features)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPCA visualization saved as '{output_path}'")
    plt.close()


def main():
    """Main function to run PCA analysis."""
    parser = argparse.ArgumentParser(
        description='Perform PCA analysis on prime number dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default dataset (prime_dataset.csv)
  python pca_analysis.py
  
  # Use a custom dataset
  python pca_analysis.py --input my_dataset.csv
  
  # Specify custom output file
  python pca_analysis.py --output my_pca_plot.png
  
  # Use both custom input and output
  python pca_analysis.py --input large_dataset.csv --output large_pca.png
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='prime_dataset.csv',
        help='Path to input CSV file (default: prime_dataset.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='pca_analysis.png',
        help='Path to output PNG file (default: pca_analysis.png)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PCA Analysis on Prime Number Dataset")
    print("="*60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print("="*60)
    
    # Step 1: Load dataset
    print("\nStep 1: Loading dataset...")
    print("-" * 60)
    df = load_dataset(args.input)
    print(f"✓ Dataset loaded successfully")
    print(f"Dataset shape: {df.shape}")
    
    # Count primes and non-primes
    num_primes = df['prime'].sum()
    num_non_primes = len(df) - num_primes
    print(f"Primes: {num_primes}, Non-primes: {num_non_primes}")
    
    # Step 2: Perform PCA
    print("\nStep 2: Performing PCA (7D → 2D)...")
    print("-" * 60)
    X_pca, prime_mask, pca, scaler = perform_pca_analysis(df)
    print(f"✓ PCA completed")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Step 3: Create visualization
    print("\nStep 3: Creating side-by-side PCA visualizations...")
    print("-" * 60)
    plot_pca_results(X_pca, prime_mask, pca, args.output)
    print("✓ Visualization completed")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total samples analyzed: {len(df)}")
    print(f"  - Prime samples: {num_primes}")
    print(f"  - Non-prime samples: {num_non_primes}")
    print(f"Original features: 7 (digit positions)")
    print(f"PCA components: 2")
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Output saved: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
