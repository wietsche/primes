"""
Generate Prime Number Dataset CSV

This script generates a dataset of 11-digit prime and non-prime numbers,
extracts digit features, and saves to CSV format.
It allows customization of the number of samples to generate.
"""

import argparse
import sys

try:
    from prime_ml_classifier import (
        generate_prime_numbers,
        generate_non_prime_numbers,
        create_dataset
    )
except ImportError as e:
    print("Error: Failed to import from prime_ml_classifier.py", file=sys.stderr)
    print(f"Make sure prime_ml_classifier.py is in the same directory.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)


def main():
    """Main function to generate dataset CSV."""
    parser = argparse.ArgumentParser(
        description='Generate a dataset of 11-digit prime and non-prime numbers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default 200 samples (100 primes + 100 non-primes)
  python generate_dataset.py
  
  # Generate 1000 samples (500 primes + 500 non-primes)
  python generate_dataset.py --primes 500 --non-primes 500
  
  # Generate to custom output file
  python generate_dataset.py --output my_dataset.csv --primes 200 --non-primes 200
  
  # Generate 2000 primes and 2000 non-primes for larger training set
  python generate_dataset.py --primes 2000 --non-primes 2000 --output prime_dataset_large.csv
        """
    )
    
    parser.add_argument(
        '--primes',
        type=int,
        default=100,
        help='Number of prime numbers to generate (default: 100)'
    )
    
    parser.add_argument(
        '--non-primes',
        type=int,
        default=100,
        help='Number of non-prime numbers to generate (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='prime_dataset.csv',
        help='Output CSV file path (default: prime_dataset.csv)'
    )
    
    parser.add_argument(
        '--min-value',
        type=int,
        default=10000000000,
        help='Minimum value for 11-digit numbers (default: 10000000000)'
    )
    
    parser.add_argument(
        '--max-value',
        type=int,
        default=99999999999,
        help='Maximum value for 11-digit numbers (default: 99999999999)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.primes <= 0:
        print("Error: --primes must be greater than 0", file=sys.stderr)
        sys.exit(1)
    
    if args.non_primes <= 0:
        print("Error: --non-primes must be greater than 0", file=sys.stderr)
        sys.exit(1)
    
    if args.min_value < 10000000000 or args.max_value > 99999999999:
        print("Error: min-value and max-value must be within 11-digit range (10000000000-99999999999)", file=sys.stderr)
        sys.exit(1)
    
    if args.min_value >= args.max_value:
        print("Error: min-value must be less than max-value", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration
    print("="*60)
    print("Prime Number Dataset Generator")
    print("="*60)
    print(f"Configuration:")
    print(f"  Prime numbers to generate: {args.primes}")
    print(f"  Non-prime numbers to generate: {args.non_primes}")
    print(f"  Total samples: {args.primes + args.non_primes}")
    print(f"  Output file: {args.output}")
    print(f"  Number range: {args.min_value} - {args.max_value}")
    print("="*60)
    
    # Generate prime numbers
    print(f"\nGenerating {args.primes} prime numbers...")
    try:
        prime_numbers = generate_prime_numbers(
            args.primes, 
            min_val=args.min_value, 
            max_val=args.max_value
        )
        print(f"✓ Generated {len(prime_numbers)} prime numbers")
        if len(prime_numbers) >= 5:
            print(f"  Sample primes: {prime_numbers[:5]}")
    except ValueError as e:
        print(f"Error generating primes: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate non-prime numbers
    print(f"\nGenerating {args.non_primes} non-prime numbers...")
    try:
        non_prime_numbers = generate_non_prime_numbers(
            args.non_primes,
            min_val=args.min_value,
            max_val=args.max_value
        )
        print(f"✓ Generated {len(non_prime_numbers)} non-prime numbers")
        if len(non_prime_numbers) >= 5:
            print(f"  Sample non-primes: {non_prime_numbers[:5]}")
    except ValueError as e:
        print(f"Error generating non-primes: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create dataset
    print(f"\nCreating dataset with digit features...")
    df = create_dataset(prime_numbers, non_prime_numbers)
    print(f"✓ Dataset created with shape: {df.shape}")
    
    # Save to CSV
    print(f"\nSaving to '{args.output}'...")
    df.to_csv(args.output, index=False)
    print(f"✓ Dataset saved successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"  - Prime samples: {args.primes}")
    print(f"  - Non-prime samples: {args.non_primes}")
    print(f"Features: 11 digit positions (ten_power_0 to ten_power_10) + mathematical features")
    print(f"Output file: {args.output}")
    print("="*60)
    print("\nDataset preview:")
    print(df.head(10))


if __name__ == "__main__":
    main()
