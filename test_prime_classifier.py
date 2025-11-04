"""
Simple test to verify the prime ML classifier functions work correctly.
"""

import sys
import os
import tempfile
import subprocess
import numpy as np
from prime_ml_classifier import (
    is_prime, 
    number_to_features, 
    generate_prime_numbers,
    generate_non_prime_numbers,
    load_dataset_from_csv,
    prepare_features,
    NUM_DIGITS
)


def test_is_prime():
    """Test the is_prime function."""
    print("Testing is_prime function...")
    
    # Test known primes
    assert is_prime(2) == True, "2 should be prime"
    assert is_prime(3) == True, "3 should be prime"
    assert is_prime(5) == True, "5 should be prime"
    assert is_prime(7) == True, "7 should be prime"
    assert is_prime(1000003) == True, "1000003 should be prime"
    
    # Test known non-primes
    assert is_prime(1) == False, "1 should not be prime"
    assert is_prime(4) == False, "4 should not be prime"
    assert is_prime(6) == False, "6 should not be prime"
    assert is_prime(8) == False, "8 should not be prime"
    assert is_prime(1000000) == False, "1000000 should not be prime"
    
    print("✓ is_prime tests passed")


def test_number_to_features():
    """Test the number_to_features function."""
    print("\nTesting number_to_features function...")
    
    # Test with 11-digit number
    features = number_to_features(12345678901)
    assert features['ten_power_0'] == 1, "Ones digit should be 1"
    assert features['ten_power_1'] == 0, "Tens digit should be 0"
    assert features['ten_power_2'] == 9, "Hundreds digit should be 9"
    assert features['ten_power_10'] == 1, "Ten billions digit should be 1"
    
    # Test one-hot encoding
    assert features['ten_power_0_is_1'] == 1, "ten_power_0_is_1 should be 1"
    assert features['ten_power_0_is_0'] == 0, "ten_power_0_is_0 should be 0"
    assert features['ten_power_1_is_0'] == 1, "ten_power_1_is_0 should be 1"
    
    # Test mathematical features
    assert features['sum_digits'] == 46, "Sum of digits should be 1+2+3+4+5+6+7+8+9+0+1=46"
    assert features['digital_root'] == 1, "Digital root of 46 is 4+6=10, 1+0=1"
    assert features['product_digits'] == 0, "Product should be 0 (contains 0)"
    assert features['last_two_digits'] == 1, "Last two digits should be 01"
    
    # Test new features
    assert 'alternating_digit_sum' in features, "Should have alternating_digit_sum"
    assert 'mod_2' in features, "Should have mod_2"
    assert 'mod_3' in features, "Should have mod_3"
    assert 'mod_5' in features, "Should have mod_5"
    assert 'mod_7' in features, "Should have mod_7"
    assert 'mod_11' in features, "Should have mod_11"
    
    # Test modulo calculations
    assert features['mod_2'] == 12345678901 % 2, "mod_2 should match"
    assert features['mod_3'] == 12345678901 % 3, "mod_3 should match"
    
    # Test with another number
    features2 = number_to_features(98765432109)
    assert features2['ten_power_0'] == 9, "Ones digit should be 9"
    assert features2['ten_power_10'] == 9, "Ten billions digit should be 9"
    
    print("✓ number_to_features tests passed")


def test_generate_prime_numbers():
    """Test prime number generation."""
    print("\nTesting generate_prime_numbers function...")
    
    from prime_ml_classifier import MIN_VAL, MAX_VAL
    primes = generate_prime_numbers(10)
    
    assert len(primes) == 10, "Should generate 10 prime numbers"
    
    # Check all are 11-digit numbers
    for p in primes:
        assert MIN_VAL <= p <= MAX_VAL, f"{p} should be 11-digit"
        assert is_prime(p), f"{p} should be prime"
    
    # Check uniqueness
    assert len(primes) == len(set(primes)), "All primes should be unique"
    
    print(f"✓ Generated 10 primes: {primes[:5]}...")


def test_generate_non_prime_numbers():
    """Test non-prime number generation."""
    print("\nTesting generate_non_prime_numbers function...")
    
    from prime_ml_classifier import MIN_VAL, MAX_VAL
    non_primes = generate_non_prime_numbers(10)
    
    assert len(non_primes) == 10, "Should generate 10 non-prime numbers"
    
    # Check all are 11-digit numbers and not prime
    for np in non_primes:
        assert MIN_VAL <= np <= MAX_VAL, f"{np} should be 11-digit"
        assert not is_prime(np), f"{np} should not be prime"
    
    # Check uniqueness
    assert len(non_primes) == len(set(non_primes)), "All non-primes should be unique"
    
    print(f"✓ Generated 10 non-primes: {non_primes[:5]}...")


def test_number_endings():
    """Test that generated numbers don't end in 0, 2, or 5."""
    print("\nTesting that numbers don't end in 0, 2, or 5...")
    
    # Generate numbers
    primes = generate_prime_numbers(20)
    non_primes = generate_non_prime_numbers(20)
    
    # Check primes
    for p in primes:
        last_digit = p % 10
        assert last_digit not in [0, 2, 5], f"Prime {p} should not end in 0, 2, or 5"
        assert last_digit in [1, 3, 7, 9], f"Prime {p} should end in 1, 3, 7, or 9"
    
    # Check non-primes
    for np in non_primes:
        last_digit = np % 10
        assert last_digit not in [0, 2, 5], f"Non-prime {np} should not end in 0, 2, or 5"
        assert last_digit in [1, 3, 7, 9], f"Non-prime {np} should end in 1, 3, 7, or 9"
    
    print("✓ All numbers correctly end in 1, 3, 7, or 9")


def test_load_dataset_from_csv():
    """Test loading dataset from CSV file."""
    print("\nTesting load_dataset_from_csv function...")
    
    # Create a temporary CSV file with minimal required columns for 11-digit numbers
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        # Write header with all 11 digit columns
        header = ','.join([f'ten_power_{i}' for i in range(NUM_DIGITS)] + ['prime', 'number'])
        tmp.write(header + '\n')
        # Write a sample row (12345678901)
        tmp.write('1,0,9,8,7,6,5,4,3,2,1,1,12345678901\n')
    
    try:
        # Load the dataset
        df = load_dataset_from_csv(tmp_path)
        
        # Verify the data
        assert len(df) == 1, "Should have 1 row"
        assert df.shape[1] >= 13, "Should have at least 13 columns (11 digits + prime + number)"
        
        # Check row
        assert df.iloc[0]['number'] == 12345678901, "Number should be 12345678901"
        assert df.iloc[0]['prime'] == 1, "Number should be marked as prime"
        assert df.iloc[0]['ten_power_0'] == 1, "ten_power_0 should be 1"
        
        
        print("✓ load_dataset_from_csv tests passed")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_prepare_features():
    """Test the prepare_features function."""
    print("\nTesting prepare_features function...")
    
    import pandas as pd
    
    # Create a simple test with actual generated features
    from prime_ml_classifier import create_dataset
    primes = [12345678901, 98765432109]
    non_primes = [12345678903, 98765432111]
    df = create_dataset(primes, non_primes)
    
    # Test feature extraction
    X, feature_info = prepare_features(df)
    
    # Check that we got features
    assert X.shape[0] == 4, "Should have 4 samples"
    assert X.shape[1] > 110, "Should have more than 110 features (one-hot + math)"
    assert feature_info['total_features'] == X.shape[1], "Total features should match X shape"
    assert feature_info['onehot_features'] == 110, "Should have 110 one-hot features (11*10)"
    assert feature_info['math_features'] > 0, "Should have mathematical features"
    
    print("✓ prepare_features tests passed")


def test_classifier_with_csv_input():
    """Test that prime_ml_classifier.py works with CSV input."""
    print("\nTesting prime_ml_classifier.py with CSV input...")
    
    # Determine the directory containing the test script
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.getcwd()
    
    # Generate a small test dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use generate_dataset.py to create a small dataset
        result = subprocess.run(
            [sys.executable, 'generate_dataset.py', 
             '--primes', '20', '--non-primes', '20', '--output', tmp_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"
        assert os.path.exists(tmp_path), "Dataset file should exist"
        
        # Now test prime_ml_classifier.py with the CSV input
        result = subprocess.run(
            [sys.executable, 'prime_ml_classifier.py', '--input', tmp_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        assert result.returncode == 0, f"Classifier failed: {result.stderr}"
        assert "Loading dataset from" in result.stdout, "Should indicate loading from CSV"
        assert "✓ Dataset loaded successfully" in result.stdout, "Should successfully load dataset"
        assert "Dataset shape: (40, 133)" in result.stdout, "Should have 40 samples with 133 columns"
        assert "Best Model:" in result.stdout, "Should train and select a model"
        
        print("✓ prime_ml_classifier.py with CSV input tests passed")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    """Run all tests."""
    print("="*60)
    print("Running Prime ML Classifier Tests")
    print("="*60)
    
    try:
        test_is_prime()
        test_number_to_features()
        test_prepare_features()
        test_generate_prime_numbers()
        test_generate_non_prime_numbers()
        test_number_endings()
        test_load_dataset_from_csv()
        test_classifier_with_csv_input()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
