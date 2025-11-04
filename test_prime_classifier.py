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
    one_hot_encode_features,
    prepare_features
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
    
    # Test with 7-digit number
    features = number_to_features(1234567)
    assert features['ten_power_0'] == 7, "Ones digit should be 7"
    assert features['ten_power_1'] == 6, "Tens digit should be 6"
    assert features['ten_power_2'] == 5, "Hundreds digit should be 5"
    assert features['ten_power_3'] == 4, "Thousands digit should be 4"
    assert features['ten_power_4'] == 3, "Ten thousands digit should be 3"
    assert features['ten_power_5'] == 2, "Hundred thousands digit should be 2"
    assert features['ten_power_6'] == 1, "Millions digit should be 1"
    
    # Test mathematical features
    assert features['sum_digits'] == 28, "Sum of digits should be 1+2+3+4+5+6+7=28"
    assert features['digital_root'] == 1, "Digital root of 28 is 2+8=10, 1+0=1"
    assert features['product_digits'] == 5040, "Product should be 1*2*3*4*5*6*7=5040"
    assert features['last_two_digits'] == 67, "Last two digits should be 67"
    
    # Test with another number
    features2 = number_to_features(9876543)
    assert features2['ten_power_0'] == 3, "Ones digit should be 3"
    assert features2['ten_power_6'] == 9, "Millions digit should be 9"
    assert features2['sum_digits'] == 42, "Sum of digits should be 9+8+7+6+5+4+3=42"
    assert features2['digital_root'] == 6, "Digital root of 42 is 4+2=6"
    
    # Test with number containing zeros
    features3 = number_to_features(1000001)
    assert features3['product_digits'] == 0, "Product should be 0 when any digit is 0"
    assert features3['sum_digits'] == 2, "Sum of digits should be 1+0+0+0+0+0+1=2"
    
    print("✓ number_to_features tests passed")


def test_one_hot_encode_features():
    """Test the one-hot encoding transformer."""
    print("\nTesting one_hot_encode_features function...")
    
    # Test with simple examples
    X = np.array([[7, 6, 5, 4, 3, 2, 1],
                  [0, 1, 2, 3, 4, 5, 6],
                  [9, 9, 9, 9, 9, 9, 9]])
    
    X_encoded = one_hot_encode_features(X)
    
    # Check shape
    assert X_encoded.shape == (3, 70), f"Shape should be (3, 70), got {X_encoded.shape}"
    
    # Check that each row sums to 7 (one feature active per original feature)
    for i in range(len(X)):
        assert X_encoded[i].sum() == 7, f"Row {i} should have exactly 7 ones"
    
    # Check first sample: ten_power_0=7 should have position 7 set to 1
    assert X_encoded[0, 7] == 1, "ten_power_0=7 should set position 7 to 1"
    assert X_encoded[0, 0:7].sum() == 0, "ten_power_0=7 should have positions 0-6 set to 0"
    assert X_encoded[0, 8:10].sum() == 0, "ten_power_0=7 should have positions 8-9 set to 0"
    
    # Check first sample: ten_power_1=6 should have position 10+6=16 set to 1
    assert X_encoded[0, 16] == 1, "ten_power_1=6 should set position 16 to 1"
    
    # Check second sample: ten_power_0=0 should have position 0 set to 1
    assert X_encoded[1, 0] == 1, "ten_power_0=0 should set position 0 to 1"
    
    # Check third sample: all features are 9, so positions 9, 19, 29, ... 69 should be 1
    expected_positions = [9, 19, 29, 39, 49, 59, 69]
    for pos in expected_positions:
        assert X_encoded[2, pos] == 1, f"Position {pos} should be 1 for all 9s"
    
    # Check that encoded features are binary
    assert np.all((X_encoded == 0) | (X_encoded == 1)), "All values should be 0 or 1"
    
    print("✓ one_hot_encode_features tests passed")


def test_generate_prime_numbers():
    """Test prime number generation."""
    print("\nTesting generate_prime_numbers function...")
    
    primes = generate_prime_numbers(10, min_val=1000000, max_val=9999999)
    
    assert len(primes) == 10, "Should generate 10 prime numbers"
    
    # Check all are 7-digit numbers
    for p in primes:
        assert 1000000 <= p <= 9999999, f"{p} should be 7-digit"
        assert is_prime(p), f"{p} should be prime"
    
    # Check uniqueness
    assert len(primes) == len(set(primes)), "All primes should be unique"
    
    print(f"✓ Generated 10 primes: {primes[:5]}...")


def test_generate_non_prime_numbers():
    """Test non-prime number generation."""
    print("\nTesting generate_non_prime_numbers function...")
    
    non_primes = generate_non_prime_numbers(10, min_val=1000000, max_val=9999999)
    
    assert len(non_primes) == 10, "Should generate 10 non-prime numbers"
    
    # Check all are 7-digit numbers and not prime
    for np in non_primes:
        assert 1000000 <= np <= 9999999, f"{np} should be 7-digit"
        assert not is_prime(np), f"{np} should not be prime"
    
    # Check uniqueness
    assert len(non_primes) == len(set(non_primes)), "All non-primes should be unique"
    
    print(f"✓ Generated 10 non-primes: {non_primes[:5]}...")


def test_number_endings():
    """Test that generated numbers don't end in 0, 2, or 5."""
    print("\nTesting that numbers don't end in 0, 2, or 5...")
    
    # Generate numbers
    primes = generate_prime_numbers(20, min_val=1000000, max_val=9999999)
    non_primes = generate_non_prime_numbers(20, min_val=1000000, max_val=9999999)
    
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
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        # Write a small test dataset
        tmp.write("ten_power_0,ten_power_1,ten_power_2,ten_power_3,ten_power_4,ten_power_5,ten_power_6,prime,number\n")
        tmp.write("7,6,5,4,3,2,1,1,1234567\n")
        tmp.write("3,4,5,6,7,8,9,0,9876543\n")
    
    try:
        # Load the dataset
        df = load_dataset_from_csv(tmp_path)
        
        # Verify the data
        assert len(df) == 2, "Should have 2 rows"
        assert df.shape[1] == 9, "Should have 9 columns"
        
        # Check first row
        assert df.iloc[0]['number'] == 1234567, "First number should be 1234567"
        assert df.iloc[0]['prime'] == 1, "First number should be marked as prime"
        assert df.iloc[0]['ten_power_0'] == 7, "First number ones digit should be 7"
        
        # Check second row
        assert df.iloc[1]['number'] == 9876543, "Second number should be 9876543"
        assert df.iloc[1]['prime'] == 0, "Second number should be marked as non-prime"
        assert df.iloc[1]['ten_power_6'] == 9, "Second number millions digit should be 9"
        
        print("✓ load_dataset_from_csv tests passed")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_prepare_features():
    """Test the prepare_features function."""
    print("\nTesting prepare_features function...")
    
    import pandas as pd
    
    # Create a test dataset with mathematical features
    test_data = {
        'ten_power_0': [7, 3, 9],
        'ten_power_1': [6, 4, 8],
        'ten_power_2': [5, 5, 7],
        'ten_power_3': [4, 6, 6],
        'ten_power_4': [3, 7, 5],
        'ten_power_5': [2, 8, 4],
        'ten_power_6': [1, 9, 3],
        'sum_digits': [28, 42, 42],
        'digital_root': [1, 6, 6],
        'product_digits': [5040, 1451520, 1451520],
        'last_two_digits': [67, 43, 89],
        'prime': [1, 0, 1],
        'number': [1234567, 9876543, 3456789]
    }
    df = pd.DataFrame(test_data)
    
    # Test feature preparation
    X, feature_info = prepare_features(df)
    
    # Check shape - should be 70 one-hot + 4 mathematical features
    assert X.shape == (3, 74), f"Expected shape (3, 74), got {X.shape}"
    assert feature_info['digit_features'] == 70, "Should have 70 digit features"
    assert feature_info['math_features'] == 4, "Should have 4 mathematical features"
    assert feature_info['total_features'] == 74, "Should have 74 total features"
    
    # Check that scaler was created
    assert feature_info['scaler'] is not None, "Scaler should be created"
    assert len(feature_info['math_feature_names']) == 4, "Should have 4 math feature names"
    
    # Test with dataset without mathematical features (backward compatibility)
    df_old = df[['ten_power_0', 'ten_power_1', 'ten_power_2', 'ten_power_3',
                 'ten_power_4', 'ten_power_5', 'ten_power_6', 'prime', 'number']]
    X_old, feature_info_old = prepare_features(df_old)
    
    assert X_old.shape == (3, 70), f"Expected shape (3, 70) for old format, got {X_old.shape}"
    assert feature_info_old['digit_features'] == 70, "Should have 70 digit features"
    assert feature_info_old['math_features'] == 0, "Should have 0 mathematical features"
    assert feature_info_old['scaler'] is None, "Scaler should be None for old format"
    
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
        assert "Dataset shape: (40, 13)" in result.stdout, "Should have 40 samples with 13 columns"
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
        test_one_hot_encode_features()
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
