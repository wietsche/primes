"""
Simple test to verify the prime ML classifier functions work correctly.
"""

import sys
import os
import tempfile
import subprocess
from prime_ml_classifier import (
    is_prime, 
    number_to_features, 
    generate_prime_numbers,
    generate_non_prime_numbers,
    load_dataset_from_csv
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
    
    # Test with another number
    features2 = number_to_features(9876543)
    assert features2['ten_power_0'] == 3, "Ones digit should be 3"
    assert features2['ten_power_6'] == 9, "Millions digit should be 9"
    
    print("✓ number_to_features tests passed")


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
        assert "Dataset shape: (40, 9)" in result.stdout, "Should have 40 samples"
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
        test_generate_prime_numbers()
        test_generate_non_prime_numbers()
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
