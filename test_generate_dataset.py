"""
Tests for the generate_dataset.py script.
"""

import os
import sys
import tempfile
import pandas as pd
import subprocess


def test_help_message():
    """Test that help message works."""
    print("Testing help message...")
    result = subprocess.run(
        [sys.executable, 'generate_dataset.py', '--help'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Help should exit with code 0"
    assert 'Generate a dataset' in result.stdout, "Help should contain description"
    assert '--primes' in result.stdout, "Help should document --primes argument"
    assert '--non-primes' in result.stdout, "Help should document --non-primes argument"
    print("✓ Help message test passed")


def test_default_generation():
    """Test default dataset generation."""
    print("\nTesting default generation (100+100)...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'generate_dataset.py', '--output', tmp_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Generation failed: {result.stderr}"
        assert os.path.exists(tmp_path), "Output file should exist"
        
        # Verify CSV content
        df = pd.read_csv(tmp_path)
        assert df.shape[0] == 200, "Should have 200 samples"
        assert df.shape[1] == 9, "Should have 9 columns"
        
        # Check column names
        expected_cols = ['ten_power_0', 'ten_power_1', 'ten_power_2', 'ten_power_3',
                         'ten_power_4', 'ten_power_5', 'ten_power_6', 'prime', 'number']
        assert list(df.columns) == expected_cols, "Should have correct columns"
        
        # Check prime distribution
        prime_count = df['prime'].sum()
        assert prime_count == 100, "Should have 100 primes"
        assert len(df) - prime_count == 100, "Should have 100 non-primes"
        
        print(f"✓ Generated {len(df)} samples successfully")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_custom_counts():
    """Test with custom prime and non-prime counts."""
    print("\nTesting custom counts (50+75)...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'generate_dataset.py', 
             '--primes', '50', '--non-primes', '75', '--output', tmp_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Generation failed: {result.stderr}"
        
        df = pd.read_csv(tmp_path)
        assert df.shape[0] == 125, "Should have 125 samples (50+75)"
        
        prime_count = df['prime'].sum()
        assert prime_count == 50, "Should have 50 primes"
        assert len(df) - prime_count == 75, "Should have 75 non-primes"
        
        print(f"✓ Generated {len(df)} samples with custom counts")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_invalid_arguments():
    """Test that invalid arguments are rejected."""
    print("\nTesting invalid arguments...")
    
    # Test zero primes
    result = subprocess.run(
        [sys.executable, 'generate_dataset.py', '--primes', '0'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    assert result.returncode != 0, "Should fail with zero primes"
    
    # Test zero non-primes
    result = subprocess.run(
        [sys.executable, 'generate_dataset.py', '--non-primes', '0'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    assert result.returncode != 0, "Should fail with zero non-primes"
    
    print("✓ Invalid argument tests passed")


def test_data_integrity():
    """Test that generated data has correct digit features."""
    print("\nTesting data integrity...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'generate_dataset.py', 
             '--primes', '10', '--non-primes', '10', '--output', tmp_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, "Generation should succeed"
        
        df = pd.read_csv(tmp_path)
        
        # Verify digit extraction for each sample
        for idx, row in df.iterrows():
            number = row['number']
            digits = str(number).zfill(7)
            
            # Verify each digit position
            for i in range(7):
                expected_digit = int(digits[6 - i])
                actual_digit = row[f'ten_power_{i}']
                assert expected_digit == actual_digit, \
                    f"Digit mismatch for number {number} at position {i}"
            
            # Verify number is 7-digit
            assert 1000000 <= number <= 9999999, f"{number} should be 7-digit"
        
        print("✓ Data integrity verified for all samples")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    """Run all tests."""
    print("="*60)
    print("Running Generate Dataset Script Tests")
    print("="*60)
    
    try:
        test_help_message()
        test_default_generation()
        test_custom_counts()
        test_invalid_arguments()
        test_data_integrity()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
