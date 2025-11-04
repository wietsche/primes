"""
Tests for the generate_dataset.py script.
"""

import os
import sys
import tempfile
import pandas as pd
import subprocess

# Import constants from prime_ml_classifier to avoid duplication
try:
    from prime_ml_classifier import NUM_DIGITS
except ImportError:
    # Fallback if import fails (shouldn't happen in normal execution)
    NUM_DIGITS = 11

# Calculate expected column count: 
# NUM_DIGITS basic + NUM_DIGITS*10 one-hot + 10 math features + 2 meta (prime, number)
EXPECTED_COLUMNS = NUM_DIGITS + (NUM_DIGITS * 10) + 10 + 2


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
        # Use calculated expected columns based on NUM_DIGITS
        assert df.shape[1] == EXPECTED_COLUMNS, f"Should have {EXPECTED_COLUMNS} columns, got {df.shape[1]}"
        
        # Check that basic digit columns exist
        for i in range(NUM_DIGITS):
            assert f'ten_power_{i}' in df.columns, f"Should have ten_power_{i}"
        
        # Check for one-hot encoded columns
        assert 'ten_power_0_is_0' in df.columns, "Should have one-hot encoded features"
        assert 'ten_power_10_is_9' in df.columns, "Should have one-hot encoded features for all positions"
        
        # Check mathematical features
        assert 'sum_digits' in df.columns, "Should have sum_digits"
        assert 'digital_root' in df.columns, "Should have digital_root"
        assert 'product_digits' in df.columns, "Should have product_digits"
        assert 'last_two_digits' in df.columns, "Should have last_two_digits"
        assert 'alternating_digit_sum' in df.columns, "Should have alternating_digit_sum"
        assert 'mod_2' in df.columns, "Should have mod_2"
        assert 'mod_11' in df.columns, "Should have mod_11"
        
        # Check meta columns
        assert 'prime' in df.columns, "Should have prime label"
        assert 'number' in df.columns, "Should have original number"
        
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
            digits = str(number).zfill(NUM_DIGITS)
            
            # Verify each digit position
            for i in range(NUM_DIGITS):
                expected_digit = int(digits[NUM_DIGITS - 1 - i])
                actual_digit = row[f'ten_power_{i}']
                assert expected_digit == actual_digit, \
                    f"Digit mismatch for number {number} at position {i}"
            
            # Verify number is NUM_DIGITS-digit
            min_val = 10 ** (NUM_DIGITS - 1)
            max_val = (10 ** NUM_DIGITS) - 1
            assert min_val <= number <= max_val, f"{number} should be {NUM_DIGITS}-digit"
        
        print("✓ Data integrity verified for all samples")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_number_endings():
    """Test that generated numbers don't end in 0, 2, or 5."""
    print("\nTesting that generated numbers don't end in 0, 2, or 5...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'generate_dataset.py', 
             '--primes', '30', '--non-primes', '30', '--output', tmp_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, "Generation should succeed"
        
        df = pd.read_csv(tmp_path)
        
        # Check that all numbers end in 1, 3, 7, or 9
        for idx, row in df.iterrows():
            number = row['number']
            last_digit = number % 10
            assert last_digit in [1, 3, 7, 9], \
                f"Number {number} should end in 1, 3, 7, or 9, not {last_digit}"
            assert last_digit not in [0, 2, 5], \
                f"Number {number} should not end in 0, 2, or 5"
            
            # Also verify ten_power_0 matches
            assert row['ten_power_0'] == last_digit, \
                f"ten_power_0 should match last digit for {number}"
        
        print(f"✓ All {len(df)} numbers correctly end in 1, 3, 7, or 9")
        
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
        test_number_endings()
        
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
