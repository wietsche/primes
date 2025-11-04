"""
Test for the neural network model evaluation with hyperparameter search.
"""

import sys
import os
import tempfile
import subprocess
import numpy as np


def test_prime_ml_classifier_nn_with_csv():
    """Test that prime_ml_classifier_nn.py works with CSV input."""
    print("\nTesting prime_ml_classifier_nn.py with CSV input...")
    
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
             '--primes', '30', '--non-primes', '30', '--output', tmp_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"
        assert os.path.exists(tmp_path), "Dataset file should exist"
        
        # Now test prime_ml_classifier_nn.py with the CSV input
        # Using cv=2 to make it faster for testing
        result = subprocess.run(
            [sys.executable, 'prime_ml_classifier_nn.py', 
             '--input', tmp_path, '--cv', '2'],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Model evaluation failed: {result.stderr}"
        assert "Loading dataset from" in result.stdout, "Should indicate loading from CSV"
        assert "✓ Dataset loaded successfully" in result.stdout, "Should successfully load dataset"
        # Check for correct number of samples
        assert "60" in result.stdout and "Dataset shape:" in result.stdout, "Should have 60 samples (30 primes + 30 non-primes)"
        assert "Neural Network Hyperparameter Search" in result.stdout, "Should perform hyperparameter search"
        assert "Best Cross-Validation F1 Score:" in result.stdout, "Should report best CV F1 score"
        assert "Best Parameters:" in result.stdout, "Should report best parameters"
        assert "Test F1 Score:" in result.stdout, "Should report test F1 score"
        assert "Extracting features from dataset" in result.stdout, "Should extract full feature set"
        # Check that we're using the full feature set (110 one-hot + 10 math = 120 features)
        assert "120" in result.stdout or "Total features:" in result.stdout, "Should use full feature set"
        
        print("✓ prime_ml_classifier_nn.py with CSV input tests passed")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_prime_ml_classifier_nn_output_file():
    """Test that prime_ml_classifier_nn.py creates output file."""
    print("\nTesting prime_ml_classifier_nn.py output file creation...")
    
    # Determine the directory containing the test script
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.getcwd()
    
    # Generate a small test dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_data:
        tmp_data_path = tmp_data.name
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Use generate_dataset.py to create a small dataset
            result = subprocess.run(
                [sys.executable, 'generate_dataset.py', 
                 '--primes', '20', '--non-primes', '20', '--output', tmp_data_path],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"
            
            # Run prime_ml_classifier_nn.py with custom output directory
            result = subprocess.run(
                [sys.executable, 'prime_ml_classifier_nn.py', 
                 '--input', tmp_data_path, '--output-dir', tmp_dir, '--cv', '2'],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            assert result.returncode == 0, f"Model evaluation failed: {result.stderr}"
            
            # Check that output file was created with new name
            output_file = os.path.join(tmp_dir, 'prime_ml_classifier_nn.png')
            assert os.path.exists(output_file), f"Output file should exist: {output_file}"
            assert os.path.getsize(output_file) > 0, "Output file should not be empty"
            
            print("✓ prime_ml_classifier_nn.py output file creation tests passed")
            
        finally:
            if os.path.exists(tmp_data_path):
                os.remove(tmp_data_path)


def main():
    """Run all tests."""
    print("="*60)
    print("Running Prime ML Classifier NN Tests")
    print("="*60)
    
    try:
        test_prime_ml_classifier_nn_with_csv()
        test_prime_ml_classifier_nn_output_file()
        
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
