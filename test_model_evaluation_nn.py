"""
Test for the neural network model evaluation with hyperparameter search.
"""

import sys
import os
import tempfile
import subprocess
import numpy as np


def test_model_evaluation_nn_with_csv():
    """Test that model_evaluation_nn.py works with CSV input."""
    print("\nTesting model_evaluation_nn.py with CSV input...")
    
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
        
        # Now test model_evaluation_nn.py with the CSV input
        # Using cv=2 to make it faster for testing
        result = subprocess.run(
            [sys.executable, 'model_evaluation_nn.py', 
             '--input', tmp_path, '--cv', '2'],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Model evaluation failed: {result.stderr}"
        assert "Loading dataset from" in result.stdout, "Should indicate loading from CSV"
        assert "✓ Dataset loaded successfully" in result.stdout, "Should successfully load dataset"
        assert "Dataset shape: (60, 9)" in result.stdout, "Should have 60 samples"
        assert "Neural Network Hyperparameter Search" in result.stdout, "Should perform hyperparameter search"
        assert "Best Cross-Validation F1 Score:" in result.stdout, "Should report best CV F1 score"
        assert "Best Parameters:" in result.stdout, "Should report best parameters"
        assert "Test F1 Score:" in result.stdout, "Should report test F1 score"
        assert "Applying one-hot encoding transformation" in result.stdout, "Should use one-hot encoding"
        assert "One-hot encoded features shape: (48, 70)" in result.stdout, "Should have 70 features after one-hot encoding"
        
        print("✓ model_evaluation_nn.py with CSV input tests passed")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_model_evaluation_nn_output_file():
    """Test that model_evaluation_nn.py creates output file."""
    print("\nTesting model_evaluation_nn.py output file creation...")
    
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
            
            # Run model_evaluation_nn.py with custom output directory
            result = subprocess.run(
                [sys.executable, 'model_evaluation_nn.py', 
                 '--input', tmp_data_path, '--output-dir', tmp_dir, '--cv', '2'],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            assert result.returncode == 0, f"Model evaluation failed: {result.stderr}"
            
            # Check that output file was created
            output_file = os.path.join(tmp_dir, 'model_evaluation_nn.png')
            assert os.path.exists(output_file), f"Output file should exist: {output_file}"
            assert os.path.getsize(output_file) > 0, "Output file should not be empty"
            
            print("✓ model_evaluation_nn.py output file creation tests passed")
            
        finally:
            if os.path.exists(tmp_data_path):
                os.remove(tmp_data_path)


def main():
    """Run all tests."""
    print("="*60)
    print("Running Model Evaluation NN Tests")
    print("="*60)
    
    try:
        test_model_evaluation_nn_with_csv()
        test_model_evaluation_nn_output_file()
        
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
