"""
Simple test to verify the PCA analysis script works correctly.
"""

import sys
import os
import tempfile
import subprocess


def get_script_dir():
    """Helper function to determine the script directory."""
    if '__file__' in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def test_pca_analysis_default():
    """Test PCA analysis with default parameters."""
    print("Testing PCA analysis with default parameters...")
    
    script_dir = get_script_dir()
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_output = tmp.name
    
    try:
        # Run PCA analysis with the default dataset
        result = subprocess.run(
            [sys.executable, 'pca_analysis.py', '--output', tmp_output],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"PCA analysis failed: {result.stderr}"
        assert os.path.exists(tmp_output), "Output file should exist"
        assert os.path.getsize(tmp_output) > 0, "Output file should not be empty"
        
        # Check expected output messages
        assert "Loading dataset..." in result.stdout, "Should indicate loading dataset"
        assert "✓ Dataset loaded successfully" in result.stdout, "Should successfully load dataset"
        assert "Performing PCA" in result.stdout, "Should perform PCA"
        assert "✓ PCA completed" in result.stdout, "Should complete PCA"
        assert "Creating side-by-side PCA visualizations" in result.stdout, "Should create visualizations"
        assert "✓ Visualization completed" in result.stdout, "Should complete visualization"
        assert "PCA visualization saved" in result.stdout, "Should save visualization"
        
        print(f"✓ PCA analysis with default parameters test passed")
        
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


def test_pca_analysis_custom_dataset():
    """Test PCA analysis with a custom dataset."""
    print("\nTesting PCA analysis with custom dataset...")
    
    script_dir = get_script_dir()
    
    # Create a temporary dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_data:
        tmp_dataset = tmp_data.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_out:
        tmp_output = tmp_out.name
    
    try:
        # Generate a small test dataset
        result = subprocess.run(
            [sys.executable, 'generate_dataset.py', 
             '--primes', '30', '--non-primes', '30', '--output', tmp_dataset],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"
        assert os.path.exists(tmp_dataset), "Dataset file should exist"
        
        # Run PCA analysis with the custom dataset
        result = subprocess.run(
            [sys.executable, 'pca_analysis.py', 
             '--input', tmp_dataset, '--output', tmp_output],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"PCA analysis failed: {result.stderr}"
        assert os.path.exists(tmp_output), "Output file should exist"
        assert os.path.getsize(tmp_output) > 0, "Output file should not be empty"
        
        # Check that it used the custom dataset
        assert "Dataset shape: (60, 9)" in result.stdout, "Should load 60 samples"
        assert "Primes: 30, Non-primes: 30" in result.stdout, "Should have correct counts"
        
        print(f"✓ PCA analysis with custom dataset test passed")
        
    finally:
        if os.path.exists(tmp_dataset):
            os.remove(tmp_dataset)
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


def test_pca_analysis_missing_file():
    """Test PCA analysis with missing input file."""
    print("\nTesting PCA analysis with missing input file...")
    
    script_dir = get_script_dir()
    
    # Run PCA analysis with non-existent file
    result = subprocess.run(
        [sys.executable, 'pca_analysis.py', '--input', 'nonexistent_file.csv'],
        cwd=script_dir,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode != 0, "Should fail with non-existent file"
    assert "not found" in result.stderr or "not found" in result.stdout, "Should indicate file not found"
    
    print(f"✓ PCA analysis with missing file test passed")


def main():
    """Run all tests."""
    print("="*60)
    print("Running PCA Analysis Tests")
    print("="*60)
    
    try:
        test_pca_analysis_default()
        test_pca_analysis_custom_dataset()
        test_pca_analysis_missing_file()
        
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
