"""
Test script to verify the 2D3D pipeline implementation.
"""

import os
import sys
import tempfile
import shutil
from sample_pipeline import main as sample_main

def test_pipeline():
    """Test the end-to-end pipeline."""
    print("Testing 2D3D pipeline...")
    
    # Change to a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    
    try:
        os.chdir(test_dir)
        
        # Run the sample pipeline
        result = sample_main()
        
        # Check if files were created
        expected_files = [
            "sample_generated.png",
            "sample_generated_left.png",
            "sample_generated_right.png",
            "sample_generated_stereo_sbs.png",
            "sample_generated_anaglyph.png"
        ]
        
        missing_files = []
        for filename in expected_files:
            if not os.path.exists(filename):
                missing_files.append(filename)
                
        if missing_files:
            print(f"ERROR: Missing expected files: {missing_files}")
            return False
        else:
            print("SUCCESS: All expected files were created")
            return True
            
    except Exception as e:
        print(f"ERROR: Exception during testing: {e}")
        return False
    finally:
        # Clean up
        os.chdir(original_dir)
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)