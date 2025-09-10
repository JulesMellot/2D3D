"""
Simple test script for depth estimation
"""

import cv2
import numpy as np
from depth_estimation.midas import DepthEstimator

def main():
    print("Testing depth estimation...")
    
    # Create a simple test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Create depth estimator
        estimator = DepthEstimator(profile="fast")
        print("Depth estimator created successfully")
        
        # Predict depth
        depth = estimator.predict_depth(image)
        print(f"Depth prediction successful. Shape: {depth.shape}")
        print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()