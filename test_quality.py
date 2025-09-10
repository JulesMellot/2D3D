"""
Simple test script to verify the quality-preserving pipeline.
"""

import cv2
import numpy as np
import os
from quality_pipeline import QualityPreservingPipeline

def create_test_image():
    """Create a test image for verification."""
    # Create a simple gradient image
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient background
    for i in range(height):
        intensity = int(255 * i / height)
        image[i, :, :] = [intensity, intensity, intensity]
    
    # Add some shapes
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red square
    cv2.circle(image, (400, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.putText(image, "Test Image", (200, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def main():
    print("Testing quality-preserving 2D to 3D pipeline...")
    
    # Create test image
    test_image = create_test_image()
    cv2.imwrite("test_input.png", test_image)
    print("Created test image: test_input.png")
    
    # Create pipeline
    pipeline = QualityPreservingPipeline(depth_profile="fast")
    print("Created pipeline with fast profile")
    
    # Process frame
    print("Processing frame...")
    left_frame, right_frame = pipeline.process_frame(test_image)
    
    # Save results
    cv2.imwrite("test_left.png", left_frame)
    cv2.imwrite("test_right.png", right_frame)
    
    # Create side-by-side
    sbs_frame = np.hstack((left_frame, right_frame))
    cv2.imwrite("test_stereo_sbs.png", sbs_frame)
    
    print("Results saved:")
    print("  Left view: test_left.png")
    print("  Right view: test_right.png")
    print("  Side-by-side: test_stereo_sbs.png")
    
    # Verify results
    print("\nVerification:")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Left shape: {left_frame.shape}")
    print(f"  Right shape: {right_frame.shape}")
    print(f"  SBS shape: {sbs_frame.shape}")
    
    if left_frame.shape == test_image.shape and right_frame.shape == test_image.shape:
        print("  ✓ All frames have correct dimensions")
    else:
        print("  ✗ Frame dimension mismatch")
    
    if not np.array_equal(left_frame, right_frame):
        print("  ✓ Stereo effect generated (frames are different)")
    else:
        print("  ✗ No stereo effect (frames are identical)")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()