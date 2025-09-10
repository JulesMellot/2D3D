"""
Sample script demonstrating end-to-end conversion of an image to stereo 3D.
This script shows how to use the depth estimation and stereo warping modules.
"""

import cv2
import numpy as np
import urllib.request
import os
from pipeline import Pipeline

def download_sample_image():
    """Download a sample image for testing."""
    # Try a different image source
    url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80"
    filename = "sample.jpg"
    
    if not os.path.exists(filename):
        try:
            print("Downloading sample image...")
            urllib.request.urlretrieve(url, filename)
            print(f"Sample image saved as {filename}")
        except Exception as e:
            print(f"Could not download sample image: {e}")
            return None
    
    return filename

def create_sample_image():
    """Create a simple sample image if download fails."""
    # Create a simple gradient image as fallback (larger size)
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient background
    for i in range(height):
        intensity = int(255 * i / height)
        image[i, :, :] = [intensity, intensity, intensity]
    
    # Add some shapes
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red square
    cv2.circle(image, (400, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.putText(image, "Sample Image", (200, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    filename = "sample_generated.png"
    cv2.imwrite(filename, image)
    print(f"Generated sample image saved as {filename}")
    return filename

def main():
    print("2D3D Sample Pipeline Demo")
    print("=" * 30)
    
    # Try to download a sample image, or create one
    image_path = download_sample_image()
    if image_path is None or not os.path.exists(image_path):
        print("Creating a generated sample image...")
        image_path = create_sample_image()
    
    # Verify image exists and is valid
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return 1
        
    # Load image to check dimensions
    test_image = cv2.imread(image_path)
    if test_image is None:
        print(f"Error: Could not load image at {image_path}")
        return 1
        
    print(f"Loaded image with dimensions: {test_image.shape}")
    
    # Create pipeline with balanced profile
    print("\nInitializing pipeline...")
    try:
        pipeline = Pipeline(depth_profile="fast")  # Use fast profile for testing
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Make sure you have installed all dependencies correctly.")
        return 1
    
    # Process the image
    print("Processing image...")
    try:
        left_path, right_path = pipeline.process_image(image_path)
        print("Processing completed successfully!")
        print(f"Left view saved to: {left_path}")
        print(f"Right view saved to: {right_path}")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())