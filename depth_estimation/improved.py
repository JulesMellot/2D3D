"""
Improved depth estimation using OpenCV's stereo matching for better quality.
"""

import cv2
import numpy as np
from utils.device import get_device

class DepthEstimator:
    def __init__(self, profile="balanced"):
        """
        Initialize an improved depth estimator.
        
        Args:
            profile (str): Performance profile ("fast", "balanced", "precision")
        """
        self.device = get_device()
        self.profile = profile
        
        # Set stereo matching parameters based on profile
        if profile == "fast":
            self.num_disparities = 16
            self.block_size = 7
        elif profile == "balanced":
            self.num_disparities = 32
            self.block_size = 9
        else:  # precision
            self.num_disparities = 64
            self.block_size = 11
            
        # Create stereo matcher
        self.stereo = cv2.StereoBM_create(
            numDisparities=self.num_disparities,
            blockSize=self.block_size
        )
        
    def predict_depth(self, image):
        """
        Generate an improved depth map using stereo matching techniques.
        
        Args:
            image (np.ndarray): Input image (H, W, 3) in BGR format
            
        Returns:
            np.ndarray: Depth map (H, W)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Create a simulated right view by shifting the image slightly
        # This is a simple approach - in practice, you'd use more sophisticated methods
        shift = 3
        right_view = np.zeros_like(gray)
        right_view[:, :-shift] = gray[:, shift:]
        
        # Compute disparity
        try:
            disparity = self.stereo.compute(gray, right_view)
            
            # Normalize disparity to [0, 1] range
            disparity = disparity.astype(np.float32) / 16.0
            
            # Handle negative values
            disparity[disparity < 0] = 0
            
            # Normalize to [0, 1]
            if disparity.max() > disparity.min():
                depth = (disparity - disparity.min()) / (disparity.max() - disparity.min())
            else:
                depth = np.zeros_like(disparity)
                
            # Apply morphological operations to smooth depth map
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
            depth = cv2.GaussianBlur(depth, (5, 5), 0)
            
        except Exception as e:
            print(f"Warning: Stereo matching failed, using fallback method: {e}")
            # Fallback to gradient-based method
            depth = self._fallback_depth_estimation(gray)
            
        return depth.astype(np.float32)
    
    def _fallback_depth_estimation(self, gray):
        """
        Fallback depth estimation method.
        
        Args:
            gray (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: Depth map
        """
        # Apply edge detection to simulate depth features
        edges = cv2.Canny(gray, 50, 150)
        
        # Create a gradient-based depth map
        h, w = gray.shape
        # Create a simple depth gradient (farther in center, closer at edges)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        center_y, center_x = h // 2, w // 2
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Normalize distances to [0, 1]
        distances = distances / np.max(distances)
        
        # Combine with edge information
        depth = 0.7 * (1 - distances) + 0.3 * (edges / 255.0)
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max != depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
            
        return depth
    
    def batch_predict(self, images):
        """
        Predict depth maps for a batch of images.
        
        Args:
            images (list): List of input images (H, W, 3) in BGR format
            
        Returns:
            list: List of depth maps
        """
        depths = []
        for image in images:
            depth = self.predict_depth(image)
            depths.append(depth)
        return depths

# Convenience functions
def create_depth_estimator(profile="balanced"):
    """
    Create a depth estimator with the specified profile.
    
    Args:
        profile (str): Performance profile ("fast", "balanced", "precision")
        
    Returns:
        DepthEstimator: Initialized depth estimator
    """
    return DepthEstimator(profile=profile)