"""
Stereo view synthesis through image warping based on depth maps.
Supports occlusion handling and multiple stereo formats.
"""

import cv2
import numpy as np
from scipy import ndimage

class StereoWarper:
    def __init__(self, baseline=0.1, focal_length=1000):
        """
        Initialize stereo warper.
        
        Args:
            baseline (float): Distance between left and right cameras
            focal_length (float): Focal length of the camera
        """
        self.baseline = baseline
        self.focal_length = focal_length
        
    def create_stereo_pair(self, image, depth_map, shift_direction="horizontal"):
        """
        Create a stereo pair from a single image and depth map.
        
        Args:
            image (np.ndarray): Input image (H, W, 3)
            depth_map (np.ndarray): Depth map (H, W) normalized to [0, 1]
            shift_direction (str): Direction of shift ("horizontal" or "vertical")
            
        Returns:
            tuple: (left_image, right_image) - both same size as input
        """
        # Normalize depth to disparity range
        disparity_map = depth_map * self.baseline * self.focal_length
        
        # Create left and right images
        left_image = image.copy()
        right_image = np.zeros_like(image)
        
        # Generate coordinate grids
        h, w = image.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate shifted coordinates based on disparity
        if shift_direction == "horizontal":
            # Right view: shift pixels to the left based on depth
            x_shifted = np.clip(x_coords - disparity_map, 0, w - 1).astype(int)
            y_shifted = y_coords
            
            # Create right image by shifting pixels
            for c in range(image.shape[2]):
                right_image[y_coords, x_coords, c] = image[y_shifted, x_shifted, c]
        else:
            # For vertical stereo (less common)
            x_shifted = x_coords
            y_shifted = np.clip(y_coords - disparity_map, 0, h - 1).astype(int)
            
            for c in range(image.shape[2]):
                right_image[y_coords, x_coords, c] = image[y_shifted, x_shifted, c]
        
        return left_image, right_image
    
    def create_anaglyph(self, left_image, right_image):
        """
        Create an anaglyph (red-cyan) stereo image.
        
        Args:
            left_image (np.ndarray): Left view image
            right_image (np.ndarray): Right view image
            
        Returns:
            np.ndarray: Anaglyph image
        """
        # Red channel from left image, blue/green channels from right image
        anaglyph = np.zeros_like(left_image)
        anaglyph[:, :, 0] = left_image[:, :, 0]  # Red channel
        anaglyph[:, :, 1] = right_image[:, :, 1]  # Green channel
        anaglyph[:, :, 2] = right_image[:, :, 2]  # Blue channel
        
        return anaglyph
    
    def create_side_by_side(self, left_image, right_image):
        """
        Create a side-by-side stereo image.
        
        Args:
            left_image (np.ndarray): Left view image
            right_image (np.ndarray): Right view image
            
        Returns:
            np.ndarray: Side-by-side image
        """
        return np.hstack((left_image, right_image))
    
    def create_top_bottom(self, left_image, right_image):
        """
        Create a top-bottom stereo image.
        
        Args:
            left_image (np.ndarray): Left view image
            right_image (np.ndarray): Right view image
            
        Returns:
            np.ndarray: Top-bottom image
        """
        return np.vstack((left_image, right_image))
    
    def fill_occlusions(self, image, depth_map, method="inpaint_ns"):
        """
        Fill occluded regions in stereo images.
        
        Args:
            image (np.ndarray): Image with occlusions
            depth_map (np.ndarray): Depth map
            method (str): Inpainting method ("inpaint_ns" or "inpaint_telea")
            
        Returns:
            np.ndarray: Image with filled occlusions
        """
        # Create mask for occluded regions
        # This is a simplified approach - in practice, you'd compute actual occlusions
        # from the disparity between left and right images
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Simple occlusion detection based on depth discontinuities
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold for detecting depth discontinuities
        threshold = np.mean(grad_mag) + 2 * np.std(grad_mag)
        mask[grad_mag > threshold] = 255
        
        # Apply inpainting
        if method == "inpaint_ns":
            filled_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        elif method == "inpaint_telea":
            filled_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:
            filled_image = image.copy()
            
        return filled_image