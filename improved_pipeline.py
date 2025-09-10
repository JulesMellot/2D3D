"""
Improved pipeline with temporal smoothing and better depth estimation.
"""

import cv2
import numpy as np
import os
from depth_estimation.improved import DepthEstimator
from stereo_warp.warper import StereoWarper
from inpainting.basic import BasicInpainter
from utils.device import get_device

class ImprovedPipeline:
    def __init__(self, depth_profile="balanced", baseline=0.05, focal_length=1000):
        """
        Initialize the improved 2D to 3D conversion pipeline.
        
        Args:
            depth_profile (str): Depth estimation profile ("fast", "balanced", "precision")
            baseline (float): Stereo baseline distance
            focal_length (float): Camera focal length
        """
        self.depth_estimator = DepthEstimator(profile=depth_profile)
        self.stereo_warper = StereoWarper(baseline=baseline, focal_length=focal_length)
        self.inpainter = BasicInpainter(method="telea")
        self.device = get_device()
        
        # For temporal smoothing
        self.previous_depth = None
        self.smoothing_factor = 0.3  # Adjust this value (0.0 to 1.0) for more/less smoothing
        
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image to create a stereo pair.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save output (default: same as input)
            
        Returns:
            tuple: Paths to left and right images
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Estimate depth
        print("Estimating depth...")
        depth_map = self.depth_estimator.predict_depth(image)
        
        # Apply temporal smoothing if we have a previous frame
        if self.previous_depth is not None:
            depth_map = self._apply_temporal_smoothing(depth_map, self.previous_depth)
            
        # Store current depth for next frame
        self.previous_depth = depth_map.copy()
        
        # Create stereo pair
        print("Generating stereo pair...")
        left_image, right_image = self.stereo_warper.create_stereo_pair(image, depth_map)
        
        # Fill occlusions (basic approach)
        print("Filling occlusions...")
        # This is a simplified approach - in practice, you'd compute actual occlusion masks
        # between left and right images
        right_image = self.inpainter.inpaint(right_image, np.zeros_like(depth_map, dtype=np.uint8))
        
        # Apply post-processing for crisper images
        left_image = self._post_process_image(left_image)
        right_image = self._post_process_image(right_image)
        
        # Save results
        if output_dir is None:
            output_dir = os.path.dirname(image_path) or "."
            
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        left_path = os.path.join(output_dir, f"{base_name}_left.png")
        right_path = os.path.join(output_dir, f"{base_name}_right.png")
        stereo_path = os.path.join(output_dir, f"{base_name}_stereo_sbs.png")
        anaglyph_path = os.path.join(output_dir, f"{base_name}_anaglyph.png")
        
        cv2.imwrite(left_path, left_image)
        cv2.imwrite(right_path, right_image)
        
        # Create side-by-side stereo image
        sbs_image = self.stereo_warper.create_side_by_side(left_image, right_image)
        cv2.imwrite(stereo_path, sbs_image)
        
        # Create anaglyph
        anaglyph_image = self.stereo_warper.create_anaglyph(left_image, right_image)
        cv2.imwrite(anaglyph_path, anaglyph_image)
        
        print(f"Results saved to:")
        print(f"  Left view: {left_path}")
        print(f"  Right view: {right_path}")
        print(f"  Side-by-side: {stereo_path}")
        print(f"  Anaglyph: {anaglyph_path}")
        
        return left_path, right_path
    
    def process_frame(self, frame):
        """
        Process a single video frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (left_frame, right_frame)
        """
        # Estimate depth
        depth_map = self.depth_estimator.predict_depth(frame)
        
        # Apply temporal smoothing if we have a previous frame
        if self.previous_depth is not None:
            depth_map = self._apply_temporal_smoothing(depth_map, self.previous_depth)
            
        # Store current depth for next frame
        self.previous_depth = depth_map.copy()
        
        # Create stereo pair
        left_frame, right_frame = self.stereo_warper.create_stereo_pair(frame, depth_map)
        
        # Apply post-processing for crisper images
        left_frame = self._post_process_image(left_frame)
        right_frame = self._post_process_image(right_frame)
        
        return left_frame, right_frame
    
    def _apply_temporal_smoothing(self, current_depth, previous_depth):
        """
        Apply temporal smoothing to reduce flickering artifacts.
        
        Args:
            current_depth (np.ndarray): Current depth map
            previous_depth (np.ndarray): Previous depth map
            
        Returns:
            np.ndarray: Smoothed depth map
        """
        # Simple exponential smoothing
        smoothed = (1 - self.smoothing_factor) * previous_depth + self.smoothing_factor * current_depth
        return smoothed
    
    def _post_process_image(self, image):
        """
        Apply post-processing to make images crisper.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Processed image
        """
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Apply slight contrast enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

# Example usage function
def convert_image_to_stereo(image_path, output_dir=None, profile="balanced"):
    """
    Convert a single image to stereo 3D.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save output
        profile (str): Depth estimation profile
        
    Returns:
        tuple: Paths to generated stereo images
    """
    pipeline = ImprovedPipeline(depth_profile=profile)
    return pipeline.process_image(image_path, output_dir)