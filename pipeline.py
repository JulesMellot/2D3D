"""
Main pipeline for 2D to 3D conversion.
Integrates depth estimation, stereo warping, and inpainting.
"""

import cv2
import numpy as np
import os
from depth_estimation.simple import DepthEstimator
from stereo_warp.warper import StereoWarper
from inpainting.basic import BasicInpainter
from utils.device import get_device

class Pipeline:
    def __init__(self, depth_profile="balanced", baseline=0.05, focal_length=1000):
        """
        Initialize the 2D to 3D conversion pipeline.
        
        Args:
            depth_profile (str): Depth estimation profile ("fast", "balanced", "precision")
            baseline (float): Stereo baseline distance
            focal_length (float): Camera focal length
        """
        self.depth_estimator = DepthEstimator(profile=depth_profile)
        self.stereo_warper = StereoWarper(baseline=baseline, focal_length=focal_length)
        self.inpainter = BasicInpainter(method="telea")
        self.device = get_device()
        
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
        
        # Create stereo pair
        print("Generating stereo pair...")
        left_image, right_image = self.stereo_warper.create_stereo_pair(image, depth_map)
        
        # Fill occlusions (basic approach)
        print("Filling occlusions...")
        # This is a simplified approach - in practice, you'd compute actual occlusion masks
        # between left and right images
        right_image = self.inpainter.inpaint(right_image, np.zeros_like(depth_map, dtype=np.uint8))
        
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
        
        # Create stereo pair
        left_frame, right_frame = self.stereo_warper.create_stereo_pair(frame, depth_map)
        
        # In a full implementation, you would:
        # 1. Compute proper occlusion masks
        # 2. Apply temporal smoothing
        # 3. Use more advanced inpainting
        
        return left_frame, right_frame

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
    pipeline = Pipeline(depth_profile=profile)
    return pipeline.process_image(image_path, output_dir)