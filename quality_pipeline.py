"""
Quality-preserving 2D to 3D conversion pipeline.
"""

import cv2
import numpy as np
import os
import tempfile
import shutil
from utils.device import get_device

class QualityPreservingPipeline:
    def __init__(self, depth_profile="balanced", baseline=0.05, focal_length=1000):
        """
        Initialize the quality-preserving 2D to 3D conversion pipeline.
        
        Args:
            depth_profile (str): Depth estimation profile ("fast", "balanced", "precision")
            baseline (float): Stereo baseline distance
            focal_length (float): Camera focal length
        """
        self.depth_profile = depth_profile
        self.baseline = baseline
        self.focal_length = focal_length
        self.device = get_device()
        
        # Quality preservation settings
        self.preserve_colors = True
        self.temporal_smoothing = True
        self.post_process = False
        
        # For temporal smoothing
        self.previous_depth = None
        self.smoothing_factor = 0.1  # Low smoothing to preserve details
        
        # Initialize depth estimator (using improved method that works)
        from depth_estimation.improved import DepthEstimator
        self.depth_estimator = DepthEstimator(profile=self.depth_profile)
        self.depth_method = "improved"
    
    def convert_video(self, input_path, output_path, format="sbs", max_dimension=None):
        """
        Convert a 2D video to stereoscopic 3D while preserving quality.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            format (str): Stereo format ("sbs", "tb", "anaglyph")
            max_dimension (int): Maximum dimension for processing (None for original size)
        """
        print(f"Converting {input_path} to {format} stereo format")
        print(f"Depth method: {self.depth_method}, Profile: {self.depth_profile}")
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        processed_dir = os.path.join(temp_dir, "processed")
        
        try:
            # Extract frames
            print("Extracting frames...")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Extract frames with original quality
            os.system(f"ffmpeg -i {input_path} -vf 'fps=24' {frames_dir}/frame_%05d.png")
            
            # Count frames
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            print(f"Extracted {len(frame_files)} frames")
            
            if not frame_files:
                raise ValueError("No frames extracted from video")
            
            # Process frames
            print("Processing frames...")
            for i, frame_file in enumerate(frame_files):
                print(f"Processing frame {i+1}/{len(frame_files)}")
                
                # Load frame
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"Warning: Could not load frame {frame_file}")
                    continue
                
                # Resize frame if necessary
                if max_dimension:
                    height, width = frame.shape[:2]
                    if max(height, width) > max_dimension:
                        scale = max_dimension / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        print(f"  Resized frame to {new_width}x{new_height}")
                
                # Process frame
                try:
                    left_frame, right_frame = self.process_frame(frame)
                    
                    # Save processed frames
                    left_path = os.path.join(processed_dir, f"left_{frame_file}")
                    right_path = os.path.join(processed_dir, f"right_{frame_file}")
                    cv2.imwrite(left_path, left_frame)
                    cv2.imwrite(right_path, right_frame)
                except Exception as e:
                    print(f"Error processing frame {frame_file}: {e}")
                    # Copy original frame as fallback
                    shutil.copy(frame_path, os.path.join(processed_dir, f"left_{frame_file}"))
                    shutil.copy(frame_path, os.path.join(processed_dir, f"right_{frame_file}"))
            
            # Get video properties
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            # Create stereo video with preserved audio
            print("Creating stereo video with preserved audio...")
            self._create_stereo_video(processed_dir, input_path, output_path, fps, format)
            
            print(f"Video conversion completed! Output saved to: {output_path}")
            
        except Exception as e:
            print(f"Error converting video: {e}")
            raise
        finally:
            # Clean up temporary directories
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def process_frame(self, frame):
        """
        Process a single frame while preserving quality.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (left_frame, right_frame)
        """
        # Store original frame for quality preservation
        original_frame = frame.copy()
        
        # Estimate depth
        depth_map = self.depth_estimator.predict_depth(frame)
        
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing and self.previous_depth is not None:
            depth_map = self._apply_temporal_smoothing(depth_map, self.previous_depth)
            
        # Store current depth for next frame
        self.previous_depth = depth_map.copy()
        
        # Create stereo pair
        left_frame, right_frame = self._create_stereo_pair(original_frame, depth_map)
        
        # Preserve original colors if enabled
        if self.preserve_colors:
            left_frame = self._preserve_colors(original_frame, left_frame)
            right_frame = self._preserve_colors(original_frame, right_frame)
        
        # Apply light post-processing if enabled
        if self.post_process:
            left_frame = self._post_process_image(left_frame)
            right_frame = self._post_process_image(right_frame)
        
        return left_frame, right_frame
    
    def _create_stereo_pair(self, frame, depth_map):
        """
        Create stereo pair from frame and depth map.
        
        Args:
            frame (np.ndarray): Original frame
            depth_map (np.ndarray): Depth map
            
        Returns:
            tuple: (left_frame, right_frame)
        """
        # Normalize depth to disparity range
        disparity_map = depth_map * self.baseline * self.focal_length
        
        # Create left and right images
        left_image = frame.copy()
        right_image = np.zeros_like(frame)
        
        # Generate coordinate grids
        h, w = frame.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate shifted coordinates based on disparity
        x_shifted = np.clip(x_coords - disparity_map, 0, w - 1).astype(int)
        y_shifted = y_coords
        
        # Create right image by shifting pixels
        for c in range(frame.shape[2]):
            right_image[y_coords, x_coords, c] = frame[y_shifted, x_shifted, c]
        
        return left_image, right_image
    
    def _apply_temporal_smoothing(self, current_depth, previous_depth):
        """
        Apply light temporal smoothing to reduce flickering.
        
        Args:
            current_depth (np.ndarray): Current depth map
            previous_depth (np.ndarray): Previous depth map
            
        Returns:
            np.ndarray: Smoothed depth map
        """
        # Very light smoothing to preserve details
        smoothed = (1 - self.smoothing_factor) * previous_depth + self.smoothing_factor * current_depth
        return smoothed
    
    def _preserve_colors(self, original, processed):
        """
        Preserve original colors in processed frame.
        
        Args:
            original (np.ndarray): Original frame
            processed (np.ndarray): Processed frame
            
        Returns:
            np.ndarray: Color-preserved frame
        """
        # Convert to LAB color space
        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        processed_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        
        # Replace L channel of processed with original L channel
        processed_lab[:, :, 0] = original_lab[:, :, 0]
        
        # Convert back to BGR
        result = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
        return result
    
    def _post_process_image(self, image):
        """
        Apply light post-processing to enhance image quality.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Processed image
        """
        # Very light denoising
        denoised = cv2.bilateralFilter(image, 5, 25, 25)
        
        # Very light sharpening
        kernel = np.array([[0, -0.5, 0], 
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _create_stereo_video(self, processed_dir, input_path, output_path, fps, format):
        """
        Create stereo video with preserved audio.
        
        Args:
            processed_dir (str): Directory with processed frames
            input_path (str): Path to input video
            output_path (str): Path to output video
            fps (int): Frames per second
            format (str): Stereo format
        """
        left_pattern = f"{processed_dir}/left_frame_%05d.png"
        right_pattern = f"{processed_dir}/right_frame_%05d.png"
        
        # Build FFmpeg command based on format
        if format == "sbs":  # Side-by-side
            filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
        elif format == "tb":  # Top-bottom
            filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
        elif format == "anaglyph":  # Red-cyan anaglyph
            filter_complex = "[0:v]format=yuv444p,lutyuv=y=gammaval(0.5)[left];[1:v]format=yuv444p,lutyuv=y=gammaval(0.5)[right];[left][right]anaglyph[v]"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Use ffmpeg to create stereo video with audio preserved
        cmd = (f"ffmpeg -y -framerate {fps} -i {left_pattern} "
               f"-framerate {fps} -i {right_pattern} "
               f"-i {input_path} "
               f"-filter_complex '{filter_complex}' "
               f"-map '[v]' -map 2:a? "
               f"-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "
               f"-c:a copy {output_path}")
        
        os.system(cmd)

# Convenience functions
def convert_video_to_stereo(input_path, output_path, profile="balanced", format="sbs"):
    """
    Convert a video to stereo 3D with quality preservation.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        profile (str): Depth estimation profile
        format (str): Stereo format
        
    Returns:
        str: Path to output video
    """
    pipeline = QualityPreservingPipeline(depth_profile=profile)
    pipeline.convert_video(input_path, output_path, format=format)
    return output_path