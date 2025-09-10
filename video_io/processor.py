"""
Video processing pipeline using FFmpeg.
Handles video decoding/encoding while preserving audio and metadata.
"""

import subprocess
import os
import cv2
import tempfile
import shutil
from pathlib import Path

class VideoProcessor:
    def __init__(self, input_path, output_path):
        """
        Initialize video processor.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
        """
        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = None
        
    def extract_frames(self, output_dir=None, fps=None):
        """
        Extract frames from video using FFmpeg.
        
        Args:
            output_dir (str): Directory to save frames (default: temporary directory)
            fps (float): Target FPS for extraction (default: original)
            
        Returns:
            str: Path to directory containing extracted frames
        """
        if output_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            output_dir = self.temp_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-i", self.input_path]
        
        if fps is not None:
            cmd.extend(["-vf", f"fps={fps}"])
            
        cmd.extend([
            "-start_number", "0",
            f"{output_dir}/frame_%06d.png"
        ])
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return output_dir
    
    def create_video_from_frames(self, frame_dir, fps=30, format="sbs"):
        """
        Create video from processed frames using FFmpeg.
        
        Args:
            frame_dir (str): Directory containing processed frames
            fps (int): FPS for output video
            format (str): Output format ("sbs", "tb", "anaglyph")
        """
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-i", f"{frame_dir}/frame_%06d.png",
            "-i", self.input_path,  # Use original video as second input for audio
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",  # Copy audio stream without re-encoding
            self.output_path
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def create_stereo_video(self, left_frame_dir, right_frame_dir, fps=30, format="sbs"):
        """
        Create stereo video from left and right frame directories.
        
        Args:
            left_frame_dir (str): Directory containing left view frames
            right_frame_dir (str): Directory containing right view frames
            fps (int): FPS for output video
            format (str): Stereo format ("sbs", "tb", "anaglyph")
        """
        # Build FFmpeg command based on format
        if format == "sbs":  # Side-by-side
            filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
        elif format == "tb":  # Top-bottom
            filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
        elif format == "anaglyph":  # Red-cyan anaglyph
            filter_complex = "[0:v]format=yuv444p,lutyuv=y=gammaval(0.5)[left];[1:v]format=yuv444p,lutyuv=y=gammaval(0.5)[right];[left][right]anaglyph[v]"
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-i", f"{left_frame_dir}/frame_%06d.png",
            "-framerate", str(fps),
            "-i", f"{right_frame_dir}/frame_%06d.png",
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-i", self.input_path,  # Use original video as third input for audio
            "-map", "2:a?",  # Map audio if it exists
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",  # Copy audio stream without re-encoding
            self.output_path
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

# Context manager for automatic cleanup
class VideoProcessorContext:
    def __init__(self, input_path, output_path):
        self.processor = VideoProcessor(input_path, output_path)
        
    def __enter__(self):
        return self.processor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.processor.cleanup()