"""
Video analysis utilities for detecting aspect ratios and configuring 3D conversion settings.
"""

import cv2
import subprocess
import json
import os
from fractions import Fraction

def get_video_properties(video_path):
    """
    Get video properties including resolution, aspect ratio, and FPS.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video properties including width, height, aspect_ratio, fps, etc.
    """
    # Use OpenCV to get basic properties
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get basic properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    # Calculate aspect ratio
    aspect_ratio = width / height if height > 0 else 0
    
    # Use ffprobe for more detailed information
    try:
        cmd = [
            "ffprobe", 
            "-v", "quiet", 
            "-print_format", "json", 
            "-show_streams", 
            "-show_format",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        # Get more accurate FPS from ffprobe
        stream = next((s for s in video_info.get("streams", []) if s.get("codec_type") == "video"), None)
        if stream:
            # Get display aspect ratio if available
            if "display_aspect_ratio" in stream:
                dar_parts = stream["display_aspect_ratio"].split(":")
                if len(dar_parts) == 2:
                    dar = int(dar_parts[0]) / int(dar_parts[1])
                    # Use display aspect ratio if it's different from storage aspect ratio
                    if abs(dar - aspect_ratio) > 0.01:  # More than 1% difference
                        aspect_ratio = dar
            
            # Get more accurate FPS
            if "avg_frame_rate" in stream:
                fps_fraction = Fraction(stream["avg_frame_rate"])
                fps = float(fps_fraction)
        
        format_info = video_info.get("format", {})
        duration = float(format_info.get("duration", 0))
        
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        # Fallback to OpenCV properties if ffprobe fails
        duration = frame_count / fps if fps > 0 else 0
        print("Warning: ffprobe not available, using OpenCV for video properties")
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "resolution": f"{width}x{height}"
    }

def detect_common_aspect_ratio(width, height):
    """
    Detect common aspect ratios and return standardized values.
    
    Args:
        width (int): Video width
        height (int): Video height
        
    Returns:
        tuple: (display_width, display_height, name)
    """
    aspect_ratio = width / height
    
    # Common aspect ratios with some tolerance
    common_ratios = [
        (16, 9, "16:9"),
        (4, 3, "4:3"),
        (21, 9, "21:9"),
        (16, 10, "16:10"),
        (3, 2, "3:2"),
        (5, 4, "5:4"),
        (1, 1, "1:1")
    ]
    
    tolerance = 0.05  # 5% tolerance
    
    for dw, dh, name in common_ratios:
        if abs(aspect_ratio - (dw/dh)) < tolerance:
            return (dw, dh, name)
    
    # If no common ratio found, return the actual ratio
    return (width, height, f"{width}:{height}")

def calculate_3d_output_resolution(input_width, input_height, format_type, sbs_mode="half"):
    """
    Calculate appropriate output resolution for 3D formats.
    
    Args:
        input_width (int): Input video width
        input_height (int): Input video height
        format_type (str): 3D format ("sbs", "tb", "anaglyph")
        sbs_mode (str): SBS mode ("half" or "full") - only for SBS format
        
    Returns:
        tuple: (output_width, output_height, eye_width, eye_height)
    """
    if format_type == "sbs":  # Side-by-Side
        if sbs_mode == "half":
            # Each eye is horizontally compressed to half width
            eye_width = input_width // 2
            eye_height = input_height
            output_width = input_width
            output_height = input_height
        else:  # full
            # Each eye maintains full resolution
            eye_width = input_width
            eye_height = input_height
            output_width = input_width * 2
            output_height = input_height
    elif format_type == "tb":  # Top-Bottom
        # Each eye is vertically compressed to half height
        eye_width = input_width
        eye_height = input_height // 2
        output_width = input_width
        output_height = input_height
    else:  # anaglyph or other formats
        # Same resolution as input
        eye_width = input_width
        eye_height = input_height
        output_width = input_width
        output_height = input_height
    
    return (output_width, output_height, eye_width, eye_height)

def get_optimal_3d_settings(video_path, format_type="sbs", quality_preset="balanced"):
    """
    Get optimal 3D conversion settings based on video properties and format.
    
    Args:
        video_path (str): Path to input video
        format_type (str): 3D format ("sbs", "tb", "anaglyph")
        quality_preset (str): Quality preset ("fast", "balanced", "quality", "3dtv", "vr")
        
    Returns:
        dict: Optimal settings for 3D conversion
    """
    # Get video properties
    props = get_video_properties(video_path)
    width = props["width"]
    height = props["height"]
    aspect_ratio = props["aspect_ratio"]
    
    # Detect common aspect ratio
    display_width, display_height, aspect_name = detect_common_aspect_ratio(width, height)
    
    # Calculate 3D output resolution
    sbs_mode = "half"  # Default to half SBS for compatibility
    output_width, output_height, eye_width, eye_height = calculate_3d_output_resolution(
        width, height, format_type, sbs_mode
    )
    
    # Adjust settings based on quality preset
    settings = {
        # Basic properties
        "input_resolution": f"{width}x{height}",
        "input_aspect_ratio": aspect_ratio,
        "aspect_ratio_name": aspect_name,
        "output_resolution": (output_width, output_height),
        "eye_resolution": (eye_width, eye_height),
        "sbs_mode": sbs_mode,
        
        # Default pipeline settings
        "depth_profile": "balanced",
        "baseline": 0.05,
        "focal_length": 1000,
        "preserve_colors": True,
        "temporal_smoothing": True,
        "post_process": False,
        "temporal_smoothing_factor": 0.1,
        
        # Encoding settings
        "crf": 18,
        "preset": "medium",
        "max_dimension": None,  # No resizing by default
    }
    
    # Adjust based on quality preset
    if quality_preset == "fast":
        settings.update({
            "depth_profile": "fast",
            "baseline": 0.03,
            "preset": "veryfast",
            "crf": 23,
            "temporal_smoothing": False,
            "post_process": False,
        })
        # Reduce resolution for faster processing
        settings["eye_resolution"] = (eye_width // 2, eye_height // 2)
        if format_type == "sbs":
            settings["output_resolution"] = (eye_width, eye_height)
        elif format_type == "tb":
            settings["output_resolution"] = (eye_width, eye_height // 2)
    elif quality_preset == "quality":
        settings.update({
            "depth_profile": "precision",
            "baseline": 0.07,
            "preset": "slow",
            "crf": 15,
            "temporal_smoothing": True,
            "post_process": True,
            "temporal_smoothing_factor": 0.15,
        })
    elif quality_preset == "3dtv":
        settings.update({
            "depth_profile": "balanced",
            "baseline": 0.065,  # Matches human interpupillary distance
            "preset": "medium",
            "crf": 18,
            "temporal_smoothing": True,
            "post_process": False,
        })
        # Ensure TV-compatible resolution
        if width > 1920:
            scale_factor = 1920 / width
            new_eye_width = int(eye_width * scale_factor)
            new_eye_height = int(eye_height * scale_factor)
            settings["eye_resolution"] = (new_eye_width, new_eye_height)
            if format_type == "sbs":
                settings["output_resolution"] = (new_eye_width * 2, new_eye_height)
            elif format_type == "tb":
                settings["output_resolution"] = (new_eye_width, new_eye_height * 2)
    elif quality_preset == "vr":
        settings.update({
            "depth_profile": "precision",
            "baseline": 0.05,
            "preset": "slow",
            "crf": 15,
            "temporal_smoothing": True,
            "post_process": True,
            "temporal_smoothing_factor": 0.2,
            "focal_length": 800,  # Shorter focal length for more depth in VR
        })
        # Higher resolution for VR
        if width < 3840:
            scale_factor = min(3840 / width, 2160 / height)
            if scale_factor > 1.5:
                new_eye_width = int(eye_width * scale_factor)
                new_eye_height = int(eye_height * scale_factor)
                settings["eye_resolution"] = (new_eye_width, new_eye_height)
                if format_type == "sbs":
                    settings["output_resolution"] = (new_eye_width * 2, new_eye_height)
                elif format_type == "tb":
                    settings["output_resolution"] = (new_eye_width, new_eye_height * 2)
    
    return settings

# Example usage
if __name__ == "__main__":
    # Example of how to use these functions
    try:
        # Test with a video file (replace with actual path)
        # settings = get_optimal_3d_settings("input_video.mp4", "sbs", "balanced")
        # print("Optimal 3D settings:", settings)
        pass
    except Exception as e:
        print(f"Error: {e}")