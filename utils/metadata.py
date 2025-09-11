"""
3D video metadata utilities for adding standard metadata to 3D video files.
"""

import subprocess
import json
import os

def add_3d_metadata(input_path, output_path, stereo_format, overwrite=True):
    """
    Add 3D metadata to a video file using FFmpeg.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output video file with metadata
        stereo_format (str): 3D format ("sbs", "tb", "anaglyph")
        overwrite (bool): Whether to overwrite existing file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Build FFmpeg command with 3D metadata
    cmd = ["ffmpeg", "-i", input_path]
    
    # Add overwrite flag if needed
    if overwrite:
        cmd.insert(1, "-y")
    
    # Add metadata based on stereo format
    if stereo_format == "sbs":
        # Side-by-side 3D
        cmd.extend([
            "-metadata", "stereo_mode=side_by_side",
            "-metadata", "major_brand=mp42",
            "-metadata", "minor_version=1",
            "-metadata", "compatible_brands=mp42isom",
        ])
    elif stereo_format == "tb":
        # Top-bottom 3D
        cmd.extend([
            "-metadata", "stereo_mode=top_bottom",
            "-metadata", "major_brand=mp42",
            "-metadata", "minor_version=1",
            "-metadata", "compatible_brands=mp42isom",
        ])
    elif stereo_format == "anaglyph":
        # Anaglyph 3D
        cmd.extend([
            "-metadata", "stereo_mode=anaglyph",
            "-metadata", "major_brand=mp42",
            "-metadata", "minor_version=1",
            "-metadata", "compatible_brands=mp42isom",
        ])
    
    # Add additional metadata for 3D compatibility
    cmd.extend([
        "-metadata", "3d_video=true",
        "-metadata", "3d_format=" + stereo_format,
        "-c", "copy",  # Copy streams without re-encoding
        output_path
    ])
    
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding 3D metadata: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error adding 3D metadata: {e}")
        return False

def add_advanced_3d_metadata(input_path, output_path, stereo_format, parallax_info=None, overwrite=True):
    """
    Add advanced 3D metadata including parallax information.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output video file with metadata
        stereo_format (str): 3D format ("sbs", "tb", "anaglyph")
        parallax_info (dict): Additional parallax information
        overwrite (bool): Whether to overwrite existing file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Build FFmpeg command with advanced 3D metadata
    cmd = ["ffmpeg", "-i", input_path]
    
    # Add overwrite flag if needed
    if overwrite:
        cmd.insert(1, "-y")
    
    # Add basic 3D metadata
    if stereo_format == "sbs":
        cmd.extend(["-metadata", "stereo_mode=side_by_side"])
    elif stereo_format == "tb":
        cmd.extend(["-metadata", "stereo_mode=top_bottom"])
    elif stereo_format == "anaglyph":
        cmd.extend(["-metadata", "stereo_mode=anaglyph"])
    
    # Add general 3D metadata
    cmd.extend([
        "-metadata", "3d_video=true",
        "-metadata", "3d_format=" + stereo_format,
    ])
    
    # Add parallax information if provided
    if parallax_info:
        # Add parallax metadata
        if "baseline" in parallax_info:
            cmd.extend(["-metadata", f"3d_baseline={parallax_info['baseline']}"])
        if "focal_length" in parallax_info:
            cmd.extend(["-metadata", f"3d_focal_length={parallax_info['focal_length']}"])
        if "max_disparity" in parallax_info:
            cmd.extend(["-metadata", f"3d_max_disparity={parallax_info['max_disparity']}"])
        if "convergence_distance" in parallax_info:
            cmd.extend(["-metadata", f"3d_convergence={parallax_info['convergence_distance']}"])
    
    # Add container format metadata for better compatibility
    cmd.extend([
        "-metadata", "major_brand=mp42",
        "-metadata", "minor_version=1",
        "-metadata", "compatible_brands=mp42isomavc1",
        "-c", "copy",  # Copy streams without re-encoding
        output_path
    ])
    
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding advanced 3D metadata: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error adding advanced 3D metadata: {e}")
        return False

def get_video_metadata(video_path):
    """
    Get video metadata using ffprobe.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video metadata
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting video metadata: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error getting video metadata: {e}")
        return {}

def check_3d_metadata(video_path):
    """
    Check if a video file has 3D metadata.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: 3D metadata information
    """
    metadata = get_video_metadata(video_path)
    
    # Extract 3D-related metadata
    result = {
        "is_3d": False,
        "stereo_mode": None,
        "3d_format": None,
        "baseline": None,
        "focal_length": None
    }
    
    # Check format metadata
    format_info = metadata.get("format", {}).get("tags", {})
    for key, value in format_info.items():
        key_lower = key.lower()
        if "stereo" in key_lower or "3d" in key_lower:
            result["is_3d"] = True
            if "stereo_mode" in key_lower:
                result["stereo_mode"] = value
            elif "3d_format" in key_lower:
                result["3d_format"] = value
            elif "baseline" in key_lower:
                result["baseline"] = value
            elif "focal" in key_lower:
                result["focal_length"] = value
    
    # Check stream metadata
    for stream in metadata.get("streams", []):
        tags = stream.get("tags", {})
        for key, value in tags.items():
            key_lower = key.lower()
            if "stereo" in key_lower or "3d" in key_lower:
                result["is_3d"] = True
                if "stereo_mode" in key_lower:
                    result["stereo_mode"] = value
                elif "3d_format" in key_lower:
                    result["3d_format"] = value
                elif "baseline" in key_lower:
                    result["baseline"] = value
                elif "focal" in key_lower:
                    result["focal_length"] = value
    
    return result

# Example usage
if __name__ == "__main__":
    # Example: Add 3D metadata to a video
    # add_3d_metadata("input.mp4", "output_3d.mp4", "sbs")
    pass