"""
Test script to process a small portion of the video.
"""

import cv2
import os
import tempfile
import shutil
from pipeline import Pipeline

def extract_sample_frames(video_path, output_dir, num_frames=10):
    """
    Extract a sample of frames from the video for testing.
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to save frames
        num_frames (int): Number of frames to extract
        
    Returns:
        list: List of extracted frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    # Calculate which frames to extract
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"sample_frame_{i:03d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            print(f"Saved frame {i+1}/{num_frames}: {frame_path}")
        else:
            print(f"Failed to read frame {frame_idx}")
    
    cap.release()
    return frame_paths

def process_sample_frames(frame_paths, output_dir):
    """
    Process sample frames through the 2D3D pipeline.
    
    Args:
        frame_paths (list): List of frame paths to process
        output_dir (str): Directory to save processed frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = Pipeline(depth_profile="fast")  # Use fast profile for testing
    
    print(f"Processing {len(frame_paths)} sample frames...")
    
    for i, frame_path in enumerate(frame_paths):
        print(f"Processing frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
        
        try:
            # Process frame
            left_path, right_path = pipeline.process_image(frame_path, output_dir)
            print(f"  Saved stereo pair: {os.path.basename(left_path)}, {os.path.basename(right_path)}")
        except Exception as e:
            print(f"  Error processing frame: {e}")

def main():
    video_path = "test2.mp4"
    sample_dir = "sample_frames"
    output_dir = "processed_frames"
    
    print("Extracting sample frames...")
    frame_paths = extract_sample_frames(video_path, sample_dir, num_frames=5)
    
    print("\nProcessing sample frames...")
    process_sample_frames(frame_paths, output_dir)
    
    print("\nSample processing completed!")
    print(f"Sample frames saved to: {sample_dir}")
    print(f"Processed frames saved to: {output_dir}")

if __name__ == "__main__":
    main()