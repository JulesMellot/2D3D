"""
Script to process video files and convert them to stereoscopic 3D.
"""

import cv2
import os
import tempfile
import shutil
from pipeline import Pipeline
from video_io.processor import VideoProcessor, VideoProcessorContext

def process_video(input_path, output_path, profile="balanced", format="sbs"):
    """
    Process a video file and convert it to stereoscopic 3D.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output stereo video file
        profile (str): Depth estimation profile ("fast", "balanced", "precision")
        format (str): Stereo format ("sbs", "tb", "anaglyph")
    """
    print(f"Processing video: {input_path}")
    print(f"Output format: {format}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Create pipeline
    pipeline = Pipeline(depth_profile=profile)
    
    # Use video processor context for automatic cleanup
    with VideoProcessorContext(input_path, output_path) as processor:
        # Extract frames
        print("Extracting frames...")
        frame_dir = processor.extract_frames()
        print(f"Frames extracted to: {frame_dir}")
        
        # Create directories for left and right frames
        left_dir = os.path.join(frame_dir, "left")
        right_dir = os.path.join(frame_dir, "right")
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        
        # Process each frame
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png") and f.startswith("frame_")])
        print(f"Processing {len(frame_files)} frames...")
        
        for i, frame_file in enumerate(frame_files):
            if frame_file.startswith("frame_") and frame_file.endswith(".png"):
                print(f"Processing frame {i+1}/{len(frame_files)}")
                
                # Load frame
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"Warning: Could not load frame {frame_file}")
                    continue
                
                # Process frame
                try:
                    left_frame, right_frame = pipeline.process_frame(frame)
                    
                    # Save processed frames
                    left_path = os.path.join(left_dir, frame_file)
                    right_path = os.path.join(right_dir, frame_file)
                    cv2.imwrite(left_path, left_frame)
                    cv2.imwrite(right_path, right_frame)
                except Exception as e:
                    print(f"Error processing frame {frame_file}: {e}")
                    # Copy original frame as fallback
                    shutil.copy(frame_path, os.path.join(left_dir, frame_file))
                    shutil.copy(frame_path, os.path.join(right_dir, frame_file))
        
        # Get video properties for output
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Create stereo video
        print("Creating stereo video...")
        processor.create_stereo_video(left_dir, right_dir, fps=fps, format=format)
        print(f"Stereo video saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 2D video to stereoscopic 3D")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-p", "--profile", choices=["fast", "balanced", "precision"], 
                        default="balanced", help="Depth estimation profile")
    parser.add_argument("-f", "--format", choices=["sbs", "tb", "anaglyph"], 
                        default="sbs", help="Stereo format")
    
    args = parser.parse_args()
    
    # Determine output path if not provided
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}_stereo.mp4"
    
    try:
        process_video(args.input, output_path, args.profile, args.format)
        print("Video processing completed successfully!")
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())