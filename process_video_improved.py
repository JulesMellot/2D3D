"""
Script to process video with improved quality and audio preservation.
"""

import cv2
import os
import tempfile
import shutil
from improved_pipeline import ImprovedPipeline

def process_video_with_audio(input_path, output_path, profile="balanced", max_dimension=1920):
    """
    Process video with improved quality and audio preservation.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        profile (str): Depth estimation profile ("fast", "balanced", "precision")
        max_dimension (int): Maximum dimension for processing (to reduce memory usage)
    """
    print(f"Processing video: {input_path}")
    print(f"Using profile: {profile}")
    print(f"Max dimension: {max_dimension}")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    
    try:
        # Extract frames
        print("Extracting frames...")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Use ffmpeg to extract frames
        os.system(f"ffmpeg -i {input_path} -vf fps=24 {frames_dir}/frame_%05d.png")
        
        # Count frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        print(f"Extracted {len(frame_files)} frames")
        
        if not frame_files:
            raise ValueError("No frames extracted from video")
        
        # Create pipeline
        pipeline = ImprovedPipeline(depth_profile=profile)
        
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
            
            # Resize frame if necessary to reduce memory usage
            height, width = frame.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"  Resized frame to {new_width}x{new_height}")
            
            # Process frame
            try:
                left_frame, right_frame = pipeline.process_frame(frame)
                
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
        
        # Create side-by-side stereo video with audio
        print("Creating stereo video with audio...")
        left_pattern = f"{processed_dir}/left_frame_%05d.png"
        right_pattern = f"{processed_dir}/right_frame_%05d.png"
        
        # Use ffmpeg to create stereo video with audio preserved
        os.system(f"ffmpeg -y -framerate {fps} -i {left_pattern} -framerate {fps} -i {right_pattern} "
                  f"-i {input_path} -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' -map 2:a? "
                  f"-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a copy {output_path}")
        
        print(f"Video processing completed! Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise
    finally:
        # Clean up temporary directories
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 2D video to stereoscopic 3D with improved quality")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-p", "--profile", choices=["fast", "balanced", "precision"], 
                        default="balanced", help="Depth estimation profile")
    parser.add_argument("-m", "--max-dimension", type=int, default=1920,
                        help="Maximum dimension for processing (to reduce memory usage)")
    
    args = parser.parse_args()
    
    # Determine output path if not provided
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}_stereo_improved.mp4"
    
    try:
        process_video_with_audio(args.input, output_path, args.profile, args.max_dimension)
        print("Video processing completed successfully!")
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())