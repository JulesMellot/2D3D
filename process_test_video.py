"""
Process the test video with quality-preserving pipeline.
"""

import os
from quality_pipeline import QualityPreservingPipeline

def main():
    input_path = "test2.mp4"
    output_path = "test2_quality_stereo.mp4"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found!")
        return
    
    print(f"Processing {input_path} with quality-preserving pipeline...")
    
    # Create pipeline with balanced profile for better quality
    pipeline = QualityPreservingPipeline(depth_profile="balanced", baseline=0.03)
    
    # Convert video (processing at original resolution to preserve quality)
    pipeline.convert_video(
        input_path, 
        output_path, 
        format="sbs",  # Side-by-side format
        max_dimension=None,  # Process at original resolution
        output_resolution=(960, 1080),  # Output resolution for each eye
        auto_settings=True  # Enable automatic settings detection
    )
    
    print(f"Processing completed! Output saved to {output_path}")

if __name__ == "__main__":
    main()