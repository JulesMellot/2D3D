"""
Main entry point for the 2D3D application.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert 2D videos to stereoscopic 3D with quality preservation")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-p", "--profile", choices=["fast", "balanced", "precision"], 
                        default="balanced", help="Depth estimation profile")
    parser.add_argument("--baseline", type=float, default=0.05, help="Stereo baseline")
    parser.add_argument("--focal-length", type=float, default=1000, help="Camera focal length")
    parser.add_argument("-f", "--format", choices=["sbs", "tb", "anaglyph"], 
                        default="sbs", help="Stereo format")
    parser.add_argument("-m", "--max-dimension", type=int, 
                        help="Maximum dimension for processing (to reduce memory usage)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI application")
    
    args = parser.parse_args()
    
    # Launch GUI if requested
    if args.gui:
        try:
            from gui_app import main as gui_main
            gui_main()
            return 0
        except ImportError as e:
            print(f"GUI dependencies not available: {e}")
            print("Install tkinter or run in command line mode")
            return 1
    
    # Check if input file is provided
    if not args.input:
        parser.print_help()
        return 1
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}_stereo_{args.format}.mp4"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import here to avoid tkinter dependency for CLI usage
        from enhanced_pipeline import EnhancedPipeline
        
        # Create pipeline
        pipeline = EnhancedPipeline(
            depth_profile=args.profile,
            baseline=args.baseline,
            focal_length=args.focal_length
        )
        
        # Convert video
        pipeline.convert_video(
            args.input, 
            output_path, 
            format=args.format,
            max_dimension=args.max_dimension
        )
        
        print("Conversion completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())