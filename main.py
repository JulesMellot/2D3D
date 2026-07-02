"""CLI entry point for 2D3D."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert 2D videos to stereoscopic 3D")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-p", "--profile", choices=["fast", "balanced", "precision"],
                        default="fast", help="Depth model size (default: fast)")
    parser.add_argument("-f", "--format", choices=["sbs", "tb", "anaglyph"],
                        default="sbs", help="Stereo format")
    parser.add_argument("--strength", type=float, default=0.03,
                        help="3D strength: max disparity as fraction of width (default: 0.03)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="Temporal depth smoothing 0-1, 0 disables (default: 0.3)")
    parser.add_argument("--eye-resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Output resolution per eye (default: source resolution)")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality (default: 18)")
    parser.add_argument("--preset", default="medium", help="x264 preset (default: medium)")
    parser.add_argument("--spatial", action="store_true",
                        help="Also produce an Apple spatial video (.mov) for Vision Pro "
                             "(sbs format only, needs `brew install spatial`)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI application")
    args = parser.parse_args()

    if args.gui:
        from gui_app import main as gui_main
        gui_main()
        return 0

    if not args.input:
        parser.print_help()
        return 1
    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found")
        return 1
    if args.spatial and args.format != "sbs":
        print("Error: --spatial requires --format sbs")
        return 1

    output = args.output or f"{os.path.splitext(os.path.basename(args.input))[0]}_stereo_{args.format}.mp4"
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    from pipeline import Pipeline, make_spatial

    pipeline = Pipeline(profile=args.profile, strength=args.strength, smoothing=args.smoothing)
    pipeline.convert(
        args.input, output,
        format=args.format,
        eye_resolution=tuple(args.eye_resolution) if args.eye_resolution else None,
        crf=args.crf, preset=args.preset,
        progress_callback=lambda n, total: print(f"\rFrame {n}/{total or '?'}", end="", flush=True),
    )
    print(f"\nDone: {output}")

    if args.spatial:
        spatial_out = make_spatial(output)
        print(f"Spatial video: {spatial_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
