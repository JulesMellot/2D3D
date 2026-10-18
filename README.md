# 2D3D — 2D to 3D Video Converter

An open-source alternative to Owl3D: convert any 2D video into stereoscopic 3D using AI monocular depth estimation ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)).

[GitHub Repository](https://github.com/JulesMellot/2D3D.git)

## How it works

For each frame: estimate depth with Depth Anything V2 → keep the original frame as the left eye → warp the right eye by depth-based disparity → stream straight into ffmpeg. No temporary files, original fps and audio are preserved.

## Features

- **AI depth estimation** — Depth Anything V2 (small/base/large), GPU-accelerated (CUDA, Apple Silicon MPS); the `fast` profile runs on the Apple Neural Engine via Core ML on macOS (~3× faster than MPS)
- **Output formats** — side-by-side (SBS), top-bottom, red-cyan anaglyph
- **3D TV mode** — optional half-width SBS: each eye is squeezed horizontally for TV playback
- **Stereo metadata** — SBS and top-bottom outputs include H.264 frame-packing flags plus MP4 `st3d` or MKV StereoMode metadata for player detection
- **Apple spatial video** — optional MV-HEVC export for Vision Pro (`--spatial`)
- **Quality preservation** — the left eye is the untouched source frame; audio is copied as-is
- **Temporal smoothing** — reduces depth flicker between frames
- **CLI and GUI** — command line or a simple Tkinter interface

## Installation

Requirements: Python 3.10+, [FFmpeg](https://ffmpeg.org), and optionally a GPU (NVIDIA CUDA or Apple Silicon).

```bash
pip install -r requirements.txt
```

The depth model (~100 MB for `fast`) is downloaded automatically from Hugging Face on first run.

For Vision Pro spatial video output (macOS only):

```bash
brew install spatial
```

## Usage

### Command line

```bash
# Basic conversion (side-by-side)
python main.py input.mp4

# Half-width SBS for a 3D TV (1920x1080 source -> 1920x1080 output)
python main.py input.mp4 --3dtv

# Choose output, format and depth model size
python main.py input.mp4 -o output.mp4 -f tb -p precision

# Stronger 3D effect, custom per-eye resolution
python main.py input.mp4 --strength 0.05 --eye-resolution 1920 1080

# Also produce an Apple spatial video for Vision Pro
python main.py input.mp4 --spatial
```

| Option | Default | Description |
|---|---|---|
| `-p, --profile` | `fast` | Depth model: `fast` (Small), `balanced` (Base), `precision` (Large) |
| `-f, --format` | `sbs` | `sbs`, `tb`, or `anaglyph` |
| `--3dtv` | off | Half-width SBS for 3D TVs; SBS only, incompatible with `--spatial` |
| `--strength` | `0.03` | Max disparity as a fraction of frame width |
| `--smoothing` | `0.3` | Temporal depth smoothing (0 = off) |
| `--eye-resolution W H` | source | Resolution per eye before optional 3D TV squeezing |
| `--crf` / `--preset` | `18` / `medium` | x264 encoding quality/speed |
| `--spatial` | off | Also export MV-HEVC spatial video (SBS only, macOS) |

### GUI

```bash
python main.py --gui
```

Select **3D TV mode (half-width SBS)** with the SBS format. A 1920x1080
source produces two 960x1080 views in one 1920x1080 frame; select SBS 3D
playback on your TV to stretch each view back to full width. With a custom
`--eye-resolution W H`, the combined output is W x H in this mode, except
when W:H is narrower than 16:9: each eye receives centered black side bars
before squeezing, widening the combined output to 16:9 (rounded up to an
even width). For example, 1440x1080 becomes 1920x1080. Depth estimation and
stereo warping still process the original image without the bars. Images
at 16:9 or wider receive no padding.
Dimensions are rounded down to even values for encoding. Default SBS output
remains full width (3840x1080 for a 1920x1080 source). TV-mode filenames use
`_stereo_hsbs.mp4`.

### Python API

```python
from pipeline import Pipeline

Pipeline(profile="fast", strength=0.03).convert("input.mp4", "output.mp4", format="sbs")
```

## Model licenses

| Profile | Model | License |
|---|---|---|
| `fast` | Depth Anything V2 Small | Apache-2.0 |
| `balanced` | Depth Anything V2 Base | CC-BY-NC-4.0 (non-commercial) |
| `precision` | Depth Anything V2 Large | CC-BY-NC-4.0 (non-commercial) |

Use `fast` for commercial work — it is also the quickest and works very well in practice.

## Testing

```bash
python test_pipeline.py
```

Runs a synthetic clip through every output format with a stub depth model (no download needed).

## Roadmap

- Diffusion-based occlusion inpainting ([StereoCrafter](https://github.com/TencentARC/StereoCrafter)-style) to replace the naive warp at disocclusions
- Native MV-HEVC export via AVFoundation (currently delegated to the `spatial` CLI)

## License

MIT
