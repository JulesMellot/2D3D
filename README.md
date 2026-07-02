# 2D3D — 2D to 3D Video Converter

An open-source alternative to Owl3D: convert any 2D video into stereoscopic 3D using AI monocular depth estimation ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)).

[GitHub Repository](https://github.com/JulesMellot/2D3D.git)

## How it works

For each frame: estimate depth with Depth Anything V2 → keep the original frame as the left eye → warp the right eye by depth-based disparity → stream straight into ffmpeg. No temporary files, original fps and audio are preserved.

## Features

- **AI depth estimation** — Depth Anything V2 (small/base/large), GPU-accelerated (CUDA, Apple Silicon MPS); the `fast` profile runs on the Apple Neural Engine via Core ML on macOS (~3× faster than MPS)
- **Output formats** — side-by-side (SBS), top-bottom, red-cyan anaglyph
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
| `--strength` | `0.03` | Max disparity as a fraction of frame width |
| `--smoothing` | `0.3` | Temporal depth smoothing (0 = off) |
| `--eye-resolution W H` | source | Output resolution per eye |
| `--crf` / `--preset` | `18` / `medium` | x264 encoding quality/speed |
| `--spatial` | off | Also export MV-HEVC spatial video (SBS only, macOS) |

### GUI

```bash
python main.py --gui
```

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
