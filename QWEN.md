# 2D3D - 2D to 3D Video Converter - Project Context

## Project Overview

This project is an open-source tool for converting 2D videos to stereoscopic 3D while preserving the original video quality. It provides both a command-line interface and a graphical user interface for easy use.

### Key Features

- **Depth Estimation**: Multiple monocular depth models (MiDaS, ZoeDepth, Depth Anything V2)
- **Stereo View Synthesis**: Accurate view warping with occlusion handling
- **Quality Preservation**: Maintains original color, contrast, and visual integrity
- **Multiple Output Formats**: Side-by-side, top-bottom, anaglyph, RGBD
- **Custom Output Resolution**: Specify resolution per eye for 3D TV compatibility
- **Automatic Aspect Ratio Detection**: Intelligent settings based on input video properties
- **Advanced Progress Monitoring**: Real-time progress with accurate ETA calculation
- **Audio Preservation**: Maintains all original audio tracks and metadata
- **Cross-Platform**: GPU acceleration support (CUDA, MPS, ROCm)
- **Performance Optimization**: Chunked processing for large videos
- **Temporal Consistency**: Frame-to-frame stability algorithms
- **Graphical User Interface**: Easy-to-use GUI application with presets
- **Custom Temporary Directory**: Specify where temporary files are stored
- **Extensible Architecture**: Modular design for easy enhancements

## Codebase Structure

```
2D3D/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── setup.py                  # Package installation
├── config.json              # Default configuration
├── main.py                  # CLI interface
├── gui_app.py               # GUI application
├── enhanced_pipeline.py     # Main conversion pipeline
├── quality_pipeline.py      # Quality-preserving pipeline
├── process_test_video.py    # Test video processing
├── test_quality.py          # Quality verification
├── test_suite.py            # Test suite
├── depth_estimation/        # Depth models
│   ├── improved.py          # Improved OpenCV method
│   └── midas.py             # MiDaS integration
├── stereo_warp/             # View synthesis
│   └── warper.py            # Image warping
├── inpainting/              # Occlusion filling
│   ├── basic.py             # Basic methods
│   └── diffusion.py         # Diffusion framework
├── video_io/                # Video processing
│   └── processor.py         # FFmpeg integration
├── utils/                   # Utility functions
│   ├── device.py            # GPU detection
│   ├── config.py            # Configuration
│   ├── temporal.py          # Temporal consistency
│   └── spherical.py         # 360° video support
└── gui.py                   # GUI framework
```

## Core Components

### 1. Main Pipeline (`enhanced_pipeline.py`)
The core conversion pipeline that handles:
- Video frame extraction
- Depth estimation
- Stereo pair generation
- Quality preservation
- Video recomposition
- **Advanced progress monitoring** with accurate ETA calculation

### 2. Depth Estimation (`depth_estimation/`)
Multiple approaches for depth estimation:
- **Improved**: OpenCV-based stereo matching method
- **MiDaS**: Integration with MiDaS depth models
- **Simple**: Basic gradient-based method

### 3. Temporal Consistency (`utils/temporal.py`)
Algorithms to ensure frame-to-frame stability:
- Optical flow-based consistency
- Feature matching-based consistency

### 4. Inpainting (`inpainting/`)
Methods for filling occluded regions:
- **Basic**: Traditional methods (Telea, Navier-Stokes)
- **Diffusion**: Framework for diffusion-based inpainting

### 5. GUI Application (`gui_app.py`)
Tkinter-based graphical interface with:
- File browsing
- Parameter adjustment
- Progress monitoring
- Progress bar and detailed status information
- Error handling
- Output resolution per eye selector
- Preset system (Fast/Balanced/Quality/3DTV/VR/Custom)
- Advanced settings controls
- Tabbed interface for organized controls

## Key Technologies

- **Python**: Primary language
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning models
- **FFmpeg**: Video processing
- **NumPy**: Numerical computations
- **Tkinter**: GUI framework

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg
- CUDA-compatible GPU (optional, for NVIDIA acceleration)
- Apple Silicon Mac (optional, for MPS acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
# Convert a video to stereo 3D
python main.py input_video.mp4 -o output_video.mp4

# Convert with specific settings
python main.py input_video.mp4 -p balanced --baseline 0.05 --format sbs

# Process at specific resolution
python main.py input_video.mp4 -m 1080  # Process at max 1080p

# Convert with custom output resolution (960x1080 per eye for SBS format)
python main.py input_video.mp4 -o output_video.mp4 --output-resolution 960 1080

# Use custom temporary directory
python main.py input_video.mp4 --temp-dir /path/to/temp/dir

# Launch GUI application
python main.py --gui
```

### Graphical User Interface
```bash
# Launch the GUI application
python main.py --gui

# Or directly run the GUI
python gui_app.py
```

### Python API
```python
from enhanced_pipeline import EnhancedPipeline

# Create pipeline
pipeline = EnhancedPipeline(depth_profile="balanced", baseline=0.05)

# Convert video with default resolution and automatic settings
pipeline.convert_video("input.mp4", "output.mp4", format="sbs", temp_dir="/path/to/temp/dir")

# Convert video with custom output resolution (960x1080 per eye for SBS format)
pipeline.convert_video("input.mp4", "output.mp4", format="sbs", output_resolution=(960, 1080))

# Convert video with automatic settings disabled
pipeline.convert_video("input.mp4", "output.mp4", format="sbs", auto_settings=False)
```

## Configuration

The system uses a JSON-based configuration file (`config.json`) for all settings:

```json
{
    "depth_profile": "balanced",
    "depth_model": "MiDaS",
    "baseline": 0.05,
    "focal_length": 1000,
    "shift_direction": "horizontal",
    "preserve_colors": true,
    "temporal_smoothing": true,
    "temporal_smoothing_factor": 0.1,
    "post_process": false,
    "max_dimension": null,
    "batch_size": 1,
    "gpu_acceleration": true,
    "default_format": "sbs",
    "default_resolution": [960, 1080],
    "crf": 18,
    "preset": "medium"
}
```

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Write clear comments for complex logic

### Testing
- Unit tests in `test_suite.py`
- Quality verification in `test_quality.py`
- Run tests with `python test_suite.py`

### Extensibility
The modular design allows for easy extensions:
1. Adding new depth models in `depth_estimation/`
2. Adding new inpainting methods in `inpainting/`
3. Adding temporal consistency methods in `utils/temporal.py`

## Performance Optimization

The project includes several optimization strategies:
- GPU acceleration (CUDA, MPS, ROCm)
- Memory-efficient processing
- Chunked video processing
- Model optimization techniques

For details, see `OPTIMIZATION.md`.

## Future Enhancements

Planned improvements include:
- Full diffusion-based inpainting implementation
- Advanced depth models (ZoeDepth, Depth Anything V2)
- 360° video processing
- Web-based interface
- Mobile application

## Testing

Run the test suite with:
```bash
python test_suite.py
```

Run quality verification with:
```bash
python test_quality.py
```