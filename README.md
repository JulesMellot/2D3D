# 2D3D - 2D to 3D Video Converter

An open-source alternative to Owl3D for converting 2D videos into stereoscopic 3D while preserving the original video quality.

[GitHub Repository](https://github.com/JulesMellot/2D3D.git)

## Features

- **Depth Estimation**: Multiple monocular depth models (MiDaS, ZoeDepth, Depth Anything V2)
- **Stereo View Synthesis**: Accurate view warping with occlusion handling
- **Quality Preservation**: Maintains original color, contrast, and visual integrity
- **Multiple Output Formats**: Side-by-side, top-bottom, anaglyph, RGBD
- **Audio Preservation**: Maintains all original audio tracks and metadata
- **Cross-Platform**: GPU acceleration support (CUDA, MPS, ROCm)
- **Performance Optimization**: Chunked processing for large videos
- **Temporal Consistency**: Frame-to-frame stability algorithms
- **Graphical User Interface**: Easy-to-use GUI application
- **Extensible Architecture**: Modular design for easy enhancements

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

### Install Additional Depth Models (Optional)
```bash
# For MiDaS
pip install git+https://github.com/isl-org/MiDaS.git

# For ZoeDepth
pip install zoedepth

# For Depth Anything V2
pip install depth-anything-v2
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

# Launch GUI application
python main.py --gui
```

### Command Line Arguments
```
positional arguments:
  input                  Input video file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output video file
  -p {fast,balanced,precision}, --profile {fast,balanced,precision}
                        Depth estimation profile (default: balanced)
  --baseline BASELINE   Stereo baseline (default: 0.05)
  --focal-length FOCAL_LENGTH
                        Camera focal length (default: 1000)
  -f {sbs,tb,anaglyph}, --format {sbs,tb,anaglyph}
                        Stereo format (default: sbs)
  -m MAX_DIMENSION, --max-dimension MAX_DIMENSION
                        Maximum dimension for processing (to reduce memory usage)
  --gui                 Launch GUI application
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

# Convert video
pipeline.convert_video("input.mp4", "output.mp4", format="sbs")
```

## Pipeline Architecture

```
Input Video
    ↓
Frame Extraction (FFmpeg)
    ↓
Depth Estimation
    ↓
Temporal Consistency
    ↓
Stereo Warping
    ↓
Occlusion Filling
    ↓
Color Preservation
    ↓
Frame Recomposition
    ↓
Video Encoding (FFmpeg)
    ↓
Output Video (with preserved audio)
```

## Depth Estimation Models

### MiDaS
- **Fast Profile**: MiDaS Small model for real-time processing
- **Balanced Profile**: DPT Hybrid model for good quality/ speed balance
- **Precision Profile**: DPT Large model for highest quality

### ZoeDepth
- Specialized for indoor/outdoor scene depth estimation
- Better edge preservation than MiDaS

### Depth Anything V2
- Latest state-of-the-art model
- Excellent generalization across different scenes

## Stereo Formats

- **Side-by-Side (SBS)**: Left and right views side by side
- **Top-Bottom (TB)**: Left view on top, right view on bottom
- **Anaglyph**: Red-cyan 3D glasses compatible
- **RGBD**: Color image with depth map

## Configuration Options

### Depth Estimation
- `profile`: "fast", "balanced", or "precision"
- `model`: "MiDaS", "ZoeDepth", or "DepthAnythingV2"

### Stereo Generation
- `baseline`: Distance between virtual cameras (0.01-0.2)
- `focal_length`: Virtual camera focal length
- `shift_direction`: "horizontal" or "vertical"

### Quality Settings
- `preserve_colors`: Maintain original color integrity
- `temporal_smoothing`: Reduce flickering between frames
- `temporal_consistency`: Apply temporal consistency algorithms
- `post_process`: Apply sharpening/ denoising

### Performance Settings
- `max_dimension`: Maximum processing resolution
- `batch_size`: Number of frames to process together
- `gpu_acceleration`: Enable GPU processing

## Temporal Consistency

### Optical Flow
- Frame-to-frame motion estimation
- Depth map warping for consistency
- Configurable blending weights

### Feature Matching
- ORB feature detection and matching
- Homography-based consistency
- Robust to camera motion

## Inpainting Methods

### Basic Inpainting
- OpenCV Navier-Stokes algorithm
- Telea inpainting method
- Fast processing for real-time use

### Diffusion-Based (Framework)
- Stable Diffusion inpainting framework
- Text-guided occlusion filling
- High-quality results (future implementation)

## Performance Optimization

### GPU Acceleration
- **NVIDIA**: CUDA with xFormers attention optimization
- **Apple Silicon**: MPS (Metal Performance Shaders)
- **AMD**: ROCm support (if available)

### Memory Management
- Automatic resolution scaling for large videos
- Chunked processing for long videos
- VRAM optimization for diffusion inpainting

### Processing Options
- Configurable maximum dimensions
- Batch processing capabilities
- Multi-threaded frame extraction

## Graphical User Interface

The GUI application provides an easy-to-use interface for converting videos:

### Features
- File browsing for input/output selection
- Real-time parameter adjustment
- Progress monitoring
- Error handling and notifications
- Cross-platform compatibility

### GUI Controls
- **Input/Output**: File selection dialogs
- **Depth Profile**: Fast/Balanced/Precision options
- **Output Format**: SBS/TB/Anaglyph selection
- **Stereo Baseline**: Slider for depth effect strength
- **Max Dimension**: Resolution limiting control

## Project Structure
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

## Extending the Pipeline

### Adding New Depth Models
1. Create a new module in `depth_estimation/`
2. Implement the `DepthEstimator` interface
3. Register in the pipeline factory

### Adding New Inpainting Methods
1. Create a new module in `inpainting/`
2. Implement the `Inpainter` interface
3. Add to the pipeline configuration

### Adding Temporal Consistency Methods
1. Extend the `TemporalConsistency` class in `utils/temporal.py`
2. Add new method to configuration
3. Update pipeline to use new method

## Configuration Management

The system uses a JSON-based configuration file (`config.json`) for all settings:

```json
{
    "depth_profile": "balanced",
    "depth_model": "MiDaS",
    "baseline": 0.05,
    "focal_length": 1000,
    "preserve_colors": true,
    "temporal_smoothing": true,
    "temporal_consistency": true,
    "temporal_method": "optical_flow",
    "max_dimension": null,
    "default_format": "sbs",
    "crf": 18,
    "preset": "medium"
}
```

## Testing

### Unit Tests
```bash
python test_suite.py
```

### Quality Verification
```bash
python test_quality.py
```

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Setup
```bash
# Clone the repository
git clone https://github.com/JulesMellot/2D3D.git
cd 2D3D

# Install in development mode
pip install -e .
```

## Future Enhancements

### Advanced Depth Models
- Full ZoeDepth integration
- Depth Anything V2 implementation
- Ensemble methods for improved accuracy

### Enhanced Inpainting
- Complete diffusion-based implementation
- GAN-based occlusion filling
- Context-aware inpainting

### Specialized Video Support
- Complete 360° video processing
- HDR video conversion
- Multi-view video generation

### User Experience
- Full GUI application with advanced features
- Web-based interface
- Mobile application
- Real-time preview

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation
- [FFmpeg](https://ffmpeg.org/) for video processing
- [OpenCV](https://opencv.org/) for computer vision utilities