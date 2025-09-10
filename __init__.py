"""
2D3D - 2D to 3D Video Converter
An open-source alternative to Owl3D
"""

__version__ = "0.1.0"
__author__ = "Open Source Community"

# Expose main classes
from .pipeline import Pipeline
from .depth_estimation.midas import DepthEstimator
from .stereo_warp.warper import StereoWarper
from .inpainting.basic import BasicInpainter

__all__ = [
    "Pipeline",
    "DepthEstimator",
    "StereoWarper",
    "BasicInpainter"
]