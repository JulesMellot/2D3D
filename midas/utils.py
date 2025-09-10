"""
Simple utility functions for MiDaS
"""

import numpy as np
import cv2

def write_depth(depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # Convert to disparity if needed
    return depth

def read_image(path):
    """Read image and convert to RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def normalize_image(img):
    """Normalize image to [0, 1] range."""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img