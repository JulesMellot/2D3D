"""
Device management utilities for cross-platform GPU acceleration.
Supports CUDA (Nvidia), MPS (Apple Silicon), and ROCm (AMD) with CPU fallback.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def get_device():
    """
    Get the best available device for computation.
    
    Returns:
        torch.device: Best available device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal) device")
    elif hasattr(torch, "has_mps") and torch.has_mps:
        device = torch.device("mps")
        logger.info("Using MPS (Metal) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device

def get_device_name():
    """Get a string name for the current device."""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name()})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS (Metal)"
    elif hasattr(torch, "has_mps") and torch.has_mps:
        return "MPS (Metal)"
    else:
        return "CPU"