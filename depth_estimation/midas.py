"""
Depth estimation using MiDaS model.
Supports multiple profiles: Fast, Balanced, Precision.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from utils.device import get_device

# Try to import MiDaS modules
try:
    from midas.dpt_depth import DPTDepthModel
    from midas import transforms
    import midas.utils as utils
except ImportError as e:
    # Fallback if midas is not properly installed
    print(f"MiDaS import error: {e}")
    DPTDepthModel = None
    transforms = None
    utils = None

class DepthEstimator:
    def __init__(self, model_type="DPT_Hybrid", profile="balanced"):
        """
        Initialize the depth estimator.
        
        Args:
            model_type (str): Model type ("DPT_Large", "DPT_Hybrid", "MiDaS_small")
            profile (str): Performance profile ("fast", "balanced", "precision")
        """
        self.device = get_device()
        self.profile = profile
        self.model_type = model_type
        
        # Profile-specific settings
        self._set_profile_settings()
        
        # Initialize model
        self._init_model()
        
    def _set_profile_settings(self):
        """Set model parameters based on selected profile."""
        if self.profile == "fast":
            self.model_type = "MiDaS_small"
            self.net_w, self.net_h = 256, 256
        elif self.profile == "balanced":
            self.model_type = "DPT_Hybrid"
            self.net_w, self.net_h = 384, 384
        elif self.profile == "precision":
            self.model_type = "DPT_Large"
            self.net_w, self.net_h = 384, 384
            
    def _init_model(self):
        """Initialize the MiDaS model."""
        if DPTDepthModel is None:
            raise ImportError("MiDaS is not properly installed.")
            
        # Create model based on type
        try:
            if self.model_type == "DPT_Large":
                self.model = DPTDepthModel(
                    path=None,
                    backbone="vitl16_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                )
            elif self.model_type == "DPT_Hybrid":
                self.model = DPTDepthModel(
                    path=None,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                )
            elif self.model_type == "MiDaS_small":
                self.model = DPTDepthModel(
                    path=None,
                    backbone="efficientnet_lite3",
                    non_negative=True,
                    enable_attention_hooks=False,
                )
        except Exception as e:
            print(f"Error initializing model with backbone {self.model_type}: {e}")
            # Fallback to DPT_Hybrid
            self.model = DPTDepthModel(
                path=None,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.model_type = "DPT_Hybrid"
            
        self.model.eval()
        self.model.to(self.device)
        
        # Set up transforms using MiDaS transforms
        if transforms is not None:
            self.transform = transforms.Resize(
                width=self.net_w,
                height=self.net_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
        else:
            # Fallback to torchvision transforms
            self.transform = Compose([
                Resize((self.net_h, self.net_w)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
    def predict_depth(self, image):
        """
        Predict depth map for an input image.
        
        Args:
            image (np.ndarray): Input image (H, W, 3) in BGR format
            
        Returns:
            np.ndarray: Depth map (H, W)
        """
        # Ensure image is large enough
        if image.shape[0] < 64 or image.shape[1] < 64:
            # Resize to minimum size
            scale_factor = max(64 / image.shape[0], 64 / image.shape[1])
            new_h = int(image.shape[0] * scale_factor)
            new_w = int(image.shape[1] * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Apply transformations using MiDaS transforms
        if transforms is not None and hasattr(self, 'transform'):
            img_input = self.transform({"image": image_rgb})["image"]
            # Convert to tensor and add batch dimension
            input_tensor = torch.from_numpy(img_input).unsqueeze(0).to(self.device)
        else:
            # Fallback to torchvision transforms
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        # Convert to numpy
        depth = prediction.cpu().numpy()
        
        # Normalize depth to [0, 1] range
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max != depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
            
        return depth
    
    def batch_predict(self, images):
        """
        Predict depth maps for a batch of images.
        
        Args:
            images (list): List of input images (H, W, 3) in BGR format
            
        Returns:
            list: List of depth maps
        """
        depths = []
        for image in images:
            depth = self.predict_depth(image)
            depths.append(depth)
        return depths

# Convenience functions
def create_depth_estimator(profile="balanced"):
    """
    Create a depth estimator with the specified profile.
    
    Args:
        profile (str): Performance profile ("fast", "balanced", "precision")
        
    Returns:
        DepthEstimator: Initialized depth estimator
    """
    return DepthEstimator(profile=profile)