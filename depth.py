"""Monocular depth estimation with Depth Anything V2 (via HuggingFace transformers)."""

import cv2
import numpy as np
import torch

MODELS = {
    "fast": "depth-anything/Depth-Anything-V2-Small-hf",       # Apache-2.0
    "balanced": "depth-anything/Depth-Anything-V2-Base-hf",    # CC-BY-NC-4.0
    "precision": "depth-anything/Depth-Anything-V2-Large-hf",  # CC-BY-NC-4.0
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthEstimator:
    def __init__(self, profile="fast", device=None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device or get_device()
        name = MODELS[profile]
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()
        if self.device == "cuda":
            self.model = torch.compile(self.model)

    @torch.inference_mode()
    def predict(self, frame):
        """BGR frame -> float32 depth map in [0, 1], same HxW, 1 = near."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth  # (1, h', w'), relative inverse depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        d = depth.float().cpu().numpy()
        lo, hi = float(d.min()), float(d.max())
        if hi <= lo:
            return np.zeros_like(d, dtype=np.float32)
        return ((d - lo) / (hi - lo)).astype(np.float32)
