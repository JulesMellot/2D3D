"""Monocular depth estimation with Depth Anything V2 (torch or Core ML backend)."""

import os
import sys

import cv2
import numpy as np
import torch

MODELS = {
    "fast": "depth-anything/Depth-Anything-V2-Small-hf",       # Apache-2.0
    "balanced": "depth-anything/Depth-Anything-V2-Base-hf",    # CC-BY-NC-4.0
    "precision": "depth-anything/Depth-Anything-V2-Large-hf",  # CC-BY-NC-4.0
}


def create(profile="fast", device=None):
    """Fastest available estimator: Core ML (Neural Engine) on macOS for 'fast', else torch."""
    if profile == "fast" and device is None and sys.platform == "darwin":
        try:
            return CoreMLDepthEstimator()
        except Exception as e:
            print(f"Core ML backend unavailable ({e}), using torch", file=sys.stderr)
    return DepthEstimator(profile, device)


def _normalize(d):
    lo, hi = float(d.min()), float(d.max())
    if hi <= lo:
        return np.zeros_like(d, dtype=np.float32)
    return ((d - lo) / (hi - lo)).astype(np.float32)


class CoreMLDepthEstimator:
    """Depth Anything V2 Small via Apple's official Core ML export, runs on the Neural Engine."""

    PACKAGE = "DepthAnythingV2SmallF16.mlpackage"

    def __init__(self):
        import contextlib
        import io

        # importing coremltools probes optional converter deps (tensorflow, sklearn)
        # and floods the console with their import warnings — silence it
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import coremltools as ct
        from huggingface_hub import snapshot_download

        # local_dir: the Core ML compiler cannot follow the hub cache's symlinks
        path = snapshot_download("apple/coreml-depth-anything-v2-small",
                                 allow_patterns=[self.PACKAGE + "/*"],
                                 local_dir=os.path.expanduser("~/.cache/2d3d"))
        self.model = ct.models.MLModel(os.path.join(path, self.PACKAGE))

    def predict(self, frame):
        """BGR frame -> float32 depth map in [0, 1], same HxW, 1 = near."""
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((518, 392), Image.BILINEAR)  # fixed model input size
        depth = np.array(self.model.predict({"image": img})["depth"], dtype=np.float32)
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        return _normalize(depth)


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
        return _normalize(depth.float().cpu().numpy())
