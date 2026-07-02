"""2D -> stereoscopic 3D video conversion: depth estimation, DIBR warp, ffmpeg encode.

Frames stream from OpenCV straight into an ffmpeg stdin pipe — no temp files,
original fps and audio preserved.
"""

import shutil
import subprocess

import cv2
import numpy as np


class Pipeline:
    def __init__(self, profile="fast", strength=0.03, smoothing=0.3, depth_fn=None):
        """
        Args:
            profile: depth model size ("fast", "balanced", "precision")
            strength: max disparity as a fraction of frame width
            smoothing: EMA weight of the previous depth map, 0 disables
            depth_fn: override depth function (BGR frame -> float32 [0,1] HxW, 1 = near)
        """
        if depth_fn is None:
            from depth import create
            depth_fn = create(profile).predict
        self.depth_fn = depth_fn
        self.strength = strength
        self.smoothing = smoothing
        self._prev_depth = None

    def convert(self, input_path, output_path, format="sbs", eye_resolution=None,
                crf=18, preset="medium", progress_callback=None):
        """Convert a 2D video to stereo 3D. format: "sbs", "tb" or "anaglyph"."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ew, eh = eye_resolution or (src_w, src_h)
        ew, eh = ew - ew % 2, eh - eh % 2  # x264 wants even dimensions
        out_w, out_h = {"sbs": (ew * 2, eh), "tb": (ew, eh * 2), "anaglyph": (ew, eh)}[format]

        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
             "-r", f"{fps}", "-i", "-",
             "-i", input_path,
             "-map", "0:v", "-map", "1:a?",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-c:a", "copy",
             output_path],
            stdin=subprocess.PIPE,
        )

        self._prev_depth = None
        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                left, right = self.process_frame(frame)
                encoder.stdin.write(self._compose(left, right, format, (ew, eh)).tobytes())
                n += 1
                if progress_callback and n % 5 == 0:
                    progress_callback(f"Frame {n}/{total or '?'}")
        finally:
            cap.release()
            encoder.stdin.close()
            encoder.wait()

        if n == 0:
            raise ValueError(f"No frames decoded from {input_path}")
        if encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {encoder.returncode}")
        return output_path

    def process_frame(self, frame):
        """Left eye = original frame (quality preserved), right eye = depth-warped."""
        depth = self.depth_fn(frame)
        if self.smoothing and self._prev_depth is not None and self._prev_depth.shape == depth.shape:
            depth = (1 - self.smoothing) * depth + self.smoothing * self._prev_depth
        self._prev_depth = depth

        h, w = depth.shape
        # ponytail: naive backward warp stretches pixels at disocclusions;
        # diffusion inpainting (StereoCrafter-style) is the upgrade path.
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + depth * (self.strength * w)
        map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
        right = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return frame, right

    @staticmethod
    def _compose(left, right, format, eye_size):
        if (left.shape[1], left.shape[0]) != eye_size:
            left = cv2.resize(left, eye_size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, eye_size, interpolation=cv2.INTER_AREA)
        if format == "sbs":
            return np.ascontiguousarray(np.hstack((left, right)))
        if format == "tb":
            return np.ascontiguousarray(np.vstack((left, right)))
        # anaglyph: red channel from left eye, green/blue from right (BGR layout)
        out = right.copy()
        out[:, :, 2] = left[:, :, 2]
        return out


def make_spatial(sbs_path, output_path=None):
    """Convert an SBS video to Apple spatial video (MV-HEVC) for Vision Pro.

    Needs the `spatial` CLI: brew install spatial
    """
    if not shutil.which("spatial"):
        raise RuntimeError("`spatial` CLI not found — install it with: brew install spatial")
    output_path = output_path or sbs_path.rsplit(".", 1)[0] + "_spatial.mov"
    subprocess.run(["spatial", "make", "-i", sbs_path, "-f", "sbs", "-o", output_path], check=True)
    return output_path
