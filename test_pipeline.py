"""Self-check: run a synthetic video through the pipeline with a fake depth model.

    python test_pipeline.py
"""

import os
import subprocess
import tempfile

import cv2
import numpy as np

from pipeline import Pipeline


def fake_depth(frame):
    """Horizontal gradient: left edge far (0), right edge near (1)."""
    h, w = frame.shape[:2]
    return np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))


def main():
    # warp: left eye untouched, right eye actually warped
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    left, right = Pipeline(depth_fn=fake_depth).process_frame(frame)
    assert np.array_equal(left, frame)
    assert not np.array_equal(right, frame)

    # end-to-end: synthetic clip through every format, check output geometry
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=duration=1:size=320x240:rate=12", src], check=True)
        expected = {"sbs": (640, 240), "tb": (320, 480), "anaglyph": (320, 240)}
        for fmt, size in expected.items():
            out = os.path.join(td, f"out_{fmt}.mp4")
            Pipeline(depth_fn=fake_depth).convert(src, out, format=fmt)
            cap = cv2.VideoCapture(out)
            ok, _ = cap.read()
            got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            assert ok, f"{fmt}: no readable frame"
            assert got == size, f"{fmt}: expected {size}, got {got}"
            assert round(fps) == 12, f"{fmt}: source fps not preserved, got {fps}"

    print("OK")


if __name__ == "__main__":
    main()
