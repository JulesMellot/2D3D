"""Self-check: run a synthetic video through the pipeline with a fake depth model.

    python test_pipeline.py
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

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

    # Both eyes get their own centered pillarbox, before the half-width squeeze.
    red = np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8)
    green = np.full((240, 320, 3), (0, 255, 0), dtype=np.uint8)
    packed = Pipeline._compose(red, green, "sbs", (320, 240), tv_mode=True)
    assert packed.shape == (240, 428, 3)
    for eye, color in zip(np.hsplit(packed, 2), ((0, 0, 255), (0, 255, 0))):
        assert np.all(eye[:, :27] == 0) and np.all(eye[:, -27:] == 0)
        assert np.all(eye[:, 27:-27] == color), "Each eye must stay centered without cropping"
    for width, height in ((320, 180), (480, 180)):
        assert Pipeline._tv_eye_size((width, height)) == (width // 2, height)

    for fmt in ("tb", "anaglyph"):
        try:
            Pipeline(depth_fn=fake_depth).convert("unused.mp4", "unused_out.mp4",
                                                  format=fmt, tv_mode=True)
        except ValueError as e:
            assert "requires SBS" in str(e)
        else:
            raise AssertionError(f"3D TV mode should reject {fmt}")

    # end-to-end: synthetic clip through every format, check output geometry
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=duration=1:size=320x240:rate=12",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:a", "aac", "-shortest", src], check=True)
        cases = [
            ("sbs", False, None, (640, 240), "mp4"),
            ("tb", False, None, (320, 480), "mp4"),
            ("anaglyph", False, None, (320, 240), "mp4"),
            ("sbs", True, None, (428, 240), "mp4"),
            # Odd compressed eye width is valid: the combined width is even.
            ("sbs", True, (322, 180), (322, 180), "mp4"),
            ("sbs", True, (320, 180), (320, 180), "mp4"),
            ("sbs", True, (480, 180), (480, 180), "mp4"),
            ("sbs", True, (240, 240), (428, 240), "mkv"),
            ("sbs", False, None, (640, 240), "mkv"),
            ("tb", False, None, (320, 480), "mkv"),
        ]
        for i, (fmt, tv_mode, eye_resolution, size, ext) in enumerate(cases):
            out = os.path.join(td, f"out_{i}_{fmt}.{ext}")
            Pipeline(depth_fn=fake_depth).convert(
                src, out, format=fmt, tv_mode=tv_mode, eye_resolution=eye_resolution)
            cap = cv2.VideoCapture(out)
            ok, _ = cap.read()
            got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            assert ok, f"{fmt}: no readable frame"
            assert got == size, f"{fmt}: expected {size}, got {got}"
            assert round(fps) == 12, f"{fmt}: source fps not preserved, got {fps}"
            audio = json.loads(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_name", "-of", "json", out], text=True))
            assert audio["streams"][0]["codec_name"] == "aac", "Audio must be preserved"

            probe = json.loads(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-read_intervals", "%+#1", "-show_frames", "-show_streams", "-of", "json", out],
                text=True))
            stereo = [sd for f in probe["frames"] for sd in f.get("side_data_list", [])
                      if sd["side_data_type"] == "Stereo 3D"]
            if fmt == "anaglyph":
                assert not stereo, "Anaglyph must not be flagged as packed stereo"
            else:
                expected_type = {"sbs": "side by side", "tb": "top and bottom"}[fmt]
                expected_mode = {"sbs": "left_right", "tb": "top_bottom"}[fmt]
                assert stereo, f"{fmt}: missing stereoscopic metadata"
                # New FFprobe versions report frame packing in frame tags;
                # older versions expose type and eye order in side data.
                if "type" in stereo[0]:
                    assert stereo[0]["type"] == expected_type, stereo
                    assert stereo[0]["inverted"] == 0, "Left eye should come first"
                else:
                    assert probe["frames"][0]["tags"]["stereo_mode"] == expected_mode
                if ext == "mkv":
                    assert probe["streams"][0]["tags"]["stereo_mode"] == expected_mode
                if ext == "mp4":
                    assert any(sd["side_data_type"] == "Stereo 3D"
                               for sd in probe["streams"][0].get("side_data_list", [])), (
                                   "MP4 must contain track-level stereo metadata")
                    # Verify st3d's actual layout byte as well. Some FFprobe
                    # versions overwrite its reported type when reading vexu.
                    data = Path(out).read_bytes()
                    marker = b"\x00\x00\x00\x0dst3d\x00\x00\x00\x00"
                    offset = data.find(marker)
                    assert offset >= 0, "MP4 is missing its st3d box"
                    assert data[offset + len(marker)] == (2 if fmt == "sbs" else 1)

    print("OK")


if __name__ == "__main__":
    main()
