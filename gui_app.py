"""Tkinter GUI for 2D3D video conversion.

Preview pane (source / depth map / anaglyph) with frame scrubbing, live
determinate progress with fps + ETA, and cancellable conversions.
"""

import os
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

PREVIEW_W, PREVIEW_H = 720, 405


class App:
    def __init__(self, root):
        self.root = root
        root.title("2D3D — 2D to 3D Video Converter")
        root.resizable(False, False)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.profile = tk.StringVar(value="fast")
        self.stereo_format = tk.StringVar(value="sbs")
        self.strength = tk.DoubleVar(value=0.03)
        self.smoothing = tk.DoubleVar(value=0.3)
        self.crf = tk.IntVar(value=18)
        self.preset = tk.StringVar(value="medium")
        self.eye_width = tk.IntVar(value=0)   # 0 = source resolution
        self.eye_height = tk.IntVar(value=0)
        self.spatial = tk.BooleanVar(value=False)
        self.preview_mode = tk.StringVar(value="source")

        self._cap = None
        self._src_frame = None
        self._photo = None            # Tk drops the image without a live reference
        self._estimators = {}         # profile -> depth estimator, loaded on demand
        self._preview_seq = 0         # only the latest async preview request wins
        self._cancel = threading.Event()
        self._auto_output = True      # output path still tracks the default name

        self._build()

    # ---------- layout ----------

    def _build(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(main, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#1c1c1e", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, fill="#777",
                                text="Open a video to see a preview")

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        for text, mode in (("Source", "source"), ("Depth map", "depth"), ("3D anaglyph", "anaglyph")):
            ttk.Radiobutton(bar, text=text, value=mode, variable=self.preview_mode,
                            command=self._refresh_preview).pack(side="left", padx=(0, 12))
        self.scrub = ttk.Scale(bar, from_=0, to=0)
        self.scrub.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.scrub.bind("<ButtonRelease-1>", lambda _e: self._seek())

        files = ttk.Frame(main)
        files.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="Input:").grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.input_path, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(files, text="Open…", command=self.browse_input).grid(row=0, column=2)
        ttk.Label(files, text="Output:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(files, textvariable=self.output_path, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Button(files, text="Change…", command=self.browse_output).grid(row=1, column=2, pady=(4, 0))

        opts = ttk.LabelFrame(main, text="Settings", padding=8)
        opts.grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)
        opts.columnconfigure(1, weight=1)
        opts.columnconfigure(3, weight=1)

        ttk.Label(opts, text="Depth profile:").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(opts, textvariable=self.profile, state="readonly", width=10,
                          values=["fast", "balanced", "precision"])
        cb.grid(row=0, column=1, sticky="w", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

        ttk.Label(opts, text="Format:").grid(row=0, column=2, sticky="w")
        fmt = ttk.Combobox(opts, textvariable=self.stereo_format, state="readonly", width=10,
                           values=["sbs", "tb", "anaglyph"])
        fmt.grid(row=0, column=3, sticky="w", padx=6)
        fmt.bind("<<ComboboxSelected>>", lambda _e: self._update_default_output())

        self._slider(opts, 1, 0, "3D strength:", self.strength, 0.005, 0.08, "%.3f",
                     on_release=self._refresh_preview)
        self._slider(opts, 1, 2, "Smoothing:", self.smoothing, 0.0, 0.9, "%.2f")
        self._slider(opts, 2, 0, "Quality (CRF):", self.crf, 0, 51, "%d")

        ttk.Label(opts, text="Encoding preset:").grid(row=2, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=self.preset, state="readonly", width=10,
                     values=["ultrafast", "veryfast", "fast", "medium", "slow", "veryslow"]).grid(
            row=2, column=3, sticky="w", padx=6)

        ttk.Label(opts, text="Resolution / eye:").grid(row=3, column=0, sticky="w")
        res = ttk.Frame(opts)
        res.grid(row=3, column=1, sticky="w", padx=6)
        ttk.Entry(res, textvariable=self.eye_width, width=6).pack(side="left")
        ttk.Label(res, text=" × ").pack(side="left")
        ttk.Entry(res, textvariable=self.eye_height, width=6).pack(side="left")
        ttk.Label(res, text=" (0 = source)").pack(side="left")

        if shutil.which("spatial"):
            ttk.Checkbutton(opts, text="Also export Apple spatial video (Vision Pro)",
                            variable=self.spatial).grid(row=3, column=2, columnspan=2, sticky="w")

        self.progress = ttk.Progressbar(main, maximum=100)
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(main, text="Ready")
        self.status.grid(row=5, column=0, columnspan=3, pady=(4, 8))

        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, columnspan=3)
        self.convert_btn = ttk.Button(actions, text="Convert", command=self.start_conversion)
        self.convert_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(actions, text="Cancel", command=self._cancel.set,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        self.reveal_btn = ttk.Button(actions, text="Show in Finder",
                                     command=lambda: subprocess.run(["open", "-R", self.output_path.get()]))
        self.reveal_btn.pack(side="left", padx=4)
        self.reveal_btn.pack_forget()

    def _slider(self, parent, row, col, label, var, frm, to, fmt, on_release=None):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w")
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=col + 1, sticky="ew", padx=6)
        val = ttk.Label(cell, text=fmt % var.get(), width=5)
        scale = ttk.Scale(cell, from_=frm, to=to, variable=var,
                          command=lambda v: val.config(text=fmt % float(v)))
        scale.pack(side="left", fill="x", expand=True)
        val.pack(side="left", padx=(4, 0))
        if on_release:
            scale.bind("<ButtonRelease-1>", lambda _e: on_release())

    # ---------- files ----------

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if not filename:
            return
        self.input_path.set(filename)
        self._auto_output = True
        self._update_default_output()
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(filename)
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scrub.configure(to=max(total - 1, 0))
        self.scrub.set(0)
        self._seek()

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output Video", defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)
            self._auto_output = False

    def _update_default_output(self):
        if not self._auto_output or not self.input_path.get():
            return
        src = self.input_path.get()
        base = os.path.splitext(os.path.basename(src))[0]
        self.output_path.set(os.path.join(
            os.path.dirname(src), f"{base}_stereo_{self.stereo_format.get()}.mp4"))

    # ---------- preview ----------

    def _seek(self):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(self.scrub.get())))
        ok, frame = self._cap.read()
        if ok:
            self._src_frame = frame
            self._refresh_preview()

    def _refresh_preview(self):
        if self._src_frame is None:
            return
        mode, frame = self.preview_mode.get(), self._src_frame
        if mode == "source":
            self._show(frame)
            return
        # depth / anaglyph need the model — compute off the UI thread
        self._preview_seq += 1
        seq = self._preview_seq
        profile = self.profile.get()
        strength = self.strength.get()
        if profile not in self._estimators:
            self.status.config(text=f"Loading {profile} depth model…")
        threading.Thread(target=self._compute_preview,
                         args=(seq, mode, frame, profile, strength), daemon=True).start()

    def _compute_preview(self, seq, mode, frame, profile, strength):
        try:
            from depth import create
            from pipeline import Pipeline

            if profile not in self._estimators:
                self._estimators[profile] = create(profile)
            est = self._estimators[profile]
            if mode == "depth":
                dm = est.predict(frame)
                out = cv2.applyColorMap((dm * 255).astype("uint8"), cv2.COLORMAP_INFERNO)
            else:
                pipe = Pipeline(depth_fn=est.predict, strength=strength, smoothing=0)
                left, right = pipe.process_frame(frame)
                out = Pipeline._compose(left, right, "anaglyph", (frame.shape[1], frame.shape[0]))
        except Exception as e:
            self.root.after(0, lambda: self.status.config(text=f"Preview failed: {e}"))
            return
        if seq == self._preview_seq:
            self.root.after(0, lambda: (self._show(out), self.status.config(text="Ready")))

    def _show(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        frame = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=self._photo)

    # ---------- conversion ----------

    def start_conversion(self):
        if not self.input_path.get() or not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Please select a valid input video")
            return
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output file")
            return
        self._cancel.clear()
        self.convert_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.reveal_btn.pack_forget()
        self.progress.config(value=0)
        self.status.config(text="Loading depth model…")
        threading.Thread(target=self._convert, daemon=True).start()

    def _on_progress(self, n, total, t0):
        elapsed = time.monotonic() - t0
        fps = n / elapsed if elapsed else 0
        if total:
            pct = 100 * n / total
            eta = (total - n) / fps if fps else 0
            text = f"Frame {n}/{total} — {pct:.0f}% — {fps:.1f} fps — ETA {int(eta // 60)}:{int(eta % 60):02d}"
        else:
            pct, text = 0, f"Frame {n} — {fps:.1f} fps"
        self.root.after(0, lambda: (self.progress.config(value=pct), self.status.config(text=text)))

    def _convert(self):
        try:
            from pipeline import Pipeline, make_spatial

            pipeline = Pipeline(profile=self.profile.get(), strength=self.strength.get(),
                                smoothing=self.smoothing.get())
            eye_res = None
            if self.eye_width.get() > 0 and self.eye_height.get() > 0:
                eye_res = (self.eye_width.get(), self.eye_height.get())
            output = self.output_path.get()
            t0 = time.monotonic()
            pipeline.convert(
                self.input_path.get(), output,
                format=self.stereo_format.get(), eye_resolution=eye_res,
                crf=self.crf.get(), preset=self.preset.get(),
                progress_callback=lambda n, total: self._on_progress(n, total, t0),
                should_stop=self._cancel.is_set,
            )
            if self._cancel.is_set():
                os.remove(output)
                self.root.after(0, self._done, None, True)
                return
            if self.spatial.get() and self.stereo_format.get() == "sbs":
                self.root.after(0, lambda: self.status.config(text="Creating spatial video…"))
                make_spatial(output)
            self.root.after(0, self._done, None, False)
        except Exception as e:
            self.root.after(0, self._done, str(e), False)

    def _done(self, error, cancelled):
        self.convert_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if cancelled:
            self.progress.config(value=0)
            self.status.config(text="Cancelled")
        elif error:
            self.status.config(text="Conversion failed")
            messagebox.showerror("Error", f"Conversion failed: {error}")
        else:
            self.progress.config(value=100)
            self.status.config(text="Done!")
            self.reveal_btn.pack(side="left", padx=4)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
