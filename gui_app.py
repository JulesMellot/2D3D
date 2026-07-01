"""Tkinter GUI for 2D3D video conversion."""

import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk


class D2D3GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D3D - 2D to 3D Video Converter")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.profile = tk.StringVar(value="fast")
        self.stereo_format = tk.StringVar(value="sbs")
        self.strength = tk.DoubleVar(value=0.03)
        self.smoothing = tk.DoubleVar(value=0.3)
        self.eye_width = tk.IntVar(value=0)   # 0 = source resolution
        self.eye_height = tk.IntVar(value=0)
        self.crf = tk.IntVar(value=18)
        self.preset = tk.StringVar(value="medium")
        self.spatial = tk.BooleanVar(value=False)
        self.preview_image = None

        self._build()

    def _build(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="2D to 3D Video Converter", font=("Arial", 16, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 15))

        # Input / output
        ttk.Label(frame, text="Input video:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.input_path, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Browse...", command=self.browse_input).grid(row=1, column=2)

        ttk.Label(frame, text="Output video:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.output_path, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Browse...", command=self.browse_output).grid(row=2, column=2)

        # Preview
        self.preview_label = ttk.Label(frame)
        self.preview_label.grid(row=3, column=0, columnspan=3, pady=8)

        # Settings
        ttk.Label(frame, text="Depth profile:").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.profile, state="readonly",
                     values=["fast", "balanced", "precision"]).grid(row=4, column=1, sticky="ew", padx=5)

        ttk.Label(frame, text="Format:").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.stereo_format, state="readonly",
                     values=["sbs", "tb", "anaglyph"]).grid(row=5, column=1, sticky="ew", padx=5)

        ttk.Label(frame, text="3D strength:").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Scale(frame, from_=0.005, to=0.08, variable=self.strength,
                  orient=tk.HORIZONTAL).grid(row=6, column=1, sticky="ew", padx=5)
        ttk.Label(frame, textvariable=self.strength).grid(row=6, column=2)

        ttk.Label(frame, text="Temporal smoothing:").grid(row=7, column=0, sticky="w", pady=3)
        ttk.Scale(frame, from_=0.0, to=0.9, variable=self.smoothing,
                  orient=tk.HORIZONTAL).grid(row=7, column=1, sticky="ew", padx=5)
        ttk.Label(frame, textvariable=self.smoothing).grid(row=7, column=2)

        ttk.Label(frame, text="Resolution per eye:").grid(row=8, column=0, sticky="w", pady=3)
        res = ttk.Frame(frame)
        res.grid(row=8, column=1, sticky="w", padx=5)
        ttk.Entry(res, textvariable=self.eye_width, width=7).pack(side="left")
        ttk.Label(res, text=" x ").pack(side="left")
        ttk.Entry(res, textvariable=self.eye_height, width=7).pack(side="left")
        ttk.Label(res, text="  (0 = source)").pack(side="left")

        ttk.Label(frame, text="Quality (CRF):").grid(row=9, column=0, sticky="w", pady=3)
        ttk.Scale(frame, from_=0, to=51, variable=self.crf,
                  orient=tk.HORIZONTAL).grid(row=9, column=1, sticky="ew", padx=5)
        ttk.Label(frame, textvariable=self.crf).grid(row=9, column=2)

        ttk.Label(frame, text="Encoding preset:").grid(row=10, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.preset, state="readonly",
                     values=["ultrafast", "veryfast", "fast", "medium", "slow", "veryslow"]).grid(
            row=10, column=1, sticky="ew", padx=5)

        if shutil.which("spatial"):
            ttk.Checkbutton(frame, text="Also export Apple spatial video (Vision Pro)",
                            variable=self.spatial).grid(row=11, column=0, columnspan=2, sticky="w", pady=3)

        # Progress + action
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(10, 3))
        self.status_label = ttk.Label(frame, text="Ready")
        self.status_label.grid(row=13, column=0, columnspan=3)
        self.convert_button = ttk.Button(frame, text="Convert Video", command=self.start_conversion)
        self.convert_button.grid(row=14, column=0, columnspan=3, pady=12)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not filename:
            return
        self.input_path.set(filename)
        if not self.output_path.get():
            base = os.path.splitext(os.path.basename(filename))[0]
            self.output_path.set(os.path.join(
                os.path.dirname(filename), f"{base}_stereo_{self.stereo_format.get()}.mp4"))
        self.load_preview()

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output Video", defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)

    def load_preview(self):
        cap = cv2.VideoCapture(self.input_path.get())
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return
        h, w = frame.shape[:2]
        scale = min(200 / h, 320 / w, 1.0)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.preview_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.preview_label.configure(image=self.preview_image)

    def start_conversion(self):
        if not self.input_path.get() or not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Please select a valid input video")
            return
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output file")
            return
        self.convert_button.config(state="disabled", text="Converting...")
        self.progress.start()
        self.status_label.config(text="Loading depth model...")
        threading.Thread(target=self._convert, daemon=True).start()

    def _convert(self):
        try:
            from pipeline import Pipeline, make_spatial

            pipeline = Pipeline(profile=self.profile.get(), strength=self.strength.get(),
                                smoothing=self.smoothing.get())
            eye_res = None
            if self.eye_width.get() > 0 and self.eye_height.get() > 0:
                eye_res = (self.eye_width.get(), self.eye_height.get())
            output = self.output_path.get()
            pipeline.convert(
                self.input_path.get(), output,
                format=self.stereo_format.get(), eye_resolution=eye_res,
                crf=self.crf.get(), preset=self.preset.get(),
                progress_callback=lambda msg: self.root.after(
                    0, lambda: self.status_label.config(text=msg)),
            )
            if self.spatial.get() and self.stereo_format.get() == "sbs":
                self.root.after(0, lambda: self.status_label.config(text="Creating spatial video..."))
                make_spatial(output)
            self.root.after(0, self._done, None)
        except Exception as e:
            self.root.after(0, self._done, str(e))

    def _done(self, error):
        self.progress.stop()
        self.convert_button.config(state="normal", text="Convert Video")
        if error:
            self.status_label.config(text="Conversion failed")
            messagebox.showerror("Error", f"Conversion failed: {error}")
        else:
            self.status_label.config(text="Done!")
            messagebox.showinfo("Success", "Video conversion completed!")


def main():
    root = tk.Tk()
    D2D3GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
