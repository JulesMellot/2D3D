"""
Simple GUI application for 2D3D video conversion.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import tempfile
from enhanced_pipeline import EnhancedPipeline
from utils.config import get_config

class D2D3GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D3D - 2D to 3D Video Converter")
        self.root.geometry("900x750")
        
        # Load configuration
        self.config = get_config()
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.temp_dir = tk.StringVar()
        self.depth_profile = tk.StringVar(value=self.config.get("depth_profile", "balanced"))
        self.stereo_format = tk.StringVar(value=self.config.get("default_format", "sbs"))
        self.baseline = tk.DoubleVar(value=self.config.get("baseline", 0.05))
        self.focal_length = tk.DoubleVar(value=self.config.get("focal_length", 1000))
        self.max_dimension = tk.IntVar(value=0)  # 0 means no resizing
        self.preserve_colors = tk.BooleanVar(value=self.config.get("preserve_colors", True))
        self.temporal_smoothing = tk.BooleanVar(value=self.config.get("temporal_smoothing", True))
        self.post_process = tk.BooleanVar(value=self.config.get("post_process", False))
        self.temporal_smoothing_factor = tk.DoubleVar(value=self.config.get("temporal_smoothing_factor", 0.1))
        self.crf = tk.IntVar(value=self.config.get("crf", 18))
        self.preset = tk.StringVar(value=self.config.get("preset", "medium"))
        self.batch_size = tk.IntVar(value=self.config.get("batch_size", 1))
        self.gpu_acceleration = tk.BooleanVar(value=self.config.get("gpu_acceleration", True))
        self.auto_settings = tk.BooleanVar(value=True)  # Enable auto settings by default
        
        # Output resolution variables
        default_res = self.config.get("default_resolution", [960, 1080])
        self.output_width = tk.IntVar(value=default_res[0])
        self.output_height = tk.IntVar(value=default_res[1])
        
        # Preset variables
        self.current_preset = tk.StringVar(value="balanced")
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="2D to 3D Video Converter", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input file selection
        ttk.Label(main_frame, text="Input Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_frame, textvariable=self.input_path, state="readonly").grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(
            row=0, column=1)
        
        # Output file selection
        ttk.Label(main_frame, text="Output Video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_path, state="readonly").grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(
            row=0, column=1)
        
        # Temp directory selection
        ttk.Label(main_frame, text="Temp Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        temp_frame = ttk.Frame(main_frame)
        temp_frame.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        temp_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(temp_frame, textvariable=self.temp_dir, state="readonly").grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(temp_frame, text="Browse...", command=self.browse_temp_dir).grid(
            row=0, column=1)
        
        # Presets
        ttk.Label(main_frame, text="Preset:").grid(row=4, column=0, sticky=tk.W, pady=5)
        preset_frame = ttk.Frame(main_frame)
        preset_frame.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        preset_frame.columnconfigure(0, weight=1)
        
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.current_preset,
                                   values=["fast", "balanced", "quality", "3dtv", "vr", "custom"], 
                                   state="readonly")
        preset_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)
        ttk.Button(preset_frame, text="Apply", command=self.apply_preset).grid(row=0, column=1)
        ttk.Button(preset_frame, text="Save As...", command=self.save_preset).grid(row=0, column=2, padx=(5, 0))
        
        # Settings notebook (tabs)
        settings_notebook = ttk.Notebook(main_frame)
        settings_notebook.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Basic Settings Tab
        basic_frame = ttk.Frame(settings_notebook, padding="10")
        settings_notebook.add(basic_frame, text="Basic")
        basic_frame.columnconfigure(1, weight=1)
        
        # Depth profile
        ttk.Label(basic_frame, text="Depth Profile:").grid(row=0, column=0, sticky=tk.W, pady=5)
        profile_combo = ttk.Combobox(basic_frame, textvariable=self.depth_profile,
                                   values=["fast", "balanced", "precision"], state="readonly")
        profile_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Stereo format
        ttk.Label(basic_frame, text="Output Format:").grid(row=1, column=0, sticky=tk.W, pady=5)
        format_combo = ttk.Combobox(basic_frame, textvariable=self.stereo_format,
                                  values=["sbs", "tb", "anaglyph"], state="readonly")
        format_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Output resolution per eye
        ttk.Label(basic_frame, text="Output Resolution per Eye:").grid(row=2, column=0, sticky=tk.W, pady=5)
        res_frame = ttk.Frame(basic_frame)
        res_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        res_frame.columnconfigure(0, weight=1)
        res_frame.columnconfigure(2, weight=1)
        
        ttk.Entry(res_frame, textvariable=self.output_width, width=10).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(res_frame, text="x").grid(row=0, column=1, padx=5)
        ttk.Entry(res_frame, textvariable=self.output_height, width=10).grid(row=0, column=2, sticky=tk.W)
        ttk.Label(res_frame, text="(width x height per eye)").grid(row=0, column=3, padx=(10, 0))
        
        # Baseline
        ttk.Label(basic_frame, text="Stereo Baseline:").grid(row=3, column=0, sticky=tk.W, pady=5)
        baseline_scale = ttk.Scale(basic_frame, from_=0.01, to=0.2, 
                                 variable=self.baseline, orient=tk.HORIZONTAL)
        baseline_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        baseline_label = ttk.Label(basic_frame, textvariable=self.baseline)
        baseline_label.grid(row=3, column=2, padx=(10, 0))
        
        # Max dimension
        ttk.Label(basic_frame, text="Max Dimension:").grid(row=4, column=0, sticky=tk.W, pady=5)
        dim_frame = ttk.Frame(basic_frame)
        dim_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        dim_frame.columnconfigure(0, weight=1)
        ttk.Entry(dim_frame, textvariable=self.max_dimension).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(dim_frame, text="(0 = no resizing)").grid(row=0, column=1, padx=(10, 0))
        
        # Advanced Settings Tab
        advanced_frame = ttk.Frame(settings_notebook, padding="10")
        settings_notebook.add(advanced_frame, text="Advanced")
        advanced_frame.columnconfigure(1, weight=1)
        
        # Focal length
        ttk.Label(advanced_frame, text="Focal Length:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(advanced_frame, textvariable=self.focal_length).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Temporal smoothing factor
        ttk.Label(advanced_frame, text="Temporal Smoothing Factor:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(advanced_frame, from_=0.0, to=1.0, variable=self.temporal_smoothing_factor, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(advanced_frame, textvariable=self.temporal_smoothing_factor).grid(row=1, column=2, padx=(10, 0))
        
        # CRF
        ttk.Label(advanced_frame, text="CRF (Quality):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(advanced_frame, from_=0, to=51, variable=self.crf, orient=tk.HORIZONTAL).grid(
            row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(advanced_frame, textvariable=self.crf).grid(row=2, column=2, padx=(10, 0))
        
        # Preset
        ttk.Label(advanced_frame, text="Encoding Preset:").grid(row=3, column=0, sticky=tk.W, pady=5)
        preset_combo = ttk.Combobox(advanced_frame, textvariable=self.preset,
                                  values=["ultrafast", "superfast", "veryfast", "faster", "fast", 
                                         "medium", "slow", "slower", "veryslow"], state="readonly")
        preset_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Batch size
        ttk.Label(advanced_frame, text="Batch Size:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(advanced_frame, textvariable=self.batch_size).grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Checkboxes
        ttk.Checkbutton(advanced_frame, text="Preserve Colors", variable=self.preserve_colors).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(advanced_frame, text="Temporal Smoothing", variable=self.temporal_smoothing).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(advanced_frame, text="Post Process", variable=self.post_process).grid(
            row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(advanced_frame, text="GPU Acceleration", variable=self.gpu_acceleration).grid(
            row=8, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(advanced_frame, text="Auto Settings (detect aspect ratio and optimize)", variable=self.auto_settings).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Progress label for detailed information
        self.progress_label = ttk.Label(main_frame, text="", font=("Arial", 9))
        self.progress_label.grid(row=7, column=0, columnspan=3, pady=2)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=8, column=0, columnspan=3, pady=5)
        
        # Convert button
        self.convert_button = ttk.Button(main_frame, text="Convert Video", 
                                       command=self.start_conversion)
        self.convert_button.grid(row=9, column=0, columnspan=3, pady=20)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
    def on_preset_selected(self, event=None):
        """Handle preset selection."""
        preset = self.current_preset.get()
        if preset != "custom":
            self.apply_preset()
        
    def apply_preset(self):
        """Apply selected preset with industry-standard parameters."""
        preset = self.current_preset.get()
        
        if preset == "fast":
            # Fast preset - optimized for speed with reasonable quality
            self.depth_profile.set("fast")
            self.baseline.set(0.03)  # Lower baseline for faster processing
            self.preset.set("veryfast")  # Fast encoding
            self.crf.set(23)  # Medium quality
            self.temporal_smoothing.set(False)  # Disable for speed
            self.post_process.set(False)  # Disable for speed
            self.output_width.set(480)  # Lower resolution
            self.output_height.set(270)  # Lower resolution
            self.focal_length.set(1000)  # Standard focal length
            self.temporal_smoothing_factor.set(0.05)  # Minimal smoothing
        elif preset == "balanced":
            # Balanced preset - good compromise between speed and quality
            self.depth_profile.set("balanced")
            self.baseline.set(0.05)  # Standard baseline (6.5cm equivalent)
            self.preset.set("medium")  # Balanced encoding
            self.crf.set(18)  # High quality
            self.temporal_smoothing.set(True)  # Enable smoothing
            self.post_process.set(False)  # No post-processing for speed
            self.output_width.set(960)  # Standard resolution for SBS
            self.output_height.set(540)  # Standard resolution for SBS
            self.focal_length.set(1000)  # Standard focal length
            self.temporal_smoothing_factor.set(0.1)  # Moderate smoothing
        elif preset == "quality":
            # Quality preset - highest quality settings
            self.depth_profile.set("precision")
            self.baseline.set(0.06)  # Slightly higher baseline for stronger 3D effect
            self.preset.set("slow")  # High quality encoding
            self.crf.set(15)  # Very high quality
            self.temporal_smoothing.set(True)  # Enable smoothing
            self.post_process.set(True)  # Enable post-processing
            self.output_width.set(1920)  # Full HD resolution
            self.output_height.set(1080)  # Full HD resolution
            self.focal_length.set(1200)  # Slightly longer focal length
            self.temporal_smoothing_factor.set(0.15)  # Strong smoothing
        elif preset == "3dtv":
            # 3D TV preset - optimized for comfortable 3D TV viewing
            self.depth_profile.set("balanced")
            self.baseline.set(0.065)  # Matches human interpupillary distance (6.5cm)
            self.preset.set("medium")  # Balanced encoding
            self.crf.set(18)  # High quality
            self.temporal_smoothing.set(True)  # Enable smoothing
            self.post_process.set(False)  # No post-processing
            self.output_width.set(960)  # Standard for SBS 3D TV
            self.output_height.set(1080)  # Standard for SBS 3D TV
            self.focal_length.set(1000)  # Standard focal length
            self.temporal_smoothing_factor.set(0.1)  # Moderate smoothing
        elif preset == "vr":
            # VR preset - optimized for virtual reality viewing
            self.depth_profile.set("precision")
            self.baseline.set(0.05)  # Standard baseline
            self.preset.set("slow")  # High quality encoding
            self.crf.set(15)  # Very high quality
            self.temporal_smoothing.set(True)  # Enable smoothing
            self.post_process.set(True)  # Enable post-processing
            self.output_width.set(3840)  # 4K resolution
            self.output_height.set(2160)  # 4K resolution
            self.focal_length.set(800)  # Shorter focal length for more depth
            self.temporal_smoothing_factor.set(0.2)  # Strong smoothing
        # For "custom", we keep the current settings
        
        self.status_label.config(text=f"Preset '{preset}' applied with optimized parameters")
        
    def save_preset(self):
        """Save current settings as a custom preset."""
        # In a more advanced implementation, we could save to a file
        # For now, we'll just set the preset to "custom"
        self.current_preset.set("custom")
        self.status_label.config(text="Current settings saved as custom preset")
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            if not self.output_path.get():
                base_name = os.path.splitext(os.path.basename(filename))[0]
                output_filename = f"{base_name}_stereo_{self.stereo_format.get()}.mp4"
                self.output_path.set(os.path.join(os.path.dirname(filename), output_filename))
                
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
            
    def browse_temp_dir(self):
        directory = filedialog.askdirectory(title="Select Temporary Directory")
        if directory:
            self.temp_dir.set(directory)
            
    def start_conversion(self):
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input video file")
            return
            
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output video file")
            return
            
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return
            
        # Validate temp directory if specified
        if self.temp_dir.get() and not os.path.exists(self.temp_dir.get()):
            messagebox.showerror("Error", "Temporary directory does not exist")
            return
            
        if self.temp_dir.get() and not os.path.isdir(self.temp_dir.get()):
            messagebox.showerror("Error", "Temporary path is not a directory")
            return
            
        # Start conversion in separate thread
        self.convert_button.config(state="disabled", text="Converting...")
        self.progress.start()
        self.status_label.config(text="Converting video...")
        self.progress_label.config(text="Extracting frames...")
        
        # Run conversion in thread to prevent UI freezing
        thread = threading.Thread(target=self.convert_video)
        thread.daemon = True
        thread.start()
        
    def convert_video(self):
        try:
            # Create pipeline
            pipeline = EnhancedPipeline(
                depth_profile=self.depth_profile.get(),
                baseline=self.baseline.get(),
                focal_length=self.focal_length.get()
            )
            
            # Set pipeline properties based on GUI settings
            pipeline.preserve_colors = self.preserve_colors.get()
            pipeline.temporal_smoothing = self.temporal_smoothing.get()
            pipeline.post_process = self.post_process.get()
            pipeline.smoothing_factor = self.temporal_smoothing_factor.get()
            
            # Get output resolution
            output_resolution = (self.output_width.get(), self.output_height.get())
            
            # Get max dimension (0 means no resizing)
            max_dim = self.max_dimension.get() if self.max_dimension.get() > 0 else None
            
            # Get temp directory
            temp_dir = self.temp_dir.get() if self.temp_dir.get() else None
            
            # Define progress callback
            def progress_callback(status_message):
                self.root.after(0, lambda: self.update_progress(status_message))
            
            # Convert video
            pipeline.convert_video(
                self.input_path.get(),
                self.output_path.get(),
                format=self.stereo_format.get(),
                max_dimension=max_dim,
                temp_dir=temp_dir,
                output_resolution=output_resolution,
                auto_settings=self.auto_settings.get(),
                progress_callback=progress_callback
            )
            
            # Update UI on success
            self.root.after(0, self.conversion_success)
            
        except Exception as e:
            # Update UI on error
            self.root.after(0, self.conversion_error, str(e))
            
    def conversion_success(self):
        self.progress.stop()
        self.convert_button.config(state="normal", text="Convert Video")
        self.status_label.config(text="Conversion completed successfully!")
        self.progress_label.config(text="")
        messagebox.showinfo("Success", "Video conversion completed successfully!")
        
    def conversion_error(self, error_message):
        self.progress.stop()
        self.convert_button.config(state="normal", text="Convert Video")
        self.status_label.config(text="Conversion failed")
        self.progress_label.config(text="")
        messagebox.showerror("Error", f"Conversion failed: {error_message}")
        
    def update_progress(self, message):
        """Update progress status in GUI."""
        # Update the progress label with detailed information
        self.progress_label.config(text=message)
        # Also update the main status label
        self.status_label.config(text=message)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = D2D3GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()