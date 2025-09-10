"""
Simple GUI application for 2D3D video conversion.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from enhanced_pipeline import EnhancedPipeline

class D2D3GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D3D - 2D to 3D Video Converter")
        self.root.geometry("800x600")
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.depth_profile = tk.StringVar(value="balanced")
        self.stereo_format = tk.StringVar(value="sbs")
        self.baseline = tk.DoubleVar(value=0.05)
        self.max_dimension = tk.IntVar(value=0)  # 0 means no resizing
        
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
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Conversion Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        settings_frame.columnconfigure(1, weight=1)
        
        # Depth profile
        ttk.Label(settings_frame, text="Depth Profile:").grid(row=0, column=0, sticky=tk.W, pady=5)
        profile_combo = ttk.Combobox(settings_frame, textvariable=self.depth_profile,
                                   values=["fast", "balanced", "precision"], state="readonly")
        profile_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Stereo format
        ttk.Label(settings_frame, text="Output Format:").grid(row=1, column=0, sticky=tk.W, pady=5)
        format_combo = ttk.Combobox(settings_frame, textvariable=self.stereo_format,
                                  values=["sbs", "tb", "anaglyph"], state="readonly")
        format_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Baseline
        ttk.Label(settings_frame, text="Stereo Baseline:").grid(row=2, column=0, sticky=tk.W, pady=5)
        baseline_scale = ttk.Scale(settings_frame, from_=0.01, to=0.2, 
                                 variable=self.baseline, orient=tk.HORIZONTAL)
        baseline_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        baseline_label = ttk.Label(settings_frame, textvariable=self.baseline)
        baseline_label.grid(row=2, column=2, padx=(10, 0))
        
        # Max dimension
        ttk.Label(settings_frame, text="Max Dimension:").grid(row=3, column=0, sticky=tk.W, pady=5)
        dim_frame = ttk.Frame(settings_frame)
        dim_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        dim_frame.columnconfigure(0, weight=1)
        ttk.Entry(dim_frame, textvariable=self.max_dimension).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(dim_frame, text="(0 = no resizing)").grid(row=0, column=1, padx=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Convert button
        self.convert_button = ttk.Button(main_frame, text="Convert Video", 
                                       command=self.start_conversion)
        self.convert_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
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
            
        # Start conversion in separate thread
        self.convert_button.config(state="disabled", text="Converting...")
        self.progress.start()
        self.status_label.config(text="Converting video...")
        
        # Run conversion in thread to prevent UI freezing
        thread = threading.Thread(target=self.convert_video)
        thread.daemon = True
        thread.start()
        
    def convert_video(self):
        try:
            # Create pipeline
            pipeline = EnhancedPipeline(
                depth_profile=self.depth_profile.get(),
                baseline=self.baseline.get()
            )
            
            # Get max dimension (0 means no resizing)
            max_dim = self.max_dimension.get() if self.max_dimension.get() > 0 else None
            
            # Convert video
            pipeline.convert_video(
                self.input_path.get(),
                self.output_path.get(),
                format=self.stereo_format.get(),
                max_dimension=max_dim
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
        messagebox.showinfo("Success", "Video conversion completed successfully!")
        
    def conversion_error(self, error_message):
        self.progress.stop()
        self.convert_button.config(state="normal", text="Convert Video")
        self.status_label.config(text="Conversion failed")
        messagebox.showerror("Error", f"Conversion failed: {error_message}")

def main():
    root = tk.Tk()
    app = D2D3GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()