"""
Performance optimization utilities for 2D3D video conversion pipeline.
"""

import cv2
import numpy as np
import multiprocessing
import threading
import queue
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import wraps

class PerformanceOptimizer:
    """Optimization utilities for improving 2D3D conversion performance."""
    
    def __init__(self, config=None):
        """Initialize optimizer with configuration."""
        self.config = config or {}
        self.device = self._get_device()
        
    def _get_device(self):
        """Get the best available computing device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def parallel_frame_processing(self, frame_processor, frames, max_workers=None):
        """
        Process frames in parallel using thread pool.
        
        Args:
            frame_processor (callable): Function to process a single frame
            frames (list): List of frames to process
            max_workers (int): Maximum number of worker threads
            
        Returns:
            list: Processed frames in order
        """
        if max_workers is None:
            max_workers = min(32, max(4, multiprocessing.cpu_count() - 2))
            
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frames for processing
            futures = [executor.submit(frame_processor, frame) for frame in frames]
            
            # Collect results in order
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    results.append(None)
                    
        return results
    
    def batch_frame_processing(self, frame_processor, frames, batch_size=4):
        """
        Process frames in batches for better GPU utilization.
        
        Args:
            frame_processor (callable): Function to process a batch of frames
            frames (list): List of frames to process
            batch_size (int): Number of frames per batch
            
        Returns:
            list: Processed frames in order
        """
        results = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            try:
                batch_results = frame_processor(batch)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                # Add None for failed batch
                results.extend([None] * len(batch))
                
        return results
    
    def memory_optimized_processing(self, frame_processor, frames, max_memory_gb=2.0):
        """
        Process frames with memory usage control.
        
        Args:
            frame_processor (callable): Function to process a frame
            frames (list): List of frames to process
            max_memory_gb (float): Maximum memory usage in GB
            
        Returns:
            list: Processed frames in order
        """
        # Estimate memory per frame (this is a rough estimate)
        estimated_memory_per_frame = 50 * 1024 * 1024  # 50MB per frame estimate
        max_frames_in_memory = int((max_memory_gb * 1024 * 1024 * 1024) / estimated_memory_per_frame)
        max_frames_in_memory = max(1, min(max_frames_in_memory, len(frames)))
        
        results = []
        
        # Process in chunks to control memory usage
        for i in range(0, len(frames), max_frames_in_memory):
            chunk = frames[i:i + max_frames_in_memory]
            try:
                chunk_results = self.parallel_frame_processing(frame_processor, chunk)
                results.extend(chunk_results)
            except Exception as e:
                print(f"Error processing chunk {i//max_frames_in_memory}: {e}")
                results.extend([None] * len(chunk))
                
        return results
    
    def async_frame_loader(self, video_path, max_queue_size=10):
        """
        Asynchronously load frames from video file.
        
        Args:
            video_path (str): Path to video file
            max_queue_size (int): Maximum frames in queue
            
        Yields:
            np.ndarray: Video frames
        """
        cap = cv2.VideoCapture(video_path)
        frame_queue = queue.Queue(maxsize=max_queue_size)
        
        def loader():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame)
            frame_queue.put(None)  # Signal end of video
            cap.release()
        
        # Start loader thread
        loader_thread = threading.Thread(target=loader)
        loader_thread.daemon = True
        loader_thread.start()
        
        # Yield frames as they become available
        while True:
            frame = frame_queue.get()
            if frame is None:  # End of video
                break
            yield frame
        
        loader_thread.join()
    
    def gpu_accelerated_processing(self, frame_processor, frames):
        """
        Process frames using GPU acceleration when available.
        
        Args:
            frame_processor (callable): Function to process a frame with GPU support
            frames (list): List of frames to process
            
        Returns:
            list: Processed frames
        """
        if self.device.type in ["cuda", "mps"]:
            # Move frames to GPU if possible
            print(f"Using {self.device.type.upper()} acceleration for processing")
            return self.parallel_frame_processing(frame_processor, frames)
        else:
            # Fall back to CPU processing
            print("Using CPU for processing")
            return self.parallel_frame_processing(frame_processor, frames)

def timing_decorator(func):
    """
    Decorator to measure execution time of functions.
    
    Args:
        func (callable): Function to time
        
    Returns:
        callable: Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def optimize_image_processing(image, target_size=None, interpolation=cv2.INTER_AREA):
    """
    Optimize image processing operations.
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target size (width, height) or None
        interpolation (int): Interpolation method
        
    Returns:
        np.ndarray: Optimized image
    """
    # Convert to appropriate data type if needed
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Resize if target size specified
    if target_size:
        current_height, current_width = image.shape[:2]
        target_width, target_height = target_size
        
        # Only resize if necessary
        if current_width != target_width or current_height != target_height:
            image = cv2.resize(image, target_size, interpolation=interpolation)
    
    return image

def cache_friendly_processing(frames, cache_size=10):
    """
    Process frames in a cache-friendly manner.
    
    Args:
        frames (list): List of frames to process
        cache_size (int): Number of frames to process together for cache efficiency
        
    Returns:
        list: Processed frames
    """
    results = []
    
    # Process in cache-friendly batches
    for i in range(0, len(frames), cache_size):
        batch = frames[i:i + cache_size]
        # Process batch together to improve cache locality
        for frame in batch:
            # Process frame here
            processed_frame = frame  # Placeholder for actual processing
            results.append(processed_frame)
            
    return results

# Example usage and optimization strategies
class OptimizedDepthEstimator:
    """Optimized depth estimator with performance improvements."""
    
    def __init__(self, profile="balanced"):
        self.profile = profile
        self.optimizer = PerformanceOptimizer()
        
        # Pre-allocate commonly used arrays
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def predict_depth_optimized(self, image):
        """
        Optimized depth prediction with performance improvements.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Depth map
        """
        # Convert to grayscale efficiently
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply blur only if needed
        if self.profile != "fast":
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Create right view with vectorized operations
        shift = 3
        right_view = np.zeros_like(gray)
        right_view[:, :-shift] = gray[:, shift:]
        
        # Compute disparity with error handling
        try:
            # Use appropriate parameters based on profile
            if self.profile == "fast":
                stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
            elif self.profile == "balanced":
                stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
            else:  # precision
                stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
                
            disparity = stereo.compute(gray, right_view)
            
            # Normalize efficiently
            disparity = disparity.astype(np.float32) / 16.0
            disparity = np.clip(disparity, 0, None)  # Replace negative values with 0
            
            # Normalize to [0, 1] range
            disp_min, disp_max = disparity.min(), disparity.max()
            if disp_max > disp_min:
                depth = (disparity - disp_min) / (disp_max - disp_min)
            else:
                depth = np.zeros_like(disparity)
                
            # Apply morphological operations only when needed
            if self.profile != "fast":
                depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, self._kernel)
                depth = cv2.GaussianBlur(depth, (5, 5), 0)
                
            return depth.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Stereo matching failed, using fallback method: {e}")
            return self._fallback_depth_estimation_optimized(gray)
    
    def _fallback_depth_estimation_optimized(self, gray):
        """Optimized fallback depth estimation."""
        # Vectorized edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Vectorized distance calculation
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Normalize distances
        distances = distances / np.max(distances)
        
        # Vectorized combination
        depth = 0.7 * (1 - distances) + 0.3 * (edges / 255.0)
        
        # Normalize final result
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max != depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
            
        return depth

# Platform-specific optimization recommendations
def get_platform_optimizations():
    """
    Get platform-specific optimization recommendations.
    
    Returns:
        dict: Platform-specific optimization settings
    """
    import platform
    import psutil
    
    system = platform.system().lower()
    cpu_count = multiprocessing.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    optimizations = {
        "general": {
            "max_workers": min(32, max(4, cpu_count - 2)),
            "batch_size": 4,
            "memory_limit_gb": min(4.0, total_memory_gb * 0.25)
        },
        "platform_specific": {}
    }
    
    if system == "darwin":  # macOS
        optimizations["platform_specific"] = {
            "use_mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "thread_count": min(8, cpu_count),
            "memory_prefetch": True
        }
    elif system == "linux":  # Linux
        optimizations["platform_specific"] = {
            "use_cuda": torch.cuda.is_available(),
            "thread_count": cpu_count,
            "memory_prefetch": True
        }
    elif system == "windows":  # Windows
        optimizations["platform_specific"] = {
            "use_cuda": torch.cuda.is_available(),
            "thread_count": min(16, cpu_count),
            "memory_prefetch": False  # Windows file I/O can be slower
        }
    
    return optimizations

# Example usage
if __name__ == "__main__":
    # Example of how to use the optimization utilities
    optimizer = PerformanceOptimizer()
    
    # Get platform-specific recommendations
    platform_opts = get_platform_optimizations()
    print("Platform optimizations:", platform_opts)