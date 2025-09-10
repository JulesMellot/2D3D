"""
Progress monitoring utilities for video processing with accurate ETA calculation.
"""

import time
import sys
from collections import deque
import threading

class VideoProcessingProgress:
    """
    Advanced progress monitoring for video processing with accurate ETA calculation.
    """
    
    def __init__(self, total_frames, description="Processing"):
        self.total_frames = total_frames
        self.processed_frames = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_processed_frames = 0
        
        # For smoothing and prediction
        self.frame_times = deque(maxlen=50)  # Rolling window for frame times
        self.speed_history = deque(maxlen=30)  # Processing speed history
        
        # Threading for periodic updates
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def update(self, frames_processed=1):
        """
        Update progress with number of frames processed.
        
        Args:
            frames_processed (int): Number of frames processed since last update
        """
        with self._lock:
            current_time = time.time()
            
            # Update counters
            self.processed_frames += frames_processed
            
            # Record timing information
            elapsed_since_last = current_time - self.last_update_time
            if elapsed_since_last > 0 and frames_processed > 0:
                time_per_frame = elapsed_since_last / frames_processed
                self.frame_times.append(time_per_frame)
                
                # Calculate processing speed (frames per second)
                speed = frames_processed / elapsed_since_last
                self.speed_history.append(speed)
            
            self.last_update_time = current_time
            self.last_processed_frames = self.processed_frames
    
    def get_progress_percentage(self):
        """Get current progress as percentage."""
        if self.total_frames <= 0:
            return 0
        return min(100.0, (self.processed_frames / self.total_frames) * 100)
    
    def get_eta_simple(self):
        """
        Simple ETA calculation based on average speed.
        
        Returns:
            float: ETA in seconds, or None if not enough data
        """
        if self.processed_frames == 0:
            return None
            
        elapsed = time.time() - self.start_time
        avg_time_per_frame = elapsed / self.processed_frames
        remaining_frames = self.total_frames - self.processed_frames
        return avg_time_per_frame * remaining_frames
    
    def get_eta_smoothed(self):
        """
        ETA with smoothing using moving average.
        
        Returns:
            float: ETA in seconds, or None if not enough data
        """
        if len(self.frame_times) < 5:  # Need some data for smoothing
            return self.get_eta_simple()
            
        # Use moving average of frame processing times
        avg_time_per_frame = sum(self.frame_times) / len(self.frame_times)
        remaining_frames = max(0, self.total_frames - self.processed_frames)
        return avg_time_per_frame * remaining_frames
    
    def get_eta_exponential(self):
        """
        ETA using exponential moving average for better responsiveness.
        
        Returns:
            float: ETA in seconds, or None if not enough data
        """
        if len(self.speed_history) < 3:
            return self.get_eta_smoothed()
        
        # Calculate exponential moving average of speed
        alpha = 0.3  # Smoothing factor
        ema_speed = None
        
        for speed in self.speed_history:
            if ema_speed is None:
                ema_speed = speed
            else:
                ema_speed = alpha * speed + (1 - alpha) * ema_speed
        
        if ema_speed and ema_speed > 0:
            remaining_frames = max(0, self.total_frames - self.processed_frames)
            return remaining_frames / ema_speed
        
        return self.get_eta_smoothed()
    
    def get_current_speed(self):
        """
        Get current processing speed in frames per second.
        
        Returns:
            float: Frames per second, or 0 if not enough data
        """
        if len(self.speed_history) == 0:
            return 0
        return sum(self.speed_history) / len(self.speed_history)
    
    def get_elapsed_time(self):
        """
        Get elapsed processing time.
        
        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def is_complete(self):
        """
        Check if processing is complete.
        
        Returns:
            bool: True if all frames have been processed
        """
        return self.processed_frames >= self.total_frames
    
    def format_time(self, seconds):
        """
        Format time in human-readable format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds is None:
            return "Unknown"
            
        if seconds < 0:
            return "0s"
            
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    
    def get_status_string(self):
        """
        Get a formatted status string with progress information.
        
        Returns:
            str: Status string
        """
        percentage = self.get_progress_percentage()
        eta = self.get_eta_exponential()
        speed = self.get_current_speed()
        elapsed = self.get_elapsed_time()
        
        return (f"{self.description}: {percentage:.1f}% "
                f"({self.processed_frames}/{self.total_frames}) "
                f"[Elapsed: {self.format_time(elapsed)} | "
                f"ETA: {self.format_time(eta)} | "
                f"Speed: {speed:.1f} fps]")
    
    def print_status(self):
        """
        Print current status to stdout.
        """
        status = self.get_status_string()
        sys.stdout.write(f"\r{status}")
        sys.stdout.flush()
    
    def finish(self):
        """
        Mark processing as complete and print final status.
        """
        self.processed_frames = self.total_frames
        elapsed = self.get_elapsed_time()
        sys.stdout.write(f"\r{self.description}: 100.0% "
                        f"({self.total_frames}/{self.total_frames}) "
                        f"[Completed in {self.format_time(elapsed)}]\n")
        sys.stdout.flush()

# Context manager for automatic cleanup
class ProgressContext:
    def __init__(self, total_frames, description="Processing"):
        self.progress = VideoProcessingProgress(total_frames, description)
        
    def __enter__(self):
        return self.progress
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:  # No exception occurred
            self.progress.finish()

# Example usage
if __name__ == "__main__":
    # Example of how to use the progress monitor
    import random
    
    # Simulate video processing
    total_frames = 1000
    with ProgressContext(total_frames, "Converting video") as progress:
        for i in range(total_frames):
            # Simulate frame processing time (varies between 0.01 and 0.05 seconds)
            processing_time = random.uniform(0.01, 0.05)
            time.sleep(processing_time)
            
            # Update progress
            progress.update(1)
            
            # Print status every 10 frames
            if i % 10 == 0:
                progress.print_status()