"""
Configuration management for the 2D3D pipeline.
"""

import json
import os

class Config:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.defaults = {
            # Depth estimation settings
            "depth_profile": "balanced",
            "depth_model": "MiDaS",
            
            # Stereo generation settings
            "baseline": 0.05,
            "focal_length": 1000,
            "shift_direction": "horizontal",
            
            # Quality preservation settings
            "preserve_colors": True,
            "temporal_smoothing": True,
            "temporal_smoothing_factor": 0.1,
            "post_process": False,
            
            # Performance settings
            "max_dimension": None,
            "batch_size": 4,
            "gpu_acceleration": True,
            
            # Output settings
            "default_format": "sbs",
            "default_resolution": [960, 1080],
            "crf": 18,
            "preset": "medium",
            
            # Optimization settings
            "optimization": {
                "parallel_processing": True,
                "batch_processing": True,
                "memory_optimization": True,
                "async_loading": True,
                "max_workers": 8,
                "max_memory_gb": 2.0
            }
        }
        
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                merged_config = self.defaults.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, updates):
        """Update multiple configuration values."""
        self.config.update(updates)

# Global configuration instance
config = Config()

def get_config():
    """Get the global configuration instance."""
    return config