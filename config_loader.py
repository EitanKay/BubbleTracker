"""
Configuration loader for bubble tracking application.
Loads settings from config.json and provides easy access to configuration values.
"""

import json
import os
from typing import Dict, Any, Tuple, List

class Config:
    """Configuration manager for bubble tracking application."""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration from JSON file.
        
        Args:
            config_file: Path to the configuration JSON file
        """
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found")
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
    
    # File paths
    @property
    def input_video_path(self) -> str:
        return self._config["file_paths"]["input_video"]
    
    @property
    def test_output_video_path(self) -> str:
        return self._config["file_paths"]["test_output_video"]
    
    @property
    def output_audio_path(self) -> str:
        return self._config["file_paths"]["output_audio"]
    
    @property
    def csv_output_path(self) -> str:
        return self._config["file_paths"]["csv_output"]
    
    @property
    def tracker_output_path(self) -> str:
        return self._config["file_paths"]["tracker_output"]
    
    # Video processing
    @property
    def actual_fps(self) -> int:
        return self._config["video_processing"]["actual_fps"]
    
    @property
    def speed_factor(self) -> int:
        return self._config["video_processing"]["speed_factor"]
    
    @property
    def num_videos_in_test_output(self) -> int:
        return self._config["video_processing"]["num_videos_in_test_output"]
    
    # Frame cropping
    @property
    def crop_y(self) -> Tuple[int, int]:
        return tuple(self._config["frame_cropping"]["crop_y"])
    
    @property
    def crop_x(self) -> Tuple[int, int]:
        return tuple(self._config["frame_cropping"]["crop_x"])
    
    # Image processing
    @property
    def image_contrast(self) -> float:
        return self._config["image_processing"]["contrast"]
    
    @property
    def image_brightness(self) -> int:
        return self._config["image_processing"]["brightness"]
    
    @property
    def masking_threshold(self) -> int:
        return self._config["image_processing"]["masking_threshold"]
    
    # Morphological operations
    @property
    def kernel_size(self) -> Tuple[int, int]:
        return tuple(self._config["morphological_operations"]["kernel_size"])
    
    @property
    def morph_open_iterations(self) -> int:
        return self._config["morphological_operations"]["morph_open_iterations"]
    
    @property
    def morph_close_iterations(self) -> int:
        return self._config["morphological_operations"]["morph_close_iterations"]
    
    # Bubble detection
    @property
    def min_bubble_area(self) -> int:
        return self._config["bubble_detection"]["min_bubble_area"]
    
    @property
    def max_tracking_distance(self) -> int:
        return self._config["bubble_detection"]["max_tracking_distance"]
    
    @property
    def background_recalc_interval(self) -> int:
        return self._config["bubble_detection"]["background_recalc_interval"]

# Global configuration instance
config = Config()