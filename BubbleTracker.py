

import math
import pickle
from collections import defaultdict

"""Class responsible for tracking bubbles across frames."""
class BubbleTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.bubbles = {}  # id -> (cx, cy)
        self.history = defaultdict(list)  # id -> list of (frame, cx, cy, area)
        self.max_distance = max_distance

    def update(self, detections, frame_num):
        updated_ids = set()
        new_bubbles = {}

        for (cx, cy, area) in detections:
            matched_id = None
            min_dist = self.max_distance

            for bubble_id, (prev_cx, prev_cy) in self.bubbles.items():
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if dist < min_dist:
                    matched_id = bubble_id
                    min_dist = dist

            if matched_id is not None:
                # Existing bubble
                new_bubbles[matched_id] = (cx, cy)
                self.history[matched_id].append((frame_num, cx, cy, area))
                updated_ids.add(matched_id)
            else:
                # New bubble
                bubble_id = self.next_id
                self.next_id += 1
                new_bubbles[bubble_id] = (cx, cy)
                self.history[bubble_id].append((frame_num, cx, cy, area))
                updated_ids.add(bubble_id)

        self.bubbles = {k: v for k, v in new_bubbles.items() if k in updated_ids}

    def get_tracks(self):
        return self.history
    
    def save(self, filepath: str) -> None:
        """
        Save this BubbleTracker instance to a file using pickle.
        
        Args:
            filepath: Path where to save the tracker object
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"BubbleTracker saved to {filepath}")
        except Exception as e:
            print(f"Error saving BubbleTracker: {e}")
            raise
    
    @staticmethod
    def load(filepath: str) -> 'BubbleTracker':
        """
        Load BubbleTracker instance from a file using pickle.
        
        Args:
            filepath: Path to the saved tracker object
            
        Returns:
            BubbleTracker: Loaded tracker instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error loading the object
        """
        try:
            with open(filepath, 'rb') as f:
                tracker = pickle.load(f)
            print(f"BubbleTracker loaded from {filepath}")
            return tracker
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            raise
        except Exception as e:
            print(f"Error loading BubbleTracker: {e}")
            raise
