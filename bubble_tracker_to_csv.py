import pandas as pd
import os
from BubbleTracker import BubbleTracker
from config_loader import config

def export_bubble_tracker_to_csv(tracker, output_path, experiment_name=""):
    """
    Export bubble tracker data to CSV with optimal structure for analysis.
    
    Args:
        tracker: BubbleTracker object
        output_path: Path for the output CSV file
        experiment_name: Optional experiment identifier for combining datasets
    """
    
    # Collect all data into a list of dictionaries
    data_rows = []
    
    for bubble_id, track in tracker.get_tracks().items():
        for i, (frame, cx, cy, area) in enumerate(track):
            
            # Calculate derived metrics
            time_seconds = frame / config.actual_fps
            
            # Calculate velocity (if not first point)
            velocity_x = velocity_y = velocity_magnitude = 0
            if i > 0:
                prev_frame, prev_cx, prev_cy, _ = track[i-1]
                dt = (frame - prev_frame) / config.actual_fps
                if dt > 0:
                    velocity_x = (cx - prev_cx) / dt
                    velocity_y = (cy - prev_cy) / dt
                    velocity_magnitude = (velocity_x**2 + velocity_y**2)**0.5
            
            # Calculate bubble metrics
            bubble_lifetime = len(track)
            bubble_first_frame = track[0][0]
            bubble_last_frame = track[-1][0]
            bubble_duration = (bubble_last_frame - bubble_first_frame) / config.actual_fps
            
            row = {
                'experiment': experiment_name,
                'bubble_id': bubble_id,
                'frame': frame,
                'time_seconds': time_seconds,
                'center_x': cx,
                'center_y': cy,
                'area_pixels': area,
                'velocity_x_px_per_sec': velocity_x,
                'velocity_y_px_per_sec': velocity_y,
                'velocity_magnitude_px_per_sec': velocity_magnitude,
                'bubble_lifetime_frames': bubble_lifetime,
                'bubble_duration_seconds': bubble_duration,
                'bubble_first_frame': bubble_first_frame,
                'bubble_last_frame': bubble_last_frame,
                'detection_order': i,  # 0 for first detection, 1 for second, etc.
            }
            
            data_rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    
    # Sort by bubble_id and frame for consistent ordering
    df = df.sort_values(['bubble_id', 'frame'])
    
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} detection records for {len(tracker.get_tracks())} bubbles to {output_path}")
    
    return df

def export_bubble_summary(tracker, output_path, experiment_name=""):
    """
    Export summary statistics for each bubble (one row per bubble).
    """
    
    summary_rows = []
    
    for bubble_id, track in tracker.get_tracks().items():
        if not track:
            continue
            
        # Extract data arrays
        frames = [frame for frame, _, _, _ in track]
        areas = [area for _, _, _, area in track]
        x_coords = [cx for _, cx, _, _ in track]
        y_coords = [cy for _, _, cy, _ in track]
        
        # Calculate summary metrics
        first_frame, last_frame = min(frames), max(frames)
        duration_seconds = (last_frame - first_frame) / config.actual_fps
        
        # Area statistics
        avg_area = sum(areas) / len(areas)
        max_area = max(areas)
        min_area = min(areas)
        
        # Movement statistics
        total_distance = 0
        for i in range(1, len(track)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_distance += (dx**2 + dy**2)**0.5
        
        avg_velocity = total_distance / duration_seconds if duration_seconds > 0 else 0
        
        # Position statistics
        avg_x = sum(x_coords) / len(x_coords)
        avg_y = sum(y_coords) / len(y_coords)
        
        summary_row = {
            'experiment': experiment_name,
            'bubble_id': bubble_id,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'lifetime_frames': len(track),
            'duration_seconds': duration_seconds,
            'first_time_seconds': first_frame / config.actual_fps,
            'last_time_seconds': last_frame / config.actual_fps,
            'avg_area_pixels': avg_area,
            'max_area_pixels': max_area,
            'min_area_pixels': min_area,
            'avg_center_x': avg_x,
            'avg_center_y': avg_y,
            'total_distance_pixels': total_distance,
            'avg_velocity_px_per_sec': avg_velocity,
            'detection_count': len(track)
        }
        
        summary_rows.append(summary_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_rows)
    df = df.sort_values('bubble_id')
    
    df.to_csv(output_path, index=False)
    print(f"Exported summary for {len(df)} bubbles to {output_path}")
    
    return df

def main():
    """Main function to export bubble tracker data."""
    
    # Load the tracker
    tracker = BubbleTracker.load(config.tracker_output_path)
    
    if not tracker:
        print("No tracker file found!")
        return
    
    # Create output directory
    output_dir = "exported_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get experiment name from input video filename
    experiment_name = os.path.splitext(os.path.basename(config.input_video_path))[0]
    
    # Export detailed data (one row per detection)
    detailed_output = os.path.join(output_dir, f"{experiment_name}_detailed_tracking.csv")
    detailed_df = export_bubble_tracker_to_csv(tracker, detailed_output, experiment_name)
    
    # Export summary data (one row per bubble)
    summary_output = os.path.join(output_dir, f"{experiment_name}_bubble_summary.csv")
    summary_df = export_bubble_summary(tracker, summary_output, experiment_name)
    
    print(f"\nExport complete!")
    print(f"Detailed data: {detailed_output}")
    print(f"Summary data: {summary_output}")
    
    # Show basic statistics
    print(f"\nDataset statistics:")
    print(f"  Total bubbles: {len(tracker.get_tracks())}")
    print(f"  Total detections: {len(detailed_df)}")
    print(f"  Average bubble lifetime: {detailed_df.groupby('bubble_id')['detection_order'].count().mean():.1f} frames")

if __name__ == "__main__":
    main()