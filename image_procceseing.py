import cv2
import numpy as np
import math
from collections import defaultdict
import csv
import matplotlib.pyplot as plt
import ft_lib
from moviepy.editor import AudioFileClip
from moviepy.editor import vfx
import sys
from moviepy.editor import VideoFileClip
import os
import shutil
import pickle
from BubbleTracker import BubbleTracker
from config_loader import config

def clear_temp_directory() -> None:
    temp_dir = 'temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

def load_video(path: str) -> cv2.VideoCapture:
    """Loads a video file.

    :param path: Path to the video file.
    :type path: str
    :return: OpenCV video capture object.
    :rtype: cv2.VideoCapture
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error opening video file: {path}")
        exit()
    return cap

def calc_crop_dimensions(cap: cv2.VideoCapture, crop_x: tuple, crop_y: tuple) -> tuple:
    """
    Calculates the width and height of a cropped region from a video frame.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        crop_x (tuple): Tuple of (start_x, end_x) coordinates for cropping along the x-axis.
        crop_y (tuple): Tuple of (start_y, end_y) coordinates for cropping along the y-axis.
    Returns:
        tuple: (width, height, crop_x, crop_y) of the cropped region, ensuring the crop coordinates are within the video frame bounds.
        
    """
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_x = (max(0, crop_x[0]), min(video_width, crop_x[1]))
    crop_y = (max(0, crop_y[0]), min(video_height, crop_y[1]))
    width = crop_x[1] - crop_x[0]
    height = crop_y[1] - crop_y[0]
    return width, height, crop_x, crop_y

def init_output_video(test_out_path:str, width: int, height: int, cap: cv2.VideoCapture) -> tuple:
    """Initializes the output video writers.

    :param test_out_path: Path to the test output video file.
    :type test_out_path: str
    :param width: Width of the output video frames.
    :type width: int
    :param height: Height of the output video frames.
    :type height: int
    :param cap: OpenCV video capture object.
    :type cap: cv2.VideoCapture
    :return: Video writer objects for the output and test output videos.
    :rtype: tuple
    :return: Frames per second of the input video.
    :rtype: float
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    test_out = cv2.VideoWriter(test_out_path, fourcc, fps, (width * config.num_videos_in_test_output, height))

    return test_out, fps

def detatch_audio(path: str, output_path: str, speed: float) -> None:
    video = VideoFileClip(path)

    # speed up the video by the specified factor
    video = video.fx(vfx.speedx, factor=speed)

    # write audio to a temporary file
    video.audio.write_audiofile(output_path)

def apply_threshold_to_sub_picture(frame, threshold=15):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calcuate avarage brightness of the frame
    avg_brightness = np.mean(frame)
    
    threshold += int(avg_brightness)  # Adjust threshold based on average brightness
    _, thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def split_frame(frame, num_parts=4):
    # Split the fram into num_parts parts along the y-axis
    height, width = frame.shape[:2]
    part_height = height // num_parts
    parts = []
    for i in range(num_parts):
        start_y = i * part_height
        end_y = (i + 1) * part_height if i < num_parts - 1 else height
        parts.append(frame[start_y:end_y, :])
    return parts

def combine_parts(parts):
    # Combine the parts back into a single frame
    return np.vstack(parts)

# Create an average background frame from the first num_frames frames
def create_average_background(cap, num_frames, crop_x, crop_y) -> np.ndarray:
    
    print("Calculating average background...")
    start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Save current position
    
    # Initialize accumulator with first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    
    if not ret:
        raise ValueError("Cannot read frames from video")
    
    cropped_frame = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    accumulator = cropped_frame.astype(np.float64)  # Use float64 to avoid overflow
    frames_processed = 1
    
    # Accumulate remaining frames
    for _ in range(num_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
        accumulator += cropped_frame.astype(np.float64)
        frames_processed += 1
    
    # Calculate average and convert back to uint8
    avg_frame = (accumulator / frames_processed).astype(np.uint8)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Restore position
    
    # delete last print line
    sys.stdout.write("\033[F")  # Cursor up one line
    sys.stdout.write("\033[K")  # Clear to the end of line
    return avg_frame

def detect_bubbles(contours, cropped_frame, min_area=1300):
    """
    Processes contours to detect bubbles and draws visualizations.
    
    Args:
        contours: List of contours from cv2.findContours
        cropped_frame: Frame to draw on (will be modified)
        min_area: Minimum area threshold for bubble detection
        
    Returns:
        tuple: (detected_bubbles, bubble_in_frame)
            - detected_bubbles: List of (cx, cy, area) tuples
            - bubble_in_frame: Boolean indicating if any bubbles were found
    """
    detected_bubbles = []
    bubble_in_frame = False
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        bubble_in_frame = True
        
        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the polygon
        cv2.polylines(cropped_frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

        # Compute center of the polygon to place text
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = approx[0][0]  # fallback to first vertex

        # Collect detection
        detected_bubbles.append((cx, cy, area))
        
        # Draw area text
        cv2.putText(cropped_frame, f"{int(area)}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return detected_bubbles, bubble_in_frame

def frame_processor(cropped_frame: np.ndarray, avg_background: np.ndarray) -> tuple:
    """Processes each frame of the video and applies image processing techniques.
    """
    # create mask by thresholding the difference between the current frame and the average background
    gray_bg = cv2.cvtColor(avg_background, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.subtract(gray_bg, gray_frame)

    # brighten the gray diff
    gray_diff = cv2.convertScaleAbs(gray_diff, alpha=config.image_contrast, beta=config.image_brightness)
    
    # split, threshold, and combine to create mask
    split_frames = split_frame(gray_diff, num_parts=4)
    thresholded_parts = [apply_threshold_to_sub_picture(part, config.masking_threshold) for part in split_frames]
    mask = combine_parts(thresholded_parts)
    
    
    # Clean up the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.kernel_size)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations)

    # invert the mask
    mask_clean = cv2.bitwise_not(mask_clean)

    blurred_frame = cv2.GaussianBlur(mask_clean, (5,5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_frame, 10, 15)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, gray_diff, mask, mask_clean

def crop_frame(frame, crop_x, crop_y):
    """Crops the frame to the specified coordinates.
    """
    return frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

def draw_bubble_outlines(cropped_frame, tracker, frame_num):

    for bubble_id, history in tracker.get_tracks().items():
        if history and history[-1][0] == frame_num:
            _, cx, cy, area = history[-1]
            cv2.putText(cropped_frame, f"ID:{bubble_id}", (cx + 25, cy + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
def assemble_output_frame(gray_diff, mask, mask_clean, cropped_frame, frame_num, fps):
    # Convert both to 3-channel for concatenation
    gray_diff_3ch = cv2.cvtColor(gray_diff, cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_clean_3ch = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

    # Concatenate horizontally (side by side)
    side_by_side = np.hstack((gray_diff_3ch, mask_3ch, mask_clean_3ch, cropped_frame))
    
    # add a timestamp and frame number to the combined frame
    timestamp = f"Time: {frame_num/fps:.2f}s Frame: {frame_num}"
    cv2.putText(side_by_side, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    return side_by_side

def write_bubble_data_to_csv(tracker, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["BubbleID", "Frame", "X", "Y", "Area"])
        for bubble_id, track in tracker.get_tracks().items():
            for (frame, x, y, area) in track:
                writer.writerow([bubble_id, frame, x, y, area])



def process_frames(cap: cv2.VideoCapture, tracker: BubbleTracker, test_out: cv2.VideoWriter, 
                  fps: float, crop_x: tuple, crop_y: tuple, background_interval: int) -> None:
    """
    Main frame processing loop that handles bubble detection and tracking.
    
    Args:
        cap: OpenCV video capture object
        tracker: BubbleTracker instance for tracking bubbles across frames
        test_out: Video writer for output video
        fps: Frames per second of the video
        crop_x: X coordinates for cropping (start_x, end_x)
        crop_y: Y coordinates for cropping (start_y, end_y)
        background_interval: Number of frames between background recalculations
    """
    avg_background = create_average_background(cap, num_frames=background_interval, crop_x=crop_x, crop_y=crop_y)
    frame_num = 0
    
    while True:
        bubble_in_frame = False
        print(f"time: {frame_num/fps:.2f}s / {cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps:.2f}s")
        # clear the console output
        sys.stdout.write("\033[F")  # Cursor up one line
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        # recalculate the average background
        if frame_num % background_interval == 0:
            avg_background = create_average_background(cap, num_frames=background_interval, crop_x=crop_x, crop_y=crop_y)
            
        # increment frame number
        frame_num += 1
        
        # apply image processing
        cropped_frame = crop_frame(frame, crop_x, crop_y)
        contours, gray_diff, mask, mask_clean = frame_processor(cropped_frame, avg_background)
        
        # detect bubbles and update tracker
        detected_bubbles, bubble_in_frame = detect_bubbles(contours, cropped_frame, min_area=config.min_bubble_area)
        tracker.update(detected_bubbles, frame_num)
        
        # draw bubble outlines and IDs
        draw_bubble_outlines(cropped_frame, tracker, frame_num)
        
        # write output frame 
        if bubble_in_frame:
            side_by_side = assemble_output_frame(gray_diff, mask, mask_clean, cropped_frame, frame_num, fps)
            test_out.write(side_by_side)

def process_video():
    """Processes the input video and generates output videos and data.
    """
    
    clear_temp_directory()
    cap = load_video(config.input_video_path)
    width, height, crop_x, crop_y = calc_crop_dimensions(cap, config.crop_x, config.crop_y)
    test_out, fps = init_output_video(config.test_output_video_path, width, height, cap)

    detatch_audio(config.input_video_path, config.output_audio_path, config.speed_factor)
    
    # Initialize bubble tracker
    tracker = BubbleTracker(max_distance=config.max_tracking_distance)

    background_interval = config.background_recalc_interval  # Number of frames between background recalculations

    # Process all frames
    process_frames(cap, tracker, test_out, fps, crop_x, crop_y, background_interval)

    cap.release()
    test_out.release()
    cv2.destroyAllWindows()
    write_bubble_data_to_csv(tracker, config.csv_output_path)
    tracker.save(config.tracker_output_path)
    print("Processing complete.")

    
if __name__ == "__main__":
    process_video()