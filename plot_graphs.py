import librosa
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import ft_lib
import os
from BubbleTracker import BubbleTracker
from config_loader import config

fps = config.actual_fps

# load saved tracker
tracker = BubbleTracker.load(config.tracker_output_path)
if not tracker:
    raise ValueError(f"Failed to load BubbleTracker from {config.tracker_output_path}")
# Create a dictionary to store (time, area) for each bubble ID
result_dict = {}

for bubble_id, track in tracker.get_tracks().items():
    if len(track) > 30:
        times = []
        areas = []
        for (frame, x, y, area) in track:
            times.append(frame / fps)  # Convert frame number to seconds
            areas.append(area)
        result_dict[bubble_id] = (times, areas)
        
# remove buubles that share time with other bubbles
# Find all times where more than one bubble is present

# Map each time to the set of bubbles present at that time
time_to_bubbles = defaultdict(set)
for bubble_id, (times, _) in result_dict.items():
    for t in times:
        time_to_bubbles[round(t, 6)].add(bubble_id)  # round to avoid floating point issues

# Find all times with more than one bubble
overlap_times = {t for t, bubbles in time_to_bubbles.items() if len(bubbles) > 1}

# Remove bubbles that share any time with another bubble
bubbles_to_remove = set()
for bubble_id, (times, _) in result_dict.items():
    if any(round(t, 6) in overlap_times for t in times):
        bubbles_to_remove.add(bubble_id)

for bubble_id in bubbles_to_remove:
    del result_dict[bubble_id]


# remvoe bubbles that moved less than 500 px in the y direction
bubbles_to_remove = set()
for bubble_id, (times, areas) in result_dict.items():
    track = tracker.get_tracks()[bubble_id]
    y_positions = [y for (frame, x, y, area) in track]
    if max(y_positions) - min(y_positions) < 500:
        bubbles_to_remove.add(bubble_id)
for bubble_id in bubbles_to_remove:
    del result_dict[bubble_id]

# Set x-axis limit to half the actual video FPS to use all fft frequencies
GRAPH_XLIM = config.actual_fps / 2 

out_dir = "plot_out"
# Delete all files in plot_out directory
if os.path.exists(out_dir):
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(out_dir)
    
    
print(f"Saving plots to {out_dir}")

# dict[BubbleID] = (avg_size, dominant_area_freq, dominant_sound_freq)
size_freq_dict : dict[int, tuple[float, float, float]] = {}


max_iterations = 100000
iteration = 0
max_bubble_to_plot = 10000
for bubble in result_dict:
    iteration += 1
    if iteration > max_iterations:
        break
    print(f"Processing bubble {bubble}...")
    
    
    times, areas = result_dict[bubble]
    # Compute FFT of area signal
    # remove 3 points from start and end to avoid edge effects
    # areas = areas[3:-3]
    # times = times[3:-3]
    area_arr = np.array(areas)
    n = len(area_arr)
    

    dt = np.mean(np.diff(times))  # average time step
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(area_arr - np.mean(area_arr))

    sound_freq, sound_ampl = ft_lib.get_freq_from_wav_file(config.output_audio_path, times[0], times[-1])
    
    time_interval = (times[0], times[-1])
    if (bubble < max_bubble_to_plot):    
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
        plt.title(f"Bubble {bubble}")
        axs[0].vlines(freqs, 0, np.abs(fft_vals), alpha=0.7, linewidth=0.8)
        axs[0].scatter(freqs, np.abs(fft_vals), s=5)
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_xlim(1, GRAPH_XLIM)
        axs[0].set_xscale("log")
        axs[0].set_title(f"Bubble {bubble}: Frequency Spectrum of Area")    
        
        axs[1].vlines(sound_freq, 0, sound_ampl, alpha=0.7, linewidth=0.8)
        axs[1].scatter(sound_freq, sound_ampl, s=5)
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_title(f"Bubble {bubble}: Frequency Spectrum of Sound")
        axs[1].set_xlim(1, GRAPH_XLIM)
        axs[1].set_xscale("log")
        
        
        plt.savefig(os.path.join(out_dir, f"bubble_{bubble}_fft.png"))
        plt.close()
    
    # Load audio data for this bubble's timeframe
    audio_data, sample_rate = librosa.load(config.output_audio_path, sr=None)
    audio_duration = len(audio_data) / sample_rate
    audio_times = np.linspace(0, audio_duration, len(audio_data))
    
    
    if (bubble < max_bubble_to_plot):    
    
        # Create combined plot with area on top and audio waveform below
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top plot: Area over time
        ax1.scatter(times, areas)
        ax1.set_ylabel("Area")
        ax1.set_title(f"Bubble {bubble}: Area over Time")
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Audio waveform for bubble timeframe
        start_time, end_time = times[0], times[-1]
        audio_mask = (audio_times >= start_time) & (audio_times <= end_time)
        ax2.plot(audio_times[audio_mask], audio_data[audio_mask], linewidth=0.8)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Audio Amplitude')
        ax2.set_title(f'Audio Waveform ({start_time:.2f}s - {end_time:.2f}s)')
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bubble_{bubble}_area.png"))
        plt.close()
    
        # Find dominant sound frequency and average size

    freq_upper_limit = 13
    freq_lower_limit = 7
    
    # Filter frequencies above 6Hz (correct way)
    freq_mask = (freqs >= freq_lower_limit) & (freqs <= freq_upper_limit)  # Boolean mask for frequencies above 6 Hz
    if np.any(freq_mask):
        # Find the frequency with maximum amplitude in the filtered range
        filtered_fft_vals = np.abs(fft_vals[freq_mask])
        filtered_freqs = freqs[freq_mask]
        dominant_area_freq = filtered_freqs[np.argmax(filtered_fft_vals)]
        
    else:
        # Fallback if no frequencies above 6 Hz
        dominant_area_freq = freqs[np.argmax(np.abs(fft_vals))]

    # Apply similar filter to sound frequency (8-12 Hz range)
    sound_freq_mask = (sound_freq >= freq_lower_limit) & (sound_freq <= freq_upper_limit)
    
    if np.any(sound_freq_mask):
        # Find the frequency with maximum amplitude in the filtered range
        filtered_sound_ampl = sound_ampl[sound_freq_mask]
        filtered_sound_freq = sound_freq[sound_freq_mask]
        dominant_sound_freq = filtered_sound_freq[np.argmax(filtered_sound_ampl)]
    else:
        # Fallback if no frequencies in range
        dominant_sound_freq = sound_freq[np.argmax(sound_ampl)]

    avg_size = np.mean(areas)
    
    size_freq_dict[bubble] = (avg_size, dominant_area_freq, dominant_sound_freq)
    plt.close('all')  # Close any remaining figures
    plt.clf()         # Clear the current figure
    plt.cla()         # Clear the current axes
    


plt.figure(figsize=(8, 6))
area_freq = [size_freq_dict[b][1] for b in size_freq_dict]
sound_freq = [size_freq_dict[b][2] for b in size_freq_dict]
bubble_ids = list(size_freq_dict.keys())

# Calculate frequency bin width for both x and y axes
x_errors = []
y_errors = []

for bubble_id in bubble_ids:
    times, areas = result_dict[bubble_id]
    dt = np.mean(np.diff(times))
    n = len(areas)
    
    # X-axis error (area frequency resolution)
    df_area = 1 / (n * dt)
    x_errors.append(df_area)
    
    # Y-axis error (sound frequency resolution - same time window)
    duration = times[-1] - times[0]  # Total time window
    df_sound = 1 / duration  # Sound frequency resolution
    y_errors.append(df_sound)


# Add bubble ID labels next to each point
for i, bubble_id in enumerate(bubble_ids):
    plt.annotate(str(bubble_id), 
                (area_freq[i], sound_freq[i]), 
                xytext=(5, 5),  # offset the text by 5 points
                textcoords='offset points',
                fontsize=9,
                alpha=0.8)
plt.scatter(area_freq, sound_freq, alpha=0.7)
plt.ylabel("Dominant Sound Frequency (Hz)")
plt.xlabel("Dominant Area Frequency (Hz)")
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.title("Bubble Area Frequency vs Dominant Sound Frequency\n(with frequency bin uncertainty on both axes)")
plt.grid(True, alpha=0.3)

# add a simple y=x line
plt.plot([6, 13], [6, 13], linestyle='--', color='gray', alpha=0.5, label='y=x')

plt.legend()
plt.savefig(os.path.join(out_dir, "area_vs_sound_frequency.png"), dpi=300, bbox_inches='tight')

# Plot with error bars on both axes
plt.errorbar(area_freq, sound_freq, 
             xerr=x_errors, yerr=y_errors, 
             fmt='o', capsize=3, capthick=1, alpha=0.7, 
             elinewidth=1, markersize=5)

plt.savefig(os.path.join(out_dir, "area_vs_sound_frequency_with_error_bars.png"), dpi=300, bbox_inches='tight')
plt.close()


# plot the sound frequency as a function of the square root of the avarage area
plt.figure(figsize=(8, 6))
plt.scatter(np.sqrt([size_freq_dict[b][0] for b in size_freq_dict]), sound_freq, alpha=0.7)
plt.xlabel("Square Root of Average Bubble Area")
plt.ylabel("Dominant Sound Frequency (Hz)")
plt.title("Sound Frequency vs Square Root of Average Bubble Area")
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(out_dir, "sound_frequency_vs_sqrt_area.png"), dpi=300, bbox_inches='tight')
