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

import os
import ft_lib
print(ft_lib.__file__)
out_dir = "plot_out"

# Set x-axis limit to half the actual video FPS to use all fft frequencies
GRAPH_XLIM = config.actual_fps / 2 

# Delete all files in plot_out directory
if os.path.exists(out_dir):
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(out_dir)
    
    
print(f"Saving plots to {out_dir}")
for bubble in result_dict:
    print(f"Processing bubble {bubble}...")
    plt.figure(figsize=(10, 5))
    
    times, areas = result_dict[bubble]
    # Compute FFT of area signal
    area_arr = np.array(areas)
    n = len(area_arr)
    
    dt = np.mean(np.diff(times))  # average time step
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(area_arr - np.mean(area_arr))

    time_interval = (times[0], times[-1])
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    plt.title(f"Bubble {bubble}")
    
    axs[0].bar(freqs, np.abs(fft_vals), width=freqs[1]-freqs[0], alpha=0.7)
    axs[0].scatter(freqs, np.abs(fft_vals), s=5)
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(1, GRAPH_XLIM)
    axs[0].set_xscale("log")
    axs[0].set_title(f"Bubble {bubble}: Frequency Spectrum of Area")
    
    
    
    freq, ampl = ft_lib.get_freq_from_wav_file(config.output_audio_path, times[0], times[-1])
    
    axs[1].bar(freq, ampl, width=freq[1]-freq[0] if len(freq)>1 else 1, alpha=0.7)
    axs[1].scatter(freq, ampl, s=5)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title(f"Bubble {bubble}: Frequency Spectrum of Sound")
    axs[1].set_xlim(1, GRAPH_XLIM)
    axs[1].set_xscale("log")
    
    
    plt.savefig(os.path.join(out_dir, f"bubble_{bubble}_fft.png"))
    plt.close()
    
    plt.figure()
    plt.scatter(times, areas)
    plt.xlabel("Time (s)")
    plt.ylabel("Area")
    plt.title(f"Bubble {bubble}: Area over Time")
    plt.savefig(os.path.join(out_dir, f"bubble_{bubble}_area.png"))
    
    

    