import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import csv

def read_and_trim_wav(file_path, trim_seconds=0):
    samplerate, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]
    start_sample = int(trim_seconds * samplerate)
    if start_sample < len(data):
        data = data[start_sample:]
    else:
        print(f"File {file_path} is shorter than {trim_seconds} seconds. Skipping.")
        return None, None
    return samplerate, data

def compute_fft(data, samplerate):
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, 1.0 / samplerate)
    return xf[:N//2], np.abs(yf[:N//2])



def plot_spectrum(filename, start_time, end_time, bubble_name):
    
    samplerate, data = read_and_trim_wav(filename)

    active_data = data[int(start_time * samplerate):int(end_time * samplerate)]
    
    xf, yf = compute_fft(active_data, samplerate)
    # plt.figure(figsize=(12, 6))
    plt.plot(xf, yf)
    plt.title(f"Spectrum from {start_time} to {end_time} seconds for {bubble_name}")
    plt.xscale('log')
    # plt.ylim(0, 6e8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    ticks = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000]
    plt.xticks(ticks, [str(int(t)) for t in ticks])
    plt.xlim(1, 1000)
    plt.show()




plot_spectrum("BubbleTracker/input_files/5mm.wav", 2, 10, "Bubble 1")

