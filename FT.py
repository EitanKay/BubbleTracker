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



def plot_spectrum_from_wav_file(filename, start_time, end_time, bubble_name, plt):
    
    samplerate, data = read_and_trim_wav(filename)

    active_data = data[int(start_time * samplerate):int(end_time * samplerate)]
    
    xf, yf = compute_fft(active_data, samplerate)
    plot_spectrum(xf, yf, f"Spectrum from {start_time} to {end_time} seconds for {bubble_name} of the sound waves", plt)


def plot_spectrum(xf, yf, title, plt):
    # plt.figure(figsize=(12, 6))
    plt.plot(xf, yf)
    plt.set_title(title)
    plt.set_xscale('log')
    # plt.ylim(0, 6e8)
    plt.set_xlabel("Frequency (Hz)")
    plt.set_ylabel("Amplitude")
    ticks = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000]
    plt.set_xticks(ticks, [str(int(t)) for t in ticks])
    plt.set_xlim(1, 1000)
    # plt.show()

def get_freq_from_wav_file(filename, start_time, end_time):
    samplerate, data = read_and_trim_wav(filename)
    active_data = data[int(start_time * samplerate):int(end_time * samplerate)]
    xf, yf = compute_fft(active_data, samplerate)
    return xf, yf


# plot_spectrum_from_wav_file("./input_files/5mm.wav", 2, 10, "Bubble 1")

