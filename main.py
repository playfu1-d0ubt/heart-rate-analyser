import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


def find_beats(signal, sample_rate, min_gap=0.3):
    """Return sample numbers where heartbeats happen."""
    min_samples = int(min_gap * sample_rate)
    min_height = 0.5 * np.max(signal)  # ignore bumps under half of tallest
    peaks, _ = find_peaks(signal, distance=min_samples, prominence=min_height)
    return peaks


def calculate_bpm(peaks, sample_rate):
    """Return (average BPM, gaps between beats in seconds)."""
    if len(peaks) < 2:
        sys.exit("Error: need at least 2 beats to work out a heart rate.")
    gaps = np.diff(peaks) / sample_rate
    bpm = 60 / np.mean(gaps)
    return bpm, gaps


def filter_signal(signal, sample_rate, low=0.5, high=15, order=3):
    """Band-pass filter. Keep speeds between low and high (Hz)."""
    if high >= sample_rate / 2:
        sys.exit("Error: high cutoff must be below half the sample rate.")
    sos = butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal)


def save_signal(t, signal, path):
    """Save time and signal to a CSV file."""
    df = pd.DataFrame({"time": t, "signal": signal})
    df.to_csv(path, index=False)


def load_signal(path):
    """Load a CSV. Return (time, signal) as numpy arrays."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Error: can't find '{path}'.")

    if "time" not in df.columns or "signal" not in df.columns:
        sys.exit("Error: CSV needs 'time' and 'signal' columns.")

    return df["time"].to_numpy(), df["signal"].to_numpy()


def generate_signal(
    bpm=72, duration=10, sample_rate=250, drift=True, hum=True, noise=True
):
    """Make a fake heartbeat signal. Returns (time, signal) arrays."""
    # 1. Time stamps: one every 1/sample_rate seconds
    t = np.arange(0, duration, 1 / sample_rate)

    # 2. Blank signal, same length as t
    signal = np.zeros(len(t))

    # 3. Add one bell-shaped bump per heartbeat
    beat_gap = 60 / bpm  # seconds between beats
    beat_times = np.arange(0.5, duration, beat_gap)
    width = 0.02  # smaller = sharper spike
    for beat_time in beat_times:
        signal += np.exp(-((t - beat_time) ** 2) / (2 * width**2))

    # 4. Add the mess, each switchable
    if drift:
        signal += 0.5 * np.sin(2 * np.pi * 0.2 * t)
    if hum:
        signal += 0.2 * np.sin(2 * np.pi * 50 * t)
    if noise:
        signal += np.random.normal(0, 0.05, size=len(t))

    return t, signal


if __name__ == "__main__":
    np.random.seed(42)  # same "random" noise every run, so results repeat
    t, signal = generate_signal(bpm=120)
    save_signal(t, signal, "sample_data/fake_heartbeat.csv")
    clean = filter_signal(signal, 250)
    t2, signal2 = load_signal("sample_data/fake_heartbeat.csv")
    print(np.allclose(t, t2), np.allclose(signal, signal2))
    peaks = find_beats(clean, 250)
    print(f"Found {len(peaks)} beats")
    bpm, gaps = calculate_bpm(peaks, 250)
    print(f"Average heart rate: {bpm:.1f} BPM")
    print(f"Shortest gap: {gaps.min():.2f} s, longest: {gaps.max():.2f} s")

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
    top.plot(t, signal)
    top.set_title("Raw")
    bottom.plot(t, clean)
    bottom.plot(t[peaks], clean[peaks], "ro")
    bottom.set_title("Filtered with a detected beats")
    bottom.set_xlabel("Time (s)")
    plt.savefig("export/filtered_out.png")
    plt.show()
