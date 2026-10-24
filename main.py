import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    t, signal = generate_signal(bpm=72)
    save_signal(t, signal, "sample_data/fake_heartbeat.csv")

    t2, signal2 = load_signal("sample_data/fake_heartbeat.csv")
    print(np.allclose(t, t2), np.allclose(signal, signal2))

    plt.plot(t, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Fake heartbeat at 72 BPM")
    plt.savefig("fake_heartbeat_out.png")
    plt.show()
