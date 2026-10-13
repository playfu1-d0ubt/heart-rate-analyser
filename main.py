import numpy as np
import matplotlib.pyplot as plt


def generate_signal(bpm=72, duration=10, sample_rate=250,
                    drift=True, hum=True, noise=True):
    """Make a fake heartbeat signal. Returns (time, signal) arrays."""
    # 1. Time stamps: one every 1/sample_rate seconds
    t = np.arange(0, duration, 1 / sample_rate)

    # 2. Blank signal, same length as t
    signal = np.zeros(len(t))

    # 3. Add one bell-shaped bump per heartbeat
    beat_gap = 60 / bpm                          # seconds between beats
    beat_times = np.arange(0.5, duration, beat_gap)
    width = 0.02                                # smaller = sharper spike
    for beat_time in beat_times:
        signal += np.exp(-((t - beat_time) ** 2) / (2 * width ** 2))

    # 4. Add the mess, each switchable
    if drift:
        signal += 0.5 * np.sin(2 * np.pi * 0.2 * t)
    if hum:
        signal += 0.2 * np.sin(2 * np.pi * 50 * t)
    if noise:
        signal += np.random.normal(0, 0.05, size=len(t))

    return t, signal


if __name__ == "__main__":
    np.random.seed(42)    # same "random" noise every run, so results repeat
    t, signal = generate_signal(bpm=72)

    plt.plot(t, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Fake heartbeat at 72 BPM")
    plt.savefig("fake_heartbeat_out.png")
    plt.show()
