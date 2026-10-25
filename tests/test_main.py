"""Tests for the heart-rate analyser."""

import numpy as np
import pytest

from main import (
    calculate_bpm,
    filter_signal,
    find_beats,
    generate_signal,
    load_signal,
    save_signal,
)


def test_generate_signal_length():
    """10 seconds at 250 Hz should give 2500 samples."""
    t, signal = generate_signal(duration=10, sample_rate=250)
    assert len(t) == 2500
    assert len(signal) == 2500


def test_calculate_bpm_known_answer():
    """Beats exactly 1 second apart should give 60 BPM."""
    peaks = np.array([0, 250, 500])
    bpm, gaps = calculate_bpm(peaks, 250)
    assert bpm == pytest.approx(60)
    assert list(gaps) == [1.0, 1.0]


def test_calculate_bpm_one_beat_exits():
    """One beat has no gap, so the function should stop."""
    with pytest.raises(SystemExit):
        calculate_bpm(np.array([100]), 250)


def test_find_beats():
    t, signal = generate_signal(bpm=72, drift=False, hum=False, noise=False)
    assert len(find_beats(signal, 250)) == 12


def test_save_load(tmp_path):
    t, signal = generate_signal(bpm=72, drift=False, hum=False, noise=False)
    path = tmp_path / "x.csv"
    save_signal(t, signal, path)
    t2, signal2 = load_signal(path)
    # Assert both time and signal arrays match within tolerance
    assert np.allclose(t, t2)
    assert np.allclose(signal, signal2)


def test_filter_rejects_low_sample_rate():
    with pytest.raises(SystemExit):
        signal = np.zeros(100)
        filter_signal(signal, 20)


@pytest.mark.parametrize("bpm", [55, 72, 110])
def test_full_pipeline(bpm):
    np.random.seed(42)
    rate = 250
    t, signal = generate_signal(bpm=bpm)
    clean = filter_signal(signal, rate)
    peaks = find_beats(clean, rate)
    result, gaps = calculate_bpm(peaks, rate)
    assert result == pytest.approx(bpm, abs=1.5)
