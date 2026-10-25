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
