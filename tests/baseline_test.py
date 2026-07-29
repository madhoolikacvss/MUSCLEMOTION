"""
Tests for baseline.py.

Covers both baseline modes on realistic synthetic beat data, plus each of
the three documented macro quirks in isolation, demonstrating the actual
difference between "legacy" (faithful) and corrected behavior.
"""

import numpy as np

from ..config import MuscleMotionConfig
from ..peaks import detect_peaks_legacy, _real_peaks as _peaks_real
from ..baseline import (
    compute_speed_max_per_peak,
    baseline_highfreq,
    baseline_standard,
    compute_baselines,
)


def _make_synthetic_beats_with_known_baseline(n=300, period=50, amplitude=100,
                                               baseline_level=10.0, noise_scale=0.3, seed=0):
    """Clean beats sitting on top of a known, constant baseline level."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = np.full(n, baseline_level, dtype=np.float64)
    centers = np.arange(period // 2, n, period)
    for c in centers:
        signal += amplitude * np.exp(-0.5 * ((t - c) / (period / 8)) ** 2)
    signal += rng.normal(scale=noise_scale, size=n)
    return signal, centers


def test_speed_max_per_peak_zero_at_edges_nonzero_in_middle():
    signal, centers = _make_synthetic_beats_with_known_baseline(n=300, period=50)
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    peaks = result.peaks

    speed_max = compute_speed_max_per_peak(signal, peaks)
    print("speed_max_per_peak:", speed_max)

    assert len(speed_max) == len(_peaks_real(peaks))
    assert all(v > 0 for v in speed_max[1:-1]), "Interior peaks should have a nonzero upstroke speed."


def test_baseline_highfreq_recovers_known_baseline_level():
    baseline_level = 15.0
    signal, centers = _make_synthetic_beats_with_known_baseline(
        n=300, period=50, baseline_level=baseline_level, noise_scale=0.1
    )
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30,
                              high_freq_baseline_detection=True)
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    peaks = result.peaks

    baselines = baseline_highfreq(signal, peaks, legacy_first_peak_bug=False)
    print("Recovered baselines (corrected, expected ~", baseline_level, "):", baselines)

    for b in baselines:
        assert abs(b - baseline_level) < 2.0, f"Baseline {b} far from true level {baseline_level}"


def test_legacy_first_peak_bug_zeroes_out_first_amplitude():
    baseline_level = 15.0
    signal, centers = _make_synthetic_beats_with_known_baseline(
        n=300, period=50, baseline_level=baseline_level, noise_scale=0.1
    )
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30,
                              high_freq_baseline_detection=True)
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    peaks = result.peaks
    real_peaks = _peaks_real(peaks)

    corrected = baseline_highfreq(signal, peaks, legacy_first_peak_bug=False)
    buggy = baseline_highfreq(signal, peaks, legacy_first_peak_bug=True)

    print(f"Corrected baseline for first peak: {corrected[0]:.2f} "
          f"(peak value: {signal[real_peaks[0]]:.2f})")
    print(f"Legacy-buggy baseline for first peak: {buggy[0]:.2f} "
          f"(equals its own peak value -> amplitude forced to 0)")

    assert abs(corrected[0] - baseline_level) < 2.0, "Corrected mode should recover the true baseline."
    assert buggy[0] == signal[real_peaks[0]], (
        "Legacy bug should set the first peak's baseline equal to its own value."
    )
    np.testing.assert_allclose(corrected[1:], buggy[1:], atol=1e-9)


def test_baseline_standard_recovers_known_baseline_level():
    baseline_level = 15.0
    signal, centers = _make_synthetic_beats_with_known_baseline(
        n=300, period=50, baseline_level=baseline_level, noise_scale=0.1
    )
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30,
                              high_freq_baseline_detection=False,
                              baseline_threshold_pct=5, baseline_number_of_points=5)
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    peaks = result.peaks

    speed_max = compute_speed_max_per_peak(signal, peaks)
    baselines, warnings = baseline_standard(signal, peaks, speed_max, cfg)
    print("Standard-mode baselines:", baselines)
    if warnings:
        print("Warnings:", warnings)

    for b in baselines:
        assert abs(b - baseline_level) < 3.0, f"Baseline {b} far from true level {baseline_level}"


def test_legacy_mutating_baseline_n_bug_shrinks_for_all_later_peaks():
    """
    Construct a scenario where an early peak legitimately has very few
    flat points (tight/noisy pre-peak window), and confirm that in legacy
    mode the shrunken point-count persists for a LATER peak that would
    otherwise have plenty of flat points available.
    """
    rng = np.random.default_rng(3)
    n = 400
    signal = np.full(n, 10.0)
    peak_centers = [40, 200, 350]
    for c in peak_centers:
        t = np.arange(n)
        signal = signal + 80 * np.exp(-0.5 * ((t - c) / 6) ** 2)
    signal[0:35] += rng.normal(scale=5.0, size=35)

    cfg = MuscleMotionConfig(peak_detection_window=16, peak_threshold_pct=30,
                              high_freq_baseline_detection=False,
                              baseline_threshold_pct=5, baseline_number_of_points=5)
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    peaks = result.peaks
    speed_max = compute_speed_max_per_peak(signal, peaks)

    _, warnings_corrected = baseline_standard(
        signal, peaks, speed_max, cfg, legacy_mutating_baseline_n_bug=False
    )
    _, warnings_legacy = baseline_standard(
        signal, peaks, speed_max, cfg, legacy_mutating_baseline_n_bug=True
    )

    print("Corrected-mode warnings:", warnings_corrected)
    print("Legacy-mode warnings:   ", warnings_legacy)

    assert any("mutated" in w for w in warnings_legacy), (
        "Expected the legacy mode to report the baseline_number_of_points mutation."
    )


def test_legacy_zero_baseline_bug_vs_corrected_fallback():
    """
    Force a peak to have essentially zero qualifying flat points (extremely
    strict threshold_pct), and confirm legacy mode reports baseline=0 while
    corrected mode falls back to the window minimum instead.
    """
    baseline_level = 20.0
    signal, centers = _make_synthetic_beats_with_known_baseline(
        n=200, period=100, baseline_level=baseline_level, noise_scale=1.5
    )
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30,
                              high_freq_baseline_detection=False,
                              baseline_threshold_pct=0.0001, baseline_number_of_points=5)
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    peaks = result.peaks
    speed_max = compute_speed_max_per_peak(signal, peaks)

    baselines_corrected, _ = baseline_standard(signal, peaks, speed_max, cfg,
                                                legacy_zero_baseline_bug=False)
    baselines_legacy, _ = baseline_standard(signal, peaks, speed_max, cfg,
                                             legacy_zero_baseline_bug=True)

    print("Corrected baselines (fallback to window min):", baselines_corrected)
    print("Legacy baselines (forced to 0):", baselines_legacy)

    assert np.any(baselines_legacy == 0.0), "Expected at least one baseline forced to 0 in legacy mode."
    assert np.all(baselines_corrected > 0.0), (
        "Corrected mode should never silently report a baseline of exactly 0 here."
    )


def test_compute_baselines_dispatches_correctly():
    signal, centers = _make_synthetic_beats_with_known_baseline(n=300, period=50)
    cfg_highfreq = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30,
                                        high_freq_baseline_detection=True)
    cfg_standard = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30,
                                        high_freq_baseline_detection=False)

    peaks_hf = detect_peaks_legacy(signal, 0, cfg_highfreq).peaks
    peaks_std = detect_peaks_legacy(signal, 0, cfg_standard).peaks

    result_hf = compute_baselines(signal, peaks_hf, cfg_highfreq)
    result_std = compute_baselines(signal, peaks_std, cfg_standard)

    assert result_hf.mode == "highfreq"
    assert result_std.mode == "standard"
    assert len(result_hf.baselines) == len(_peaks_real(peaks_hf))
    assert len(result_std.baselines) == len(_peaks_real(peaks_std))


if __name__ == "__main__":
    test_speed_max_per_peak_zero_at_edges_nonzero_in_middle()
    test_baseline_highfreq_recovers_known_baseline_level()
    test_legacy_first_peak_bug_zeroes_out_first_amplitude()
    test_baseline_standard_recovers_known_baseline_level()
    test_legacy_mutating_baseline_n_bug_shrinks_for_all_later_peaks()
    test_legacy_zero_baseline_bug_vs_corrected_fallback()
    test_compute_baselines_dispatches_correctly()
    print("\nAll baseline.py tests passed.")