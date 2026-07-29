"""
Tests for transients.py.

Uses the same synthetic-beats-on-a-known-baseline construction as
test_baseline.py, then checks that the final per-beat metrics come out
numerically sensible, plus isolates the stale-percentage-crossing quirk.
"""

import numpy as np

from ..config import MuscleMotionConfig
from ..peaks import detect_peaks_legacy
from ..baseline import compute_baselines
from ..transients import analyze_transients, beats_to_records, _real_peaks


def _make_synthetic_beats(n=400, period=60, amplitude=100, baseline_level=10.0,
                           noise_scale=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = np.full(n, baseline_level, dtype=np.float64)
    # Keep a margin from both edges at least as large as the peak-detection
    # window's half-width, so every generated beat is actually detectable
    # (the detector, faithfully to the macro, can't find peaks too close
    # to either end of the signal).
    centers = np.arange(period // 2, n - period // 2, period)
    for c in centers:
        # asymmetric beat shape (faster upstroke than downstroke), closer to a real transient
        up = amplitude * np.exp(-0.5 * ((t - c) / (period / 12)) ** 2) * (t <= c)
        down = amplitude * np.exp(-0.5 * ((t - c) / (period / 6)) ** 2) * (t > c)
        signal += up + down
    signal += rng.normal(scale=noise_scale, size=n)
    return signal, centers


def _default_cfg(**overrides):
    defaults = dict(
        recorded_framerate=26,
        peak_detection_window=20,
        peak_threshold_pct=30,
        high_freq_baseline_detection=True,
        percentages=[10, 20, 30, 50, 90],
    )
    defaults.update(overrides)
    return MuscleMotionConfig(**defaults)


def test_analyze_transients_end_to_end_sane_values():
    signal, centers = _make_synthetic_beats()
    cfg = _default_cfg()

    peak_result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    baseline_result = compute_baselines(signal, peak_result.peaks, cfg)

    result = analyze_transients(signal, peak_result.peaks, baseline_result.baselines, cfg)

    print(f"Detected {result.n_peaks} beats, estimated BPM: {result.bpm_estimate:.1f}")
    for b in result.beats:
        print(f"  peak@{b.peak_index}: TTP={b.time_to_peak_ms}, RT={b.relaxation_time_ms}, "
              f"amp={b.contraction_amplitude:.1f}, %durations={b.percentage_durations_ms}")

    assert result.n_peaks == len(centers)

    # sanity bounds: every beat should have found its primary crossings
    # (clean synthetic data, generous window) and have a positive amplitude
    for b in result.beats:
        assert b.time_to_peak_ms is not None, "Expected a clean synthetic beat to find its TTP crossing."
        assert b.relaxation_time_ms is not None
        assert b.transient_duration_ms is not None
        assert b.contraction_amplitude > 50, "Expected a large, clearly-detected amplitude."
        for p, dur in b.percentage_durations_ms.items():
            assert dur >= 0, f"Percentage duration for {p}% should never be negative."

    # peak-to-peak time should be None only for the very first beat
    assert result.beats[0].peak_to_peak_time_ms is None
    for b in result.beats[1:]:
        assert b.peak_to_peak_time_ms is not None
        assert b.peak_to_peak_time_ms > 0


def test_bpm_estimate_matches_known_period():
    signal, centers = _make_synthetic_beats(period=60)
    cfg = _default_cfg()
    peak_result = detect_peaks_legacy(signal, 0, cfg=cfg)
    baseline_result = compute_baselines(signal, peak_result.peaks, cfg)
    result = analyze_transients(signal, peak_result.peaks, baseline_result.baselines, cfg)

    expected_bpm = 60000.0 / (60 * cfg.sampling_time_ms)  # period=60 frames -> ms -> bpm
    print(f"Expected BPM ~{expected_bpm:.1f}, got {result.bpm_estimate:.1f}")
    assert abs(result.bpm_estimate - expected_bpm) < 2.0


def test_beats_to_records_has_expected_columns():
    signal, centers = _make_synthetic_beats()
    cfg = _default_cfg()
    peak_result = detect_peaks_legacy(signal, 0, cfg=cfg)
    baseline_result = compute_baselines(signal, peak_result.peaks, cfg)
    result = analyze_transients(signal, peak_result.peaks, baseline_result.baselines, cfg)

    records = beats_to_records(result, cfg)
    assert len(records) == result.n_peaks

    expected_keys = {
        "Contraction duration [10% above baseline] (ms)",
        "Time-to-peak (ms)",
        "Relaxation Time (ms)",
        "Peak-to-peak time (ms)",
        "Baseline value (a.u.)",
        "Peak amplitude (a.u.)",
        "Contraction amplitude (a.u.)",
    }
    for p in cfg.percentages:
        expected_keys.add(f"{100 - p}-to-{100 - p} transient (ms)")

    assert expected_keys.issubset(records[0].keys())
    print("Sample record:", records[0])


def test_single_peak_uses_full_signal_fallback_window():
    """
    Only one real peak: no neighbor to size a search window from. Confirm
    this doesn't crash and produces a sensible (non-degenerate) result
    rather than reproducing the macro's meaningless false-arithmetic edge
    case (see module docstring).
    """
    n = 200
    signal = np.full(n, 10.0)
    signal += 100 * np.exp(-0.5 * ((np.arange(n) - 100) / 5) ** 2)
    cfg = _default_cfg(peak_detection_window=20, peak_threshold_pct=30)

    peak_result = detect_peaks_legacy(signal, 0, cfg=cfg)
    real = _real_peaks(peak_result.peaks)
    assert len(real) == 1, "Test setup should produce exactly one real peak."

    baseline_result = compute_baselines(signal, peak_result.peaks, cfg)
    result = analyze_transients(signal, peak_result.peaks, baseline_result.baselines, cfg)

    assert result.n_peaks == 1
    b = result.beats[0]
    print("Single-peak result:", b)
    assert b.time_to_peak_ms is not None
    assert b.peak_to_peak_time_ms is None


def test_legacy_stale_percentage_crossing_flag_has_no_observable_effect():
    """
    IMPORTANT CORRECTION to the initial hypothesis: reading the macro's
    source suggested a "stale crossing reused from a previous peak" bug
    might be observable. Working through the math shows it CANNOT
    actually happen: percentages must ascend (config.py enforces this),
    so the primary (smallest, strictest) percentage's level is always the
    lowest. Any point satisfying that strict threshold automatically
    satisfies every larger (looser) percentage's threshold too — and
    since every percentage searches the SAME shared window for the same
    peak, the primary crossing succeeding GUARANTEES every larger
    percentage's crossing also succeeds within that window. There is no
    reachable case where the primary gate passes but a specific
    percentage's own crossing genuinely fails for that same peak.

    This test verifies that directly: legacy_stale_percentage_crossing_bug
    should produce IDENTICAL results to the corrected default across a
    variety of scenarios, because the "stale" code path is structurally
    present but never actually reachable.
    """
    scenarios = []

    signal1, _ = _make_synthetic_beats()
    cfg1 = _default_cfg()
    peaks1 = detect_peaks_legacy(signal1, 0, cfg=cfg1).peaks
    baselines1 = compute_baselines(signal1, peaks1, cfg1).baselines
    scenarios.append((signal1, peaks1, baselines1, cfg1))

    n = 200
    baseline_level = 10.0
    t = np.arange(n)
    signal2 = np.full(n, baseline_level, dtype=np.float64)
    for c in (25, 105, 185):
        signal2 += 100 * np.exp(-0.5 * ((t - c) / 8.0) ** 2)
    peaks2 = [25, 105, 185]
    cfg2 = _default_cfg(percentages=[10, 50, 90])
    baselines2 = np.array([baseline_level] * 3)
    scenarios.append((signal2, peaks2, baselines2, cfg2))

    for i, (signal, peaks, baselines, cfg) in enumerate(scenarios):
        corrected = analyze_transients(signal, peaks, baselines, cfg,
                                        legacy_stale_percentage_crossing_bug=False)
        legacy = analyze_transients(signal, peaks, baselines, cfg,
                                     legacy_stale_percentage_crossing_bug=True)

        for b_corrected, b_legacy in zip(corrected.beats, legacy.beats):
            assert b_corrected.percentage_durations_ms == b_legacy.percentage_durations_ms, (
                f"Scenario {i}: expected legacy and corrected modes to agree "
                f"exactly, got corrected={b_corrected.percentage_durations_ms} "
                f"vs legacy={b_legacy.percentage_durations_ms}"
            )

    print("Confirmed: legacy_stale_percentage_crossing_bug never changes output, "
          "across all tested scenarios — matching the mathematical guarantee.")


if __name__ == "__main__":
    test_analyze_transients_end_to_end_sane_values()
    test_bpm_estimate_matches_known_period()
    test_beats_to_records_has_expected_columns()
    test_single_peak_uses_full_signal_fallback_window()
    test_legacy_stale_percentage_crossing_flag_has_no_observable_effect()
    print("\nAll transients.py tests passed.")