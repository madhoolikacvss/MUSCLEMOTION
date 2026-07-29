"""
Tests for peaks.py.

Includes a test (`test_legacy_padding_bug_creates_fake_peak_from_nothing`)
that directly demonstrates a real bug found while porting the macro: a
completely flat signal with ZERO genuine peaks still produces one fake
peak under the macro's literal padding logic. This is very likely a
concrete contributor to your "1-2 false peaks on a non-contracting well"
symptom, independent of thresholding/masking — worth flagging prominently
rather than burying it as an implementation detail.
"""

import numpy as np

from ..config import MuscleMotionConfig
from ..peaks import detect_peaks_legacy, detect_peaks_scipy, detect_peaks, local_spacing, _real_peaks


def _make_synthetic_beats(n=300, period=50, amplitude=100, noise_scale=0.5, seed=0):
    """
    Clean, evenly-spaced beats: baseline near 0, sharp-ish peaks every
    `period` frames. Good for checking that peak detection finds
    approximately the right count in approximately the right places.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = np.zeros(n, dtype=np.float64)
    centers = np.arange(period // 2, n, period)
    for c in centers:
        signal += amplitude * np.exp(-0.5 * ((t - c) / (period / 8)) ** 2)
    signal += rng.normal(scale=noise_scale, size=n)
    return signal, centers


def test_legacy_detects_correct_number_of_clean_peaks():
    signal, centers = _make_synthetic_beats(n=300, period=50)
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)

    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    real = _real_peaks(result.peaks)

    print(f"Expected {len(centers)} peaks near {list(centers)}, found: {real}")
    assert len(real) == len(centers), (
        f"Expected {len(centers)} peaks, found {len(real)}: {real}"
    )
    # each detected peak should land close to its true center
    for detected, true_center in zip(real, centers):
        assert abs(detected - true_center) <= 5, (
            f"Detected peak {detected} too far from true center {true_center}"
        )


def test_scipy_detects_correct_number_of_clean_peaks():
    signal, centers = _make_synthetic_beats(n=300, period=50)
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)

    result = detect_peaks_scipy(signal, reference_frame_index=0, cfg=cfg)
    real = _real_peaks(result.peaks)

    print(f"[scipy] Expected {len(centers)} peaks, found: {real}")
    assert len(real) == len(centers)
    for detected, true_center in zip(real, centers):
        assert abs(detected - true_center) <= 5


def test_window_is_forced_even():
    signal, _ = _make_synthetic_beats(n=200, period=40)
    cfg = MuscleMotionConfig(peak_detection_window=21, peak_threshold_pct=30)  # odd on purpose
    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    assert result.window_used == 22, f"Expected 21 to be bumped to 22, got {result.window_used}"


def test_single_peak_gets_none_padding():
    n = 200
    signal = np.zeros(n)
    signal[100] = 100.0  # exactly one sharp spike
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)

    result = detect_peaks_legacy(signal, reference_frame_index=0, cfg=cfg)
    assert result.peaks[-1] is None, "Expected a trailing None to pad a single real peak"
    assert len(_real_peaks(result.peaks)) == 1


def test_legacy_padding_bug_creates_fake_peak_from_nothing():
    """
    THE key bug demonstration, isolated from any noise/thresholding effects:
    a perfectly CONSTANT signal (zero variance -> perc100 == perc0 -> the
    amplitude threshold is exactly 0, and the "> threshold" check can never
    pass) genuinely has ZERO frames that qualify as peaks under the
    algorithm's own logic. The corrected default (legacy_padding_bug=False)
    reports zero peaks, as it should. Under the macro's literal padding
    logic (legacy_padding_bug=True), it instead reports a fake peak at
    index 0 regardless — purely an artifact of how the peak list was
    initialized as a scalar 0 rather than an empty list, with nothing to
    do with thresholds, noise, or masking.
    """
    flat_signal = np.full(200, 50.0)  # perfectly flat, zero variance
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)

    corrected = detect_peaks_legacy(flat_signal, reference_frame_index=0, cfg=cfg,
                                     legacy_padding_bug=False)
    buggy = detect_peaks_legacy(flat_signal, reference_frame_index=0, cfg=cfg,
                                 legacy_padding_bug=True)

    print(f"Corrected result on a perfectly flat signal: {corrected.peaks} "
          f"({len(_real_peaks(corrected.peaks))} real peaks)")
    print(f"Legacy-buggy result on the SAME flat signal:  {buggy.peaks}")

    assert len(_real_peaks(corrected.peaks)) == 0, (
        "Corrected mode should report zero peaks for a perfectly flat signal "
        "(threshold is exactly 0, so no frame can ever clear it)."
    )
    assert 0 in buggy.peaks, (
        "Legacy padding bug should manifest as a fake peak at index 0, "
        "demonstrating a real false-positive source independent of thresholding."
    )


def test_self_referential_threshold_finds_false_peaks_in_pure_noise():
    """
    A SEPARATE, important finding (bucket-A from the false-positive
    discussion, NOT the padding bug): because the amplitude threshold is
    always computed as a PERCENTAGE OF THE SIGNAL'S OWN OBSERVED RANGE,
    pure sensor-noise-only data (no real contraction at all) still
    produces multiple "real" peaks — even with the padding bug fixed.
    Scaling the noise down doesn't help, because the threshold scales
    down proportionally with it. This test documents the behavior as
    currently expected (not something this module fixes on its own — that
    requires an external, non-self-referential noise floor, planned as a
    future improvement) so future changes to peaks.py don't accidentally
    hide this without us noticing.
    """
    rng = np.random.default_rng(2)
    noisy_flat_signal = rng.normal(loc=50, scale=0.5, size=200)  # pure noise, no real beats
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)

    result = detect_peaks_legacy(noisy_flat_signal, reference_frame_index=0, cfg=cfg,
                                  legacy_padding_bug=False)
    n_false_peaks = len(_real_peaks(result.peaks))
    print(f"Pure-noise signal (no real contraction) still produced "
          f"{n_false_peaks} 'peaks' via the self-referential threshold: {result.peaks}")

    assert n_false_peaks > 0, (
        "This documents the known bucket-A limitation: a self-referential "
        "amplitude threshold cannot tell 'pure noise' apart from 'a real, "
        "smaller-amplitude signal' — this is expected with the current "
        "thresholding approach, not a bug in this test."
    )


def test_index_bug_changes_perc0_estimate():
    signal, _ = _make_synthetic_beats(n=300, period=50)
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)

    # reference_frame_index deliberately far from 0, to make the two modes diverge
    ref_idx = 250
    corrected = detect_peaks_legacy(signal, reference_frame_index=ref_idx, cfg=cfg,
                                     legacy_index_bug=False)
    legacy = detect_peaks_legacy(signal, reference_frame_index=ref_idx, cfg=cfg,
                                  legacy_index_bug=True)

    print(f"Corrected perc0 (robust median): {corrected.perc0:.2f}")
    print(f"Legacy perc0 (index into shifted array): {legacy.perc0:.2f}")
    assert corrected.perc0 != legacy.perc0


def test_local_spacing_uses_next_or_previous_peak():
    peaks = [10, 40, 100, None]  # three real peaks, one padding slot
    assert local_spacing(peaks, 0) == 30   # next peak: 40-10
    assert local_spacing(peaks, 1) == 60   # next peak: 100-40
    assert local_spacing(peaks, 2) == 60   # last real peak -> falls back to previous distance


def test_dispatcher_routes_to_correct_method():
    signal, _ = _make_synthetic_beats(n=200, period=40)
    cfg = MuscleMotionConfig(peak_detection_window=20, peak_threshold_pct=30)
    r1 = detect_peaks(signal, 0, cfg, method="legacy")
    r2 = detect_peaks(signal, 0, cfg, method="scipy")
    assert r1.method == "legacy"
    assert r2.method == "scipy"


if __name__ == "__main__":
    test_legacy_detects_correct_number_of_clean_peaks()
    test_scipy_detects_correct_number_of_clean_peaks()
    test_window_is_forced_even()
    test_single_peak_gets_none_padding()
    test_legacy_padding_bug_creates_fake_peak_from_nothing()
    test_self_referential_threshold_finds_false_peaks_in_pure_noise()
    test_index_bug_changes_perc0_estimate()
    test_local_spacing_uses_next_or_previous_peak()
    test_dispatcher_routes_to_correct_method()
    print("\nAll peaks.py tests passed.")