"""
Tests for pipeline.py — the first tests in this project that exercise the
ENTIRE chain end-to-end, starting from a raw synthetic (n_frames, H, W)
stack rather than a pre-computed 1D signal like earlier stages' tests.
"""

import os
import shutil
import tempfile

import numpy as np

from ..config import MuscleMotionConfig
from ..pipeline import (
    run_pipeline,
    save_well_outputs,
    well_summary_row,
    run_batch,
    LegacyFlags,
    speed_linearity_qc,
)


def _make_synthetic_video(n_frames=300, H=30, W=30, n_beats=5, amplitude=60,
                           blob_slice=(10, 20, 10, 20), noise_scale=1.0, seed=0):
    """
    A synthetic "well" video: a resting background plus a blob region that
    pulses n_beats times, evenly spaced with margin from both edges (same
    margin logic as earlier stages' tests, so peaks are actually detectable).
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=100, scale=1.0, size=(H, W)).astype(np.float32)

    period = n_frames // (n_beats + 1)
    centers = np.arange(period, n_frames - period // 2, period)[:n_beats]

    stack = np.zeros((n_frames, H, W), dtype=np.float32)
    r0, r1, c0, c1 = blob_slice
    for i in range(n_frames):
        frame = base + rng.normal(scale=noise_scale, size=(H, W))
        for c in centers:
            phase = np.exp(-0.5 * ((i - c) / 4.0) ** 2)
            frame[r0:r1, c0:c1] += phase * amplitude
        stack[i] = frame

    return stack, centers


def _default_cfg(**overrides):
    defaults = dict(
        recorded_framerate=26,
        speed_window=2,
        max_project=True,
        reference_frame_mode="autodetect",
        auto_detect_start=1,
        auto_detect_stop=9999,
        low_value_n=20,
        unity_selection_n=10,
        automatic_transient_detection=True,
        peak_detection_window=16,
        peak_threshold_pct=30,
        high_freq_baseline_detection=True,
    )
    defaults.update(overrides)
    return MuscleMotionConfig(**defaults)


def test_run_pipeline_end_to_end_detects_expected_beats():
    stack, centers = _make_synthetic_video(n_frames=300, n_beats=5)
    cfg = _default_cfg()

    result = run_pipeline(stack, cfg, well_name="A1")

    print(f"Reference frame chosen: {result.reference_frame_result.index} "
          f"(mode={result.reference_frame_result.mode})")
    print(f"Detected peaks: {result.peak_result.diagnostics['n_real_peaks']} "
          f"(expected {len(centers)})")
    print(f"Speed-linearity QC correlation: {result.speed_linearity_correlation:.2f}")
    print(f"BPM estimate: {result.transient_result.bpm_estimate:.1f}")

    assert result.n_frames == 300
    assert result.mask_result is not None, "max_project=True should produce a mask."
    assert result.peak_result.diagnostics["n_real_peaks"] == len(centers)
    assert result.speed_linearity_correlation is not None
    assert result.speed_linearity_correlation > 0.3, (
        "Expected a reasonably positive speed-linearity correlation for a clean synthetic well."
    )
    assert len(result.beat_records) == len(centers)


def test_run_pipeline_without_transient_detection():
    stack, _ = _make_synthetic_video(n_frames=200, n_beats=3)
    cfg = _default_cfg(automatic_transient_detection=False)

    result = run_pipeline(stack, cfg, well_name="A2")

    assert result.peak_result is None
    assert result.baseline_result is None
    assert result.transient_result is None
    assert result.beat_records is None
    # signals should still have been computed regardless
    assert len(result.signals.contraction) > 0


def test_run_pipeline_without_masking():
    stack, centers = _make_synthetic_video(n_frames=250, n_beats=4)
    cfg = _default_cfg(max_project=False)

    result = run_pipeline(stack, cfg, well_name="A3")

    assert result.mask_result is None
    assert result.peak_result.diagnostics["n_real_peaks"] == len(centers)


def test_low_framerate_warning_is_recorded():
    stack, _ = _make_synthetic_video(n_frames=150, n_beats=2)
    cfg = _default_cfg(recorded_framerate=26)  # your actual default, well below the 50fps guidance

    result = run_pipeline(stack, cfg, well_name="A4")
    assert any("low" in w.lower() for w in result.warnings), (
        "Expected a low-framerate warning to be recorded, matching the macro's own check."
    )


def test_save_well_outputs_writes_expected_files():
    stack, _ = _make_synthetic_video(n_frames=200, n_beats=3)
    cfg = _default_cfg()
    result = run_pipeline(stack, cfg, well_name="B1")

    tmp_dir = tempfile.mkdtemp()
    try:
        paths = save_well_outputs(result, tmp_dir)
        print("Files written:", paths)

        for key in ("contraction_txt", "speed_txt", "overview_csv", "log_txt"):
            assert key in paths
            assert os.path.exists(paths[key]), f"Expected {key} to exist at {paths[key]}"
            assert os.path.getsize(paths[key]) > 0

        with open(paths["overview_csv"]) as f:
            header = f.readline()
        assert "Contraction duration" in header
        assert "Peak amplitude (a.u.)" in header
    finally:
        shutil.rmtree(tmp_dir)


def test_well_summary_row_has_expected_fields():
    stack, centers = _make_synthetic_video(n_frames=250, n_beats=4)
    cfg = _default_cfg()
    result = run_pipeline(stack, cfg, well_name="C1")

    row = well_summary_row(result)
    print("Well summary row:", row)

    assert row["well_name"] == "C1"
    assert row["n_peaks"] == len(centers)
    assert row["bpm_estimate"] is not None
    assert row["mean_contraction_amplitude"] is not None


def test_run_batch_sequential_writes_plate_summary():
    stacks = []
    for seed, n_beats in [(0, 3), (1, 5), (2, 0)]:  # last "well" has NO beats at all
        stack, _ = _make_synthetic_video(n_frames=200, n_beats=n_beats, seed=seed)
        stacks.append(stack)

    cfg = _default_cfg()
    inputs = [("W1", stacks[0]), ("W2", stacks[1]), ("W3", stacks[2])]

    tmp_dir = tempfile.mkdtemp()
    try:
        results = run_batch(inputs, cfg, tmp_dir, n_jobs=1)
        assert len(results) == 3
        assert [r.well_name for r in results] == ["W1", "W2", "W3"]

        summary_path = os.path.join(tmp_dir, "plate_summary.csv")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            content = f.read()
        print("Plate summary:\n", content)
        assert "W1" in content and "W2" in content and "W3" in content

        # every well's individual output files should also exist
        for name in ("W1", "W2", "W3"):
            assert os.path.exists(os.path.join(tmp_dir, f"{name}_Contraction.txt"))
    finally:
        shutil.rmtree(tmp_dir)


def test_legacy_flags_all_legacy_runs_without_crashing():
    """
    Not a correctness check against a known reference (that needs the real
    demo dataset) — just confirms the full "everything set to legacy mode"
    path runs end-to-end without errors, since that's the configuration
    you'd use to validate against MUSCLEMOTION's own demo_results.
    """
    stack, _ = _make_synthetic_video(n_frames=200, n_beats=3)
    cfg = _default_cfg()
    legacy = LegacyFlags.all_legacy()

    result = run_pipeline(stack, cfg, well_name="LegacyTest", legacy=legacy)
    assert result.beat_records is not None
    print(f"All-legacy-mode run completed: {len(result.beat_records)} beats found.")


def test_speed_linearity_qc_low_for_pure_noise():
    """Sanity check on the QC metric itself: pure noise should NOT show strong speed linearity."""
    rng = np.random.default_rng(0)
    from ..signals import SignalResult
    n = 200
    fake_signals = SignalResult(
        contraction=rng.normal(size=n),
        speed=rng.normal(size=n - 2),
        time_contraction_ms=np.arange(n, dtype=float),
        time_speed_ms=np.arange(n - 2, dtype=float),
    )
    corr = speed_linearity_qc(fake_signals)
    print(f"Speed-linearity correlation for pure noise: {corr}")
    assert corr is None or abs(corr) < 0.5


if __name__ == "__main__":
    test_run_pipeline_end_to_end_detects_expected_beats()
    test_run_pipeline_without_transient_detection()
    test_run_pipeline_without_masking()
    test_low_framerate_warning_is_recorded()
    test_save_well_outputs_writes_expected_files()
    test_well_summary_row_has_expected_fields()
    test_run_batch_sequential_writes_plate_summary()    
    test_legacy_flags_all_legacy_runs_without_crashing()
    test_speed_linearity_qc_low_for_pure_noise()
    print("\nAll pipeline.py tests passed.")