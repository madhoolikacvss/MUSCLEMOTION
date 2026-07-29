"""
Tests for masking.py.

Not a validation against MUSCLEMOTION's literal demo-dataset output yet
(that comes once the full pipeline exists) — these are synthetic sanity
checks:

    1. A stack with a genuinely moving "blob" region against a static,
       noisy background should produce a mask that meaningfully
       concentrates on the blob, not just picks random noise pixels.
    2. mp_start_range / mp_end_range clamping behaves as documented.
    3. max_project=False disables masking entirely (returns None).
    4. The known limitation (mask is never empty, even for a fully static
       recording) is explicitly demonstrated so it's tracked, not silently
       assumed — this is exactly the "bucket A" behavior we plan to fix
       later, so we want a test that will visibly need updating (not
       silently break) once that fix lands.
"""

import numpy as np

from ..config import MuscleMotionConfig
from ..masking import compute_snr_mask, get_mask_or_none, _resolve_frame_range


def _make_blob_stack(n_frames=60, H=40, W=40, noise_scale=1.0, seed=0):
    """
    Background: static noise everywhere.
    A 10x10 blob region (rows 20-30, cols 20-30) brightens/dims over time
    while everything else stays flat — this is our stand-in "contracting
    tissue" region.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=100, scale=1.0, size=(H, W)).astype(np.float32)

    stack = np.zeros((n_frames, H, W), dtype=np.float32)
    for i in range(n_frames):
        frame = base + rng.normal(scale=noise_scale, size=(H, W))
        phase = np.sin(i / n_frames * 4 * np.pi)  # a few oscillations
        frame[20:30, 20:30] += phase * 50
        stack[i] = frame

    ref_frame = base.copy()  # a "clean" resting reference, no noise, no blob
    return stack, ref_frame


def test_mask_concentrates_on_moving_region():
    stack, ref_frame = _make_blob_stack()
    cfg = MuscleMotionConfig(max_project=True, mp_start_range=1, mp_end_range=-1)

    result = compute_snr_mask(stack, ref_frame, cfg)

    blob_region = result.mask[20:30, 20:30]
    background_region = np.delete(
        np.delete(result.mask, np.s_[20:30], axis=0), np.s_[20:30], axis=1
    )

    blob_fraction = blob_region.mean()
    background_fraction = background_region.mean()

    print(f"Fraction of blob pixels marked 'of interest':       {blob_fraction:.2f}")
    print(f"Fraction of background pixels marked 'of interest': {background_fraction:.2f}")

    assert blob_fraction > background_fraction, (
        "Expected the mask to concentrate on the moving blob region far more "
        "than on the static background."
    )
    assert blob_fraction > 0.5, "Expected most of the blob region to be flagged."


def test_frame_range_resolution_defaults_to_full_stack():
    n_frames = 100
    cfg = MuscleMotionConfig(mp_start_range=1, mp_end_range=-1)
    start, end = _resolve_frame_range(n_frames, cfg)
    assert start == 0
    assert end == n_frames


def test_frame_range_resolution_respects_explicit_subrange():
    n_frames = 100
    cfg = MuscleMotionConfig(mp_start_range=10, mp_end_range=50)
    start, end = _resolve_frame_range(n_frames, cfg)
    assert start == 9   # 1-based 10 -> 0-based 9
    assert end == 50


def test_frame_range_resolution_clamps_overshoot():
    n_frames = 30
    cfg = MuscleMotionConfig(mp_start_range=1, mp_end_range=9999)
    start, end = _resolve_frame_range(n_frames, cfg)
    assert end == n_frames


def test_max_project_disabled_returns_none():
    stack, ref_frame = _make_blob_stack(n_frames=20)
    cfg = MuscleMotionConfig(max_project=False)
    mask = get_mask_or_none(stack, ref_frame, cfg)
    assert mask is None


def test_known_limitation_static_recording_still_produces_nonempty_mask():
    """
    Documents the known bucket-A limitation: a recording with NO real
    motion at all still produces a non-empty mask, because mean+std
    thresholding always marks roughly the top ~15-20% of any distribution.
    This test should start FAILING once we add a bimodality/Otsu-based
    "is there even a real ROI here" check — that's the intended signal
    that the fix has landed, not a bug in this test.
    """
    rng = np.random.default_rng(1)
    H, W, n_frames = 40, 40, 60
    base = rng.normal(loc=100, scale=1.0, size=(H, W)).astype(np.float32)
    stack = np.stack(
        [base + rng.normal(scale=1.0, size=(H, W)) for _ in range(n_frames)], axis=0
    ).astype(np.float32)
    ref_frame = base.copy()

    cfg = MuscleMotionConfig(max_project=True)
    result = compute_snr_mask(stack, ref_frame, cfg)

    print(f"Static recording: {result.mask.sum()} / {result.mask.size} pixels flagged "
          f"'of interest' despite no real motion (expected, current behavior).")
    assert result.mask.sum() > 0, (
        "This documents the known bucket-A flaw: masking never returns 'no ROI' "
        "even when the recording is fully static."
    )


if __name__ == "__main__":
    test_mask_concentrates_on_moving_region()
    test_frame_range_resolution_defaults_to_full_stack()
    test_frame_range_resolution_respects_explicit_subrange()
    test_frame_range_resolution_clamps_overshoot()
    test_max_project_disabled_returns_none()
    test_known_limitation_static_recording_still_produces_nonempty_mask()
    print("\nAll masking.py tests passed.")