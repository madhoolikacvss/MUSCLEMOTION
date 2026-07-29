"""
Tests for signals.py.

These use hand-constructed synthetic stacks where the correct numeric
answer is known exactly, so we can assert on precise values rather than
just "did it run" — the whole point of this stage is that both signals
reduce to simple, checkable arithmetic.
"""

import numpy as np

from ..config import MuscleMotionConfig
from ..signals import (
    compute_contraction_signal,
    compute_speed_signal,
    compute_time_axis,
    compute_signals,
)


def test_contraction_signal_matches_hand_computed_values():
    """
    frame_i = reference + i  (constant offset, same everywhere in the frame)
    => |frame_i - reference| = i everywhere => mean = i, exactly.
    """
    H, W, n = 5, 5, 8
    ref = np.zeros((H, W), dtype=np.float32)
    stack = np.stack([np.full((H, W), i, dtype=np.float32) for i in range(n)], axis=0)

    cfg = MuscleMotionConfig(gaussian_blur=False)
    signal = compute_contraction_signal(stack, ref, mask=None, cfg=cfg)

    expected = np.arange(n, dtype=np.float32)
    np.testing.assert_allclose(signal, expected, atol=1e-5)
    print("Contraction signal matches hand-computed values:", signal)


def test_speed_signal_matches_hand_computed_values():
    """
    frame_i = 2*i everywhere. With speed_window=3:
    |frame_i - frame_{i+3}| = |2i - 2(i+3)| = 6, constant for every i.
    """
    H, W, n, sw = 4, 4, 10, 3
    stack = np.stack([np.full((H, W), 2 * i, dtype=np.float32) for i in range(n)], axis=0)

    cfg = MuscleMotionConfig(speed_window=sw, gaussian_blur=False)
    signal = compute_speed_signal(stack, mask=None, cfg=cfg)

    assert len(signal) == n - sw
    np.testing.assert_allclose(signal, np.full(n - sw, 6.0), atol=1e-5)
    print("Speed signal matches hand-computed values:", signal)


def test_speed_signal_rejects_window_too_large():
    stack = np.zeros((5, 4, 4), dtype=np.float32)
    cfg = MuscleMotionConfig(speed_window=10, gaussian_blur=False)
    try:
        compute_speed_signal(stack, mask=None, cfg=cfg)
        assert False, "Expected a ValueError for speed_window >= n_frames"
    except ValueError:
        pass


def test_macro_style_masking_dilutes_by_total_pixel_count():
    """
    Every pixel has the same diff value V. Mask covers exactly half the
    pixels. Macro-style masking should give V * 0.5 (denominator = ALL
    pixels), NOT V (which would be the restricted-mean answer).
    """
    H, W, n = 10, 10, 3
    V = 4.0
    ref = np.zeros((H, W), dtype=np.float32)
    stack = np.stack([np.full((H, W), V, dtype=np.float32) for _ in range(n)], axis=0)

    mask = np.zeros((H, W), dtype=bool)
    mask[:, :5] = True  # exactly half the pixels

    cfg = MuscleMotionConfig(gaussian_blur=False)

    macro_style = compute_contraction_signal(stack, ref, mask=mask, cfg=cfg, restrict_mean_to_mask=False)
    restricted = compute_contraction_signal(stack, ref, mask=mask, cfg=cfg, restrict_mean_to_mask=True)

    np.testing.assert_allclose(macro_style, np.full(n, V * 0.5), atol=1e-5)
    np.testing.assert_allclose(restricted, np.full(n, V), atol=1e-5)
    print(f"Macro-style (diluted) signal: {macro_style}, Restricted-mean signal: {restricted}")


def test_restricted_mean_handles_empty_mask_gracefully():
    H, W, n = 6, 6, 4
    ref = np.zeros((H, W), dtype=np.float32)
    stack = np.stack([np.full((H, W), 3.0, dtype=np.float32) for _ in range(n)], axis=0)
    empty_mask = np.zeros((H, W), dtype=bool)

    cfg = MuscleMotionConfig(gaussian_blur=False)
    result = compute_contraction_signal(stack, ref, mask=empty_mask, cfg=cfg, restrict_mean_to_mask=True)
    np.testing.assert_allclose(result, np.zeros(n), atol=1e-5)


def test_time_axis_matches_cumulative_construction():
    cfg = MuscleMotionConfig(recorded_framerate=26)
    n = 5
    axis = compute_time_axis(n, cfg)

    # rebuild the macro's cumulative-add version explicitly, and compare
    cumulative = np.zeros(n)
    for i in range(1, n):
        cumulative[i] = cumulative[i - 1] + cfg.sampling_time_ms
    np.testing.assert_allclose(axis, cumulative, atol=1e-9)
    print("Time axis (ms):", axis)


def test_compute_signals_end_to_end_shapes():
    H, W, n, sw = 8, 8, 30, 2
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 1, size=(H, W)).astype(np.float32)
    stack = np.stack(
        [ref + rng.normal(0, 1, size=(H, W)) for _ in range(n)], axis=0
    ).astype(np.float32)
    mask = np.ones((H, W), dtype=bool)

    cfg = MuscleMotionConfig(speed_window=sw, gaussian_blur=False)
    result = compute_signals(stack, ref, mask, cfg)

    assert result.contraction.shape == (n,)
    assert result.speed.shape == (n - sw,)
    assert result.time_contraction_ms.shape == (n,)
    assert result.time_speed_ms.shape == (n - sw,)
    print("End-to-end SignalResult shapes look correct.")


if __name__ == "__main__":
    test_contraction_signal_matches_hand_computed_values()
    test_speed_signal_matches_hand_computed_values()
    test_speed_signal_rejects_window_too_large()
    test_macro_style_masking_dilutes_by_total_pixel_count()
    test_restricted_mean_handles_empty_mask_gracefully()
    test_time_axis_matches_cumulative_construction()
    test_compute_signals_end_to_end_shapes()
    print("\nAll signals.py tests passed.")