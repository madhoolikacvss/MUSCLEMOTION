"""
signals.py — Step 3 of the pipeline: turn the (reference-frame-removed,
optionally masked) stack into the two 1D signals everything downstream
operates on.

Both signals reduce to the same primitive used throughout MUSCLEMOTION:

    diff  = |frame_A - frame_B|
    value = mean(diff)     (optionally restricted/weighted by an ROI mask)

They differ only in what "frame_B" is:

    Contraction         : frame_B is the SAME fixed reference frame every
                          time -> tracks displacement from rest.
    Speed of contraction : frame_B is frame_A shifted `speed_window` frames
                          back -> tracks instantaneous velocity of motion.

A note on how masking is applied (read before changing anything)
------------------------------------------------------------------
The original macro multiplies the diff image by the binary mask and then
takes the mean over the ENTIRE image (denominator = total pixel count),
NOT a mean restricted to just the masked-in pixels. This means the more
background gets masked out, the more the resulting signal is diluted
towards zero — background pixels contribute literal zeros to the sum but
still count in the denominator.

This module reproduces that literal behavior by default
(`restrict_mean_to_mask=False`), so we can validate against MUSCLEMOTION's
own output first. `restrict_mean_to_mask=True` computes a proper masked
mean instead (denominator = number of masked-in pixels only) — this is
very likely a better-behaved signal (less arbitrarily diluted, more
directly comparable across wells with different mask sizes), and is kept
here as a one-flag opt-in for later, once we're doing deliberate
improvements rather than a faithful port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MuscleMotionConfig
from .utils import maybe_blur


@dataclass
class SignalResult:
    contraction: np.ndarray        # shape (n_frames,)
    speed: np.ndarray               # shape (n_frames - speed_window,)
    time_contraction_ms: np.ndarray  # shape matches contraction
    time_speed_ms: np.ndarray         # shape matches speed


def _apply_mask_macro_style(diff: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    diff: (n_frames, H, W). Multiplies by mask (broadcast over frames) and
    returns the per-frame mean over the WHOLE image, matching the macro's
    literal behavior (see module docstring).
    """
    if mask is None:
        return diff.mean(axis=(1, 2))
    masked = diff * mask[np.newaxis, :, :].astype(np.float32)
    return masked.mean(axis=(1, 2))


def _apply_mask_restricted(diff: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    Proper masked mean: denominator is the number of True pixels in the
    mask, not the whole image. Opt-in improvement, not the macro's literal
    behavior — see module docstring.
    """
    if mask is None or not mask.any():
        return diff.mean(axis=(1, 2)) if mask is None else np.zeros(diff.shape[0], dtype=np.float32)
    flat = diff.reshape(diff.shape[0], -1)
    mask_flat = mask.reshape(-1)
    return flat[:, mask_flat].mean(axis=1)


def compute_contraction_signal(
    stack_no_ref: np.ndarray,
    reference_frame: np.ndarray,
    mask: np.ndarray | None,
    cfg: MuscleMotionConfig,
    restrict_mean_to_mask: bool = False,
) -> np.ndarray:
    """
    Contraction signal: mean(|frame_i - reference_frame|) for every frame,
    vectorized across the whole stack at once (the macro loops frame by
    frame; here it's one batched numpy operation).
    """
    blurred_stack = maybe_blur(stack_no_ref, cfg.gaussian_blur)
    ref = reference_frame.astype(np.float32)

    diff = np.abs(blurred_stack - ref[np.newaxis, :, :])

    if restrict_mean_to_mask:
        return _apply_mask_restricted(diff, mask)
    return _apply_mask_macro_style(diff, mask)


def compute_speed_signal(
    stack_no_ref: np.ndarray,
    mask: np.ndarray | None,
    cfg: MuscleMotionConfig,
    restrict_mean_to_mask: bool = False,
) -> np.ndarray:
    """
    Speed-of-contraction signal: mean(|frame_i - frame_{i+speed_window}|),
    a sliding/rolling comparison rather than a fixed reference frame.
    Length is n_frames - speed_window (you lose `speed_window` samples off
    the end, same as the macro).
    """
    sw = cfg.speed_window
    n_frames = stack_no_ref.shape[0]
    if sw >= n_frames:
        raise ValueError(
            f"speed_window ({sw}) must be smaller than the number of frames ({n_frames})"
        )

    blurred_stack = maybe_blur(stack_no_ref, cfg.gaussian_blur)
    a = blurred_stack[: n_frames - sw]
    b = blurred_stack[sw:]

    diff = np.abs(a - b)

    if restrict_mean_to_mask:
        return _apply_mask_restricted(diff, mask)
    return _apply_mask_macro_style(diff, mask)


def compute_time_axis(n_samples: int, cfg: MuscleMotionConfig) -> np.ndarray:
    """
    Frame index -> time in ms, matching the macro's cumulative construction
    (xTimeArray[0]=0, xTimeArray[i]=xTimeArray[i-1]+samplingTime), which is
    equivalent to a simple arange * samplingTime.
    """
    return np.arange(n_samples, dtype=np.float64) * cfg.sampling_time_ms


def compute_signals(
    stack_no_ref: np.ndarray,
    reference_frame: np.ndarray,
    mask: np.ndarray | None,
    cfg: MuscleMotionConfig,
    restrict_mean_to_mask: bool = False,
) -> SignalResult:
    """Single entry point the pipeline orchestrator should call for Step 3."""
    contraction = compute_contraction_signal(
        stack_no_ref, reference_frame, mask, cfg, restrict_mean_to_mask
    )
    speed = compute_speed_signal(stack_no_ref, mask, cfg, restrict_mean_to_mask)

    return SignalResult(
        contraction=contraction,
        speed=speed,
        time_contraction_ms=compute_time_axis(len(contraction), cfg),
        time_speed_ms=compute_time_axis(len(speed), cfg),
    )