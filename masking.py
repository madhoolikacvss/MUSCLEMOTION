"""
masking.py — Step 2 of the pipeline: build a binary "pixels of interest" mask
so that later stages (Step 3 onward) can restrict their mean-intensity
calculations to pixels that actually move, instead of averaging over the
whole frame (background included).

This is a faithful literal port of the macro's `pixelsOfInterest()`:

    1. For every frame in [mp_start_range, mp_end_range), compute
       diff = |frame - reference_frame| (optionally Gaussian-blurred first).
    2. Keep a running ELEMENTWISE MAXIMUM across all these diff frames —
       i.e. for each pixel, "what's the single largest change it ever
       underwent, at any point in the scanned range."
    3. Threshold that running-max image at (mean + 1 standard deviation).
       Anything above the threshold becomes a "pixel of interest" (mask=True).

Known limitation (intentionally NOT fixed here)
------------------------------------------------
`mean + 1*std` is a blunt heuristic: by construction it will mark roughly
the top ~15-20% of ANY image's pixels as "of interest," even a completely
static, non-contracting recording — it has no concept of "there is no ROI
here." This module keeps that behavior for now so we can first validate a
faithful port against MUSCLEMOTION's own demo dataset. The planned
improvements (Otsu/bimodality thresholding, connected-component filtering,
requiring temporal persistence rather than a single running max) are the
direct fixes to "bucket A" from the false-positive discussion, and will
live in this same file as additional, swappable functions later — nothing
in signals.py or elsewhere should need to change when we do that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import MuscleMotionConfig
from .utils import maybe_blur


@dataclass
class SNRMaskResult:
    mask: np.ndarray            # bool array, shape (H, W). True = pixel of interest.
    running_max: np.ndarray      # float array, shape (H, W). Diagnostic: max-ever-change per pixel.
    threshold: float              # the mean + std cutoff that was applied
    mean_val: float
    std_val: float
    frame_range_used: tuple[int, int]   # (start, end) 0-based indices actually scanned


def _resolve_frame_range(n_frames: int, cfg: MuscleMotionConfig) -> tuple[int, int]:
    """
    Convert the config's 1-based, macro-style mp_start_range/mp_end_range
    (where -1 means 'to the end') into a 0-based [start, end) Python range,
    clamped to the stack's actual length.
    """
    start = cfg.mp_start_range - 1
    end = n_frames if cfg.mp_end_range == -1 else cfg.mp_end_range
    end = min(end, n_frames)
    start = max(0, min(start, end - 1))
    return start, end


def compute_snr_mask(
    stack_no_ref: np.ndarray,
    reference_frame: np.ndarray,
    cfg: MuscleMotionConfig,
) -> SNRMaskResult:
    """
    Build the binary ROI mask from the (reference-frame-removed) stack.

    Parameters
    ----------
    stack_no_ref : (n_frames, H, W) array — the working stack, reference
        frame already excluded (see reference_frame.remove_reference_frame).
    reference_frame : (H, W) array — the chosen reference frame (already
        blurred if cfg.gaussian_blur was on when it was selected).
    cfg : MuscleMotionConfig
    """
    n_frames = stack_no_ref.shape[0]
    start, end = _resolve_frame_range(n_frames, cfg)

    ref = reference_frame.astype(np.float32)
    running_max = np.zeros(ref.shape, dtype=np.float32)

    for i in range(start, end):
        frame = maybe_blur(stack_no_ref[i], cfg.gaussian_blur)
        diff = np.abs(frame - ref)
        np.maximum(running_max, diff, out=running_max)

    mean_val = float(running_max.mean())
    std_val = float(running_max.std())
    threshold = mean_val + std_val

    mask = running_max >= threshold

    return SNRMaskResult(
        mask=mask,
        running_max=running_max,
        threshold=threshold,
        mean_val=mean_val,
        std_val=std_val,
        frame_range_used=(start, end),
    )


def get_mask_or_none(
    stack_no_ref: np.ndarray,
    reference_frame: np.ndarray,
    cfg: MuscleMotionConfig,
) -> Optional[np.ndarray]:
    """
    Convenience entry point for the pipeline orchestrator: returns the
    boolean mask if cfg.max_project is enabled, else None (meaning
    "don't restrict — average over the whole frame," matching the macro's
    behavior when noise-reduction is turned off).
    """
    if not cfg.max_project:
        return None
    return compute_snr_mask(stack_no_ref, reference_frame, cfg).mask