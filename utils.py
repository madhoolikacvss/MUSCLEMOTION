"""
utils.py — small, stage-agnostic helpers shared by multiple modules.

Nothing algorithm-specific lives here (no reference-frame logic, no peak
logic, etc.) — just building blocks that several stages happen to need:
blurring, and a couple of robust statistics helpers used later once we
get to the false-positive-reduction work.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


GAUSSIAN_BLUR_SIGMA = 10.0   # matches the macro's hardcoded "Gaussian Blur... sigma=10"


def blur_frame(frame: np.ndarray, sigma: float = GAUSSIAN_BLUR_SIGMA) -> np.ndarray:
    """
    Apply the same 2D Gaussian blur the macro applies before every frame diff
    (run("Gaussian Blur...", "sigma=10")), to a single 2D frame.
    """
    return gaussian_filter(frame.astype(np.float32), sigma=sigma)


def blur_stack(stack: np.ndarray, sigma: float = GAUSSIAN_BLUR_SIGMA) -> np.ndarray:
    """
    Apply the same blur to every frame of a 3D stack (n_frames, H, W) at once,
    vectorized instead of looping frame-by-frame like the macro does.
    sigma=0 on the frame axis so frames are blurred independently, not into
    each other.
    """
    return gaussian_filter(stack.astype(np.float32), sigma=(0, sigma, sigma))


def maybe_blur(x: np.ndarray, enabled: bool) -> np.ndarray:
    """Convenience wrapper: blur only if the config flag is on, else pass through as float32."""
    if enabled:
        return blur_frame(x) if x.ndim == 2 else blur_stack(x)
    return x.astype(np.float32)


def mean_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    The one primitive nearly every MUSCLEMOTION stage is built from:
    mean(|a - b|), optionally restricted to a boolean mask (True = keep).
    Works for a single frame pair.
    """
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None:
        return float(diff[mask].mean()) if mask.any() else 0.0
    return float(diff.mean())


def robust_baseline_value(values: np.ndarray) -> float:
    """
    Median-based stand-in for '(perc0 = a single frame's value)' in the original
    macro. Not used yet in the literal port (kept faithful there), but available
    for the more robust variants we build later.
    """
    return float(np.median(values))
