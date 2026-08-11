"""
reference_frame.py: Step 1 of the pipeline: choose the single frame that
every other frame will be diffed against for the Contraction signal.

Three modes (mirrors UserManual section 6, option G):

    "manual"       : user gives an explicit frame index.
    "first_frame"  : frame 0.
    "autodetect"   : find a frame that is both QUIET (low overall motion)
                     and STABLE (motion isn't actively changing, i.e. not
                     mid-transition into/out of a beat)

NB:
In the original ImageJ macro, `autoDetectStart`/`autoDetectStop` are used
to slice the already-computed motion-scan array, but the scan itself
always starts at frame 1 regardless of `autoDetectStart`. The winning
index found in that sliced sub-array is then used directly as a 1-based
frame number, WITHOUT adding `autoDetectStart` back as an offset. 

This module fixes that by default (`legacy_offset_bug=False`): the scan
window is honored as written (starts at `auto_detect_start`), and the
returned index is correctly offset back into full-stack coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import MuscleMotionConfig
from utils import maybe_blur, mean_abs_diff


@dataclass
class ReferenceFrameResult:
    index: int                 # 0-based index into the ORIGINAL stack
    frame: np.ndarray          # the (optionally blurred) reference frame itself
    mode: str                  # which mode produced this result, for logging/QC
    diagnostics: dict          # intermediate arrays, useful for debugging false positives later


def _clamp_autodetect_params(n_frames: int, cfg: MuscleMotionConfig) -> tuple[int, int, int, int]:
    
    start = cfg.auto_detect_start - 1          # convert to 0-based
    stop = cfg.auto_detect_stop - 1
    low_n = cfg.low_value_n
    unity_n = cfg.unity_selection_n

    max_valid_stop = n_frames - cfg.speed_window - 1
    if stop >= max_valid_stop:
        stop = max_valid_stop
    if stop <= start:
        start = stop - 1
    if low_n >= stop:
        low_n = stop - 1
    if low_n <= unity_n:
        unity_n = low_n

    return start, stop, low_n, unity_n


def autodetect_reference_frame(
    stack: np.ndarray,
    cfg: MuscleMotionConfig,
    legacy_offset_bug: bool = False,
) -> ReferenceFrameResult:
    """
      1. Coarse motion scan: speedY[k] = mean(|stack[k] - stack[k+speed_window]|)
      2. Pair each value with its immediate next neighbor.
      3. Score by overall motion magnitude (radian = sqrt(a^2 + b^2)); keep the
         `low_value_n` quietest candidates.
      4. Among those, score by how close the ratio a/b is to 1 ("unity line");
         keep the `unity_selection_n` flattest candidates.
      5. Pick the candidate minimizing (a * b * unity_score)
    """
    n_frames = stack.shape[0]
    sw = cfg.speed_window

    # OPTIMIZATION: Blur the ENTIRE stack ONCE, then slice it
    # Previously: blurred stack A and B separately (2x blurring)
    if cfg.gaussian_blur:
        from scipy.ndimage import gaussian_filter
        # Blur the entire stack once (vectorized across all frames)
        blurred_full = gaussian_filter(stack.astype(np.float32), sigma=(0, 10.0, 10.0))
    else:
        blurred_full = stack.astype(np.float32)
    
    # Slice the already-blurred stack
    a_full = blurred_full[:n_frames - sw]
    b_full = blurred_full[sw:]
    
    # Vectorized speed computation (mean across height and width)
    speed_y_full = np.abs(a_full - b_full).mean(axis=(1, 2))  # length n_frames - sw

    start, stop, low_n, unity_n = _clamp_autodetect_params(n_frames, cfg)

    if legacy_offset_bug:
        window = speed_y_full[start:stop]
        offset_correction = 0
    else:
        window = speed_y_full[start + 1:stop + 1]
        offset_correction = start + 1

    # Step 2: pair each value with its immediate next neighbor 
    speed_y = window[:-1]
    speed_y_shift = window[1:]

    if len(speed_y) < max(low_n, unity_n, 2):
        raise ValueError(
            "Not enough frames in the autodetect search window for the requested "
            "low_value_n / unity_selection_n: check auto_detect_start/stop and speed_window."
        )

    # Step 3: quiet filter (smallest overall motion magnitude) 
    radian = np.sqrt(speed_y**2 + speed_y_shift**2)
    quiet_candidates = np.argsort(radian)[:low_n]

    # Step 4: flatness filter (ratio closest to 1, i.e. near the "unity line") 
    # Calculate unity scores for ALL quiet candidates (no bug)
    with np.errstate(divide="ignore", invalid="ignore"):
        unity_scores = np.abs(speed_y[quiet_candidates] / speed_y_shift[quiet_candidates] - 1)
    unity_scores = np.nan_to_num(unity_scores, nan=np.inf, posinf=np.inf)
    
    # Sort by unity score and take the top unity_n candidates
    unity_order = np.argsort(unity_scores)[:unity_n]
    flattest_candidates = quiet_candidates[unity_order]
    flattest_unity_scores = unity_scores[unity_order]

    # Step 5: final combined score, quiet AND flat wins 
    combined_score = (
        speed_y[flattest_candidates] * speed_y_shift[flattest_candidates] * flattest_unity_scores
    )
    best_local = np.argmin(combined_score)
    best_index_in_window = flattest_candidates[best_local]

    ref_idx = int(best_index_in_window + offset_correction)
    ref_idx = max(0, min(ref_idx, n_frames - 1))  # safety clamp

    # Extract and blur the reference frame (only ONE frame, not the whole stack)
    ref_frame = maybe_blur(stack[ref_idx], cfg.gaussian_blur)

    return ReferenceFrameResult(
        index=ref_idx,
        frame=ref_frame,
        mode="autodetect",
        diagnostics={
            "speed_y_full": speed_y_full,
            "search_start": start,
            "search_stop": stop,
            "low_value_n_used": low_n,
            "unity_selection_n_used": unity_n,
            "quiet_candidates": quiet_candidates,
            "flattest_candidates": flattest_candidates,
            "combined_score": combined_score,
            "legacy_offset_bug": legacy_offset_bug,
            "unity_scores": unity_scores,
            "unity_order": unity_order,
        },
    )   

def manual_reference_frame(stack: np.ndarray, cfg: MuscleMotionConfig) -> ReferenceFrameResult:
    """User-specified reference frame index (0-based)."""
    idx = cfg.manual_reference_frame_index
    if idx is None or not (0 <= idx < stack.shape[0]):
        raise ValueError(f"manual_reference_frame_index out of range: {idx}")
    frame = maybe_blur(stack[idx], cfg.gaussian_blur)
    ref_frame = ReferenceFrameResult(index=idx, frame=frame, mode="manual", diagnostics={})
    return ref_frame


def first_frame_reference(stack: np.ndarray, cfg: MuscleMotionConfig) -> ReferenceFrameResult:
    """Trivial fallback: just use frame 0, matching the macro's final 'else' branch."""
    frame = maybe_blur(stack[0], cfg.gaussian_blur)
    return ReferenceFrameResult(index=0, frame=frame, mode="first_frame", diagnostics={})


def select_reference_frame(
    stack: np.ndarray,
    cfg: MuscleMotionConfig,
    legacy_offset_bug: bool = False,
) -> ReferenceFrameResult:
    """Single entry point the pipeline orchestrator should call for Step 1."""
    if cfg.reference_frame_mode == "manual":
        return manual_reference_frame(stack, cfg)
    if cfg.reference_frame_mode == "first_frame":
        return first_frame_reference(stack, cfg)
    if cfg.reference_frame_mode == "autodetect":
        return autodetect_reference_frame(stack, cfg, legacy_offset_bug=legacy_offset_bug)
    raise ValueError(f"Unknown reference_frame_mode: {cfg.reference_frame_mode}")


def remove_reference_frame(stack: np.ndarray, ref_result: ReferenceFrameResult) -> np.ndarray:
    """Fast removal using slicing (avoid np.delete which copies the array)."""
    idx = ref_result.index
    return np.concatenate([stack[:idx], stack[idx+1:]], axis=0)