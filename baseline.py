"""
baseline.py, Step 5 of the pipeline: for each detected peak, estimate the
local "resting level" (baseline) it rose from, so later stages can compute
a meaningful amplitude (peak - baseline) rather than an absolute value that
drifts with the recording.

Two modes, matching cfg.high_freq_baseline_detection:

    Mode A ("Yes"/True) baseline_highfreq: just the MINIMUM signal value
        in a window right before the peak. Simple, robust when beats are
        closely spaced (little true rest between them).
    Mode B ("No"/False) baseline_standard: search for a cluster of
        genuinely FLAT points in that window (below a per-peak noise
        threshold, itself scaled by that peak's own fastest upstroke),
        then average the LAST `baseline_number_of_points` of them
        (closest in time to the peak).

Both modes rely on `compute_speed_max_per_peak`, which measures each
peak's own fastest local upstroke, used only to scale Mode B's flatness
threshold.

Three documented quirks in the original macro (each behind its own flag,
default = corrected; set True for byte-for-byte validation against
MUSCLEMOTION's own output):

1. `legacy_first_peak_bug` (Mode A only): the macro has a dead-code branch
   that SKIPS the baseline search entirely for the very
   FIRST peak, leaving its baseline equal to its own peak value. In Mode
   A, this means the first beat's reported contraction amplitude is
   always exactly 0.

2. `legacy_mutating_baseline_n_bug` (Mode B only): when a peak doesn't
   have enough flat points, the macro permanently reassigns the shared
   `baselineNumberOfPoints` setting to whatever smaller count it found,
   changing behavior for every SUBSEQUENT peak in the same
   recording, not just the current one.

3. `legacy_zero_baseline_bug` (Mode B only): if a peak has 0 or 1 flat
   points, the macro's baseline for that peak becomes exactly 0; the corrected
   default instead falls back to the minimum value in the search window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import MuscleMotionConfig
from peaks import local_spacing


def _real_peaks(peaks: List[Optional[int]]) -> List[int]:
    return [p for p in peaks if p is not None]


@dataclass
class BaselineResult:
    baselines: np.ndarray          # one value per REAL peak, same order as _real_peaks(peaks)
    speed_max_per_peak: np.ndarray  # diagnostic / used internally by Mode B
    mode: str
    warnings: List[str]


def compute_speed_max_per_peak(y_values: np.ndarray, peaks: List[Optional[int]]) -> np.ndarray:
    """
    For each real peak, find the single largest upward frame-to-frame jump
    within a window sized to 1/4 of the distance to its neighboring peak.
    Used only to scale Mode B's flatness threshold. Matches the macro's
    boundary behavior: if the window would run off either end of the
    signal, that peak's value is left at 0 rather than computed.
    """
    y = np.asarray(y_values, dtype=np.float64)
    n = len(y)
    real = _real_peaks(peaks)
    speed_max = np.zeros(len(real), dtype=np.float64)

    for j, peak in enumerate(real):
        spacing = local_spacing(peaks, j)
        window = int(spacing / 4 + 0.5) 
        # window = round(spacing / 4)
        start = peak - window
        end = peak + window
        if start > 0 and end < n and end - 1 > start:
            jumps = y[start + 1:end] - y[start:end - 1]
            speed_max[j] = float(max(0.0, jumps.max())) if len(jumps) else 0.0
        # else: leave at 0.0, matches the macro's default-array-value behavior
        # when the window would run off either end of the signal.

    return speed_max


def baseline_highfreq(
    y_values: np.ndarray,
    peaks: List[Optional[int]],
    legacy_first_peak_bug: bool = False,
) -> np.ndarray:
    """Mode A: baseline = min(y) in the window from the previous peak's midpoint to this peak."""
    y = np.asarray(y_values, dtype=np.float64)
    real = _real_peaks(peaks)
    baselines = np.zeros(len(real), dtype=np.float64)

    for c, peak in enumerate(real):
        start = 0 if c == 0 else peak - round((peak - real[c - 1]) / 2)

        if legacy_first_peak_bug and c == 0:
            # Faithful reproduction of the macro's dead-code branch: the
            # search is skipped entirely, baseline defaults to the peak's
            # own value (=> reported amplitude for beat 0 becomes exactly 0).
            baselines[c] = y[peak]
        else:
            window = y[start:peak]
            baselines[c] = float(window.min()) if len(window) else float(y[peak])

    return baselines


def baseline_standard(
    y_values: np.ndarray,
    peaks: List[Optional[int]],
    speed_max_per_peak: np.ndarray,
    cfg: MuscleMotionConfig,
    legacy_mutating_baseline_n_bug: bool = False,
    legacy_zero_baseline_bug: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Mode B: search the same pre-peak window for a cluster of genuinely
    flat points (small step-to-step change, and not sitting on top of an
    unusually high plateau), then average the last `baseline_number_of_points`
    of them.
    """
    y = np.asarray(y_values, dtype=np.float64)
    real = _real_peaks(peaks)
    overall_mean = float(y.mean())
    baselines = np.zeros(len(real), dtype=np.float64)
    warnings: List[str] = []

    # This mirrors the macro's shared, potentially-mutated setting.
    n_points_target = cfg.baseline_number_of_points

    for c, peak in enumerate(real):
        start = 0 if c == 0 else peak - round((peak - real[c - 1]) / 2)
        threshold_value = (cfg.baseline_threshold_pct / 100.0) * speed_max_per_peak[c]

        flat_points = []
        for j in range(start, peak):
            if j + 1 >= len(y):
                break
            if abs(y[j + 1] - y[j]) < threshold_value and y[j] < 1.5 * overall_mean:
                flat_points.append(y[j])

        n_found = len(flat_points)

        if n_found > n_points_target:
            used_points = flat_points[-n_points_target:]
        else:
            used_points = flat_points
            msg = (
                f"Peak {c}: only {n_found} baseline point(s) found "
                f"(wanted {n_points_target}); "
            )
            if legacy_mutating_baseline_n_bug:
                n_points_target = n_found  # mutates for ALL subsequent peaks, faithfully
                msg += f"baseline_number_of_points mutated to {n_found} for all remaining peaks."
            else:
                msg += "averaging what was found (this peak only)."
            warnings.append(msg)

        if len(used_points) > 1:
            baselines[c] = float(np.mean(used_points))
        elif legacy_zero_baseline_bug:
            baselines[c] = 0.0
            warnings.append(f"Peak {c}: <=1 flat point found; legacy behavior forces baseline=0.")
        else:
            window = y[start:peak]
            baselines[c] = float(window.min()) if len(window) else float(y[peak])
            warnings.append(
                f"Peak {c}: <=1 flat point found; falling back to window minimum as baseline."
            )

    return baselines, warnings


def compute_baselines(
    y_values: np.ndarray,
    peaks: List[Optional[int]],
    cfg: MuscleMotionConfig,
    legacy_first_peak_bug: bool = False,
    legacy_mutating_baseline_n_bug: bool = False,
    legacy_zero_baseline_bug: bool = False,
) -> BaselineResult:
    """Single entry point the pipeline orchestrator should call for Step 5."""
    speed_max = compute_speed_max_per_peak(y_values, peaks)

    if cfg.high_freq_baseline_detection:
        baselines = baseline_highfreq(y_values, peaks, legacy_first_peak_bug=legacy_first_peak_bug)
        warnings: List[str] = []
        mode = "highfreq"
    else:
        baselines, warnings = baseline_standard(
            y_values, peaks, speed_max, cfg,
            legacy_mutating_baseline_n_bug=legacy_mutating_baseline_n_bug,
            legacy_zero_baseline_bug=legacy_zero_baseline_bug,
        )
        mode = "standard"

    return BaselineResult(
        baselines=np.asarray(baselines, dtype=np.float64),
        speed_max_per_peak=speed_max,
        mode=mode,
        warnings=warnings,
    )