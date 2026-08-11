"""
peaks.py, Step 4 of the pipeline: find beat/contraction peaks in the
Contraction signal.

Two detectors are provided behind one interface

    detect_peaks_legacy: a faithful port of the macro's hand-rolled
                            windowed-local-max detector, INCLUDING two
                            bugs in the original code (see below).
    detect_peaks_scipy: a much more robust alternative built on
                            scipy.signal.find_peaks, recommended once
                            you're past validating a literal port.

1. INDEX-OFFSET BUG in the crude baseline estimate (`perc0`):
   The macro computes `perc0 = yValues[referenceFrameSlice]`, but
   `referenceFrameSlice` is a frame number from the ORIGINAL stack
   (before the reference frame was deleted), while `yValues` (the
   Contraction signal) is one frame SHORTER because that frame was
   removed. 

2. ZERO-PEAK PADDING BUG:
   The macro initializes its peak list as the SCALAR `0` (not an empty
   array). If zero real peaks are found, its "pad if fewer than 2 peaks"
   safeguard (`Array.concat(maxList, false)`) concatenates onto that
   stray scalar `0`, producing a fake peak at index 0 even though NO
   real peak was ever detected. A genuinely flat, non-contracting
   recording can therefore still emit one bogus "peak" purely from this
   padding logic, independent of any threshold or masking issue.
   `legacy_padding_bug=True` reproduces this; the default (`False`)
   only pads when exactly ONE real peak was found (so downstream
   pairwise math doesn't crash), and correctly returns an EMPTY peak
   list when zero peaks were found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

from config import MuscleMotionConfig
from utils import robust_baseline_value


@dataclass
class PeakDetectionResult:
    peaks: List[Optional[int]]      # frame indices into y_values; may contain a trailing None (padding)
    perc0: float                     # baseline estimate used for the amplitude threshold
    perc100: float                    # max value in the signal
    threshold_value: float             # absolute amplitude cutoff = (peak_threshold_pct/100)*(perc100-perc0)
    window_used: int                    # PeakDetectionWindow after the even-forcing adjustment
    method: str
    diagnostics: dict


def _resolve_perc0(y_values: np.ndarray, reference_frame_index: int, legacy_index_bug: bool) -> float:
    if legacy_index_bug:
        idx = int(np.clip(reference_frame_index, 0, len(y_values) - 1))
        return float(y_values[idx])
    return robust_baseline_value(y_values)


def _real_peaks(peaks: List[Optional[int]]) -> List[int]:
    """Convenience: strip out None padding entries, e.g. for counting/plotting."""
    return [p for p in peaks if p is not None]


def local_spacing(peaks: List[Optional[int]], idx: int) -> int:
    """
    Distance(in frames) to the neighboring peak: the NEXT peak's distance
    if one exists, otherwise the distance from the PREVIOUS peak (matches
    the macro's rangeSpeedMax / peak-spacing logic, reused later by
    baseline.py). Assumes peaks[idx] is a real (non-None) peak.
    """
    real = _real_peaks(peaks)
    if len(real) < 2:
        raise ValueError("Need at least 2 real peaks to compute local spacing.")
    pos = real.index(peaks[idx]) if peaks[idx] in real else None
    if pos is None:
        raise ValueError(f"peaks[{idx}] is not a real peak (None/padding).")
    if pos == len(real) - 1:
        return abs(real[pos] - real[pos - 1])
    return abs(real[pos + 1] - real[pos])


def detect_peaks_legacy(
    y_values: np.ndarray,
    reference_frame_index: int,
    cfg: MuscleMotionConfig,
    legacy_index_bug: bool = False,
    legacy_padding_bug: bool = False,
) -> PeakDetectionResult:
    """
    Faithful port of the macro's windowed local-max peak detector.

    For each candidate frame u (with enough room on both sides for a full
    window): accept it as a peak only if (a) it clears an absolute
    amplitude threshold above perc0, AND (b) no neighbor within
    +/- (window/2 - 1) frames is taller.
    """
    y = np.asarray(y_values, dtype=np.float64)
    n = len(y)

    perc100 = float(y.max())
    perc0 = _resolve_perc0(y, reference_frame_index, legacy_index_bug)
    threshold_value = (cfg.peak_threshold_pct / 100.0) * (perc100 - perc0)

    W = cfg.peak_detection_window
    if W % 2 != 0:
        W += 1
    half = W // 2

    peaks: List[int] = []
    for u in range(half, n - 1 - half):
        if (y[u] - perc0) <= threshold_value:
            continue
        is_local_max = True
        for r in range(1, half):
            if y[u - r] > y[u] or y[u + r] > y[u]:
                is_local_max = False
                break
        if is_local_max:
            peaks.append(u)

    final_peaks: List[Optional[int]] = list(peaks)

    if legacy_padding_bug:
        # Faithful reproduction of the scalar-0-initialized-list bug: even
        # zero real peaks gets a fake entry at index 0.
        if len(final_peaks) < 2:
            final_peaks = ([0] if len(final_peaks) == 0 else final_peaks) + [None]
    else:
        # Corrected: only pad when exactly one real peak was found (so
        # pairwise math downstream has something to compare against);
        # zero real peaks stays an honest empty list.
        if len(final_peaks) == 1:
            final_peaks = final_peaks + [None]

    return PeakDetectionResult(
        peaks=final_peaks,
        perc0=perc0,
        perc100=perc100,
        threshold_value=threshold_value,
        window_used=W,
        method="legacy",
        diagnostics={
            "legacy_index_bug": legacy_index_bug,
            "legacy_padding_bug": legacy_padding_bug,
            "n_real_peaks": len(_real_peaks(final_peaks)),
        },
    )


def detect_peaks_scipy(
    y_values: np.ndarray,
    reference_frame_index: int,
    cfg: MuscleMotionConfig,
    legacy_index_bug: bool = False,
) -> PeakDetectionResult:
    """
    Alternative: scipy.signal.find_peaks with an amplitude
    floor (height) derived the same way as the legacy threshold, and a
    minimum spacing (distance) derived from PeakDetectionWindow, same
    conceptual inputs, much more robust non-max-suppression than the
    hand-rolled window check in detect_peaks_legacy.
    """
    y = np.asarray(y_values, dtype=np.float64)

    perc100 = float(y.max())
    perc0 = _resolve_perc0(y, reference_frame_index, legacy_index_bug)
    threshold_value = (cfg.peak_threshold_pct / 100.0) * (perc100 - perc0)

    W = cfg.peak_detection_window
    if W % 2 != 0:
        W += 1

    indices, properties = find_peaks(
        y,
        height=perc0 + threshold_value,
        distance=max(1, W),
    )
    final_peaks: List[Optional[int]] = list(int(i) for i in indices)
    if len(final_peaks) == 1:
        final_peaks = final_peaks + [None]

    return PeakDetectionResult(
        peaks=final_peaks,
        perc0=perc0,
        perc100=perc100,
        threshold_value=threshold_value,
        window_used=W,
        method="scipy",
        diagnostics={
            "legacy_index_bug": legacy_index_bug,
            "n_real_peaks": len(_real_peaks(final_peaks)),
            "scipy_properties": properties,
        },
    )


def detect_peaks(
    y_values: np.ndarray,
    reference_frame_index: int,
    cfg: MuscleMotionConfig,
    method: str = "legacy",
    **kwargs,
) -> PeakDetectionResult:
    """Single entry point the pipeline orchestrator should call for Step 4."""
    if method == "legacy":
        return detect_peaks_legacy(y_values, reference_frame_index, cfg, **kwargs)
    if method == "scipy":
        return detect_peaks_scipy(y_values, reference_frame_index, cfg, **kwargs)
    raise ValueError(f"Unknown peak detection method: {method}")