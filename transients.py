"""
transients.py — Step 6 of the pipeline: for each detected peak, find where
the Contraction signal crosses each requested percentage level on the way
up and back down, then assemble the final per-beat metrics (time-to-peak,
relaxation time, transient duration, per-percentage durations, peak-to-peak
time, amplitudes) — this produces the row-per-beat table that becomes
Overview-results.txt/csv.

Search window per peak
-----------------------
Each peak searches within a window sized to the distance to its
NEIGHBORING peak (next peak if one exists, otherwise the distance to the
PREVIOUS peak — there's no "next" peak to measure to for the last beat,
so falling back to the previous gap is the only sensible choice, and
turns out to be exactly what the macro effectively does across
iterations, coincidentally-but-correctly). For a recording with only ONE
real peak, there is no neighbor at all to size a window from; the macro's
literal behavior here involves arithmetic against its `false`-valued
placeholder padding entry (see peaks.py's zero/one-peak padding
handling) and produces an essentially meaningless window size. Rather
than faithfully reproduce that specific degenerate case, this module
falls back to searching (almost) the entire signal when there's only one
peak — documented here as a deliberate, honest choice rather than a
silent bug.

Percentages: which one "counts"
---------------------------------
Only the crossings for `cfg.percentages[0]` (typically 10%) determine
Time-to-peak, Relaxation Time, and overall Transient/Contraction
Duration — this is inherent to how the macro works, not a bug, but it
means changing the order of `percentages` changes what these headline
metrics mean. Documented in config.py as well.

A quirk in the macro that looks worse on paper than it is in practice:
`legacy_stale_percentage_crossing_bug`
------------------------------------------------------------------------
For the OTHER percentage levels (used only for the per-percentage
duration columns, e.g. "90-to-90 transient (ms)"), the macro reuses the
same small arrays across ALL peaks without resetting them between beats.
Read literally, this looks like it could let a "not found this peak"
percentage silently reuse a stale index left over from a PREVIOUS peak.

However — given `percentages` must be ascending (enforced in config.py)
and the down/up search window is identical for every percentage of the
SAME peak, this turns out to be UNREACHABLE in practice: if the primary
(smallest, strictest) percentage's crossing is found, every larger
(looser) percentage's crossing is mathematically guaranteed to also be
found within that same window — a point satisfying the stricter
threshold automatically satisfies every looser one too, so the scan for
any larger percentage cannot fail once the primary one has succeeded.
`legacy_stale_percentage_crossing_bug=True` reproduces the macro's literal
array-reuse structure anyway (for code-structure parity), but it should
always produce IDENTICAL output to the corrected default — see
test_transients.py's equivalence test, which verifies exactly that rather
than asserting a divergence that can't actually occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import MuscleMotionConfig


def _real_peaks(peaks: List[Optional[int]]) -> List[int]:
    return [p for p in peaks if p is not None]


@dataclass
class BeatMetrics:
    peak_index: int
    time_to_peak_ms: Optional[float]
    relaxation_time_ms: Optional[float]
    transient_duration_ms: Optional[float]   # macro's "Contraction duration [X% above baseline] (ms)"
    percentage_durations_ms: Dict[int, float]  # one entry per cfg.percentages value
    peak_to_peak_time_ms: Optional[float]       # None for the first beat
    baseline_value: float
    peak_amplitude: float
    contraction_amplitude: float


@dataclass
class TransientAnalysisResult:
    beats: List[BeatMetrics]
    n_peaks: int
    bpm_estimate: Optional[float]   # derived convenience metric, NOT a native MUSCLEMOTION output


def _peak_to_peak_distance_for_search(real_peaks: List[int], c: int) -> Optional[int]:
    """
    Distance to size the percentage-crossing search window around peak c.
    Returns None if there's no usable neighbor at all (only one real peak).
    """
    n = len(real_peaks)
    if n < 2:
        return None
    if c < n - 1:
        return abs(real_peaks[c + 1] - real_peaks[c])
    return abs(real_peaks[c] - real_peaks[c - 1])


def _find_crossings_for_peak(
    y: np.ndarray,
    peak: int,
    baseline: float,
    half_width: int,
    percentages: List[int],
) -> tuple[Dict[int, Optional[int]], Dict[int, Optional[int]]]:
    """
    For one peak, find the down-flank (rising, before the peak) and
    up-flank (falling, after the peak) crossing index for every requested
    percentage level. Requires 3 consecutive points below the level to
    reject single-frame noise dips as a false crossing.
    """
    n = len(y)
    perc100 = y[peak]
    perc0 = baseline

    min_border = max(2, peak - half_width)
    max_border = min(n - 3, peak + half_width)

    down: Dict[int, Optional[int]] = {}
    up: Dict[int, Optional[int]] = {}

    for p in percentages:
        level = (p / 100.0) * (perc100 - perc0) + perc0

        found_down = None
        l = peak
        while l > min_border:
            if y[l] < level and y[l - 1] < level and y[l - 2] < level:
                found_down = l
                break
            l -= 1
        down[p] = found_down

        found_up = None
        l = peak
        while l < max_border:
            if y[l] < level and y[l + 1] < level and y[l + 2] < level:
                found_up = l
                break
            l += 1
        up[p] = found_up

    return down, up


def analyze_transients(
    y_values: np.ndarray,
    peaks: List[Optional[int]],
    baselines: np.ndarray,
    cfg: MuscleMotionConfig,
    legacy_stale_percentage_crossing_bug: bool = False,
) -> TransientAnalysisResult:
    """Single entry point the pipeline orchestrator should call for Step 6."""
    y = np.asarray(y_values, dtype=np.float64)
    real = _real_peaks(peaks)
    percentages = cfg.percentages
    first_pct = percentages[0]

    # Mirrors the macro's shared, non-reset-between-peaks arrays — only
    # populated/used when legacy_stale_percentage_crossing_bug=True.
    persistent_down: Dict[int, int] = {p: 0 for p in percentages}
    persistent_up: Dict[int, int] = {p: 0 for p in percentages}

    beats: List[BeatMetrics] = []

    for c, peak in enumerate(real):
        dist = _peak_to_peak_distance_for_search(real, c)
        half_width = len(y) if dist is None else dist  # see module docstring re: single-peak fallback

        baseline = float(baselines[c])
        down, up = _find_crossings_for_peak(y, peak, baseline, half_width, percentages)

        low_down = down[first_pct]
        low_up = up[first_pct]

        if low_down is None:
            time_to_peak = None
            transient_duration = None
            relaxation_time = None if low_up is None else abs((peak - low_up) * cfg.sampling_time_ms)
        else:
            time_to_peak = abs((peak - low_down) * cfg.sampling_time_ms)
            if low_up is None:
                relaxation_time = None
                transient_duration = None
            else:
                relaxation_time = abs((peak - low_up) * cfg.sampling_time_ms)
                transient_duration = abs((low_up - low_down) * cfg.sampling_time_ms)

        percentage_durations: Dict[int, float] = {}
        if legacy_stale_percentage_crossing_bug:
            for p in percentages:
                if down[p] is not None:
                    persistent_down[p] = down[p]
                if up[p] is not None:
                    persistent_up[p] = up[p]
            for p in percentages:
                if transient_duration is not None:
                    percentage_durations[p] = (persistent_up[p] - persistent_down[p]) * cfg.sampling_time_ms
                else:
                    percentage_durations[p] = 0.0
        else:
            for p in percentages:
                if transient_duration is not None and down[p] is not None and up[p] is not None:
                    percentage_durations[p] = (up[p] - down[p]) * cfg.sampling_time_ms
                else:
                    percentage_durations[p] = 0.0

        peak_to_peak_time = None
        if c > 0:
            peak_to_peak_time = (peak - real[c - 1]) * cfg.sampling_time_ms

        beats.append(BeatMetrics(
            peak_index=peak,
            time_to_peak_ms=time_to_peak,
            relaxation_time_ms=relaxation_time,
            transient_duration_ms=transient_duration,
            percentage_durations_ms=percentage_durations,
            peak_to_peak_time_ms=peak_to_peak_time,
            baseline_value=baseline,
            peak_amplitude=float(y[peak]),
            contraction_amplitude=float(y[peak]) - baseline,
        ))

    ppt_values = [b.peak_to_peak_time_ms for b in beats if b.peak_to_peak_time_ms is not None]
    bpm = 60000.0 / np.mean(ppt_values) if ppt_values else None

    return TransientAnalysisResult(beats=beats, n_peaks=len(beats), bpm_estimate=bpm)


def beats_to_records(result: TransientAnalysisResult, cfg: MuscleMotionConfig) -> List[dict]:
    """
    Convert a TransientAnalysisResult into a list of plain dicts with the
    SAME column naming convention as MUSCLEMOTION's Overview-results.txt,
    ready to hand to pandas.DataFrame(records) or csv.DictWriter.
    """
    records = []
    for b in result.beats:
        record = {
            "Contraction duration [10% above baseline] (ms)": b.transient_duration_ms,
            "Time-to-peak (ms)": b.time_to_peak_ms,
            "Relaxation Time (ms)": b.relaxation_time_ms,
            "Peak-to-peak time (ms)": b.peak_to_peak_time_ms,
            "Baseline value (a.u.)": b.baseline_value,
            "Peak amplitude (a.u.)": b.peak_amplitude,
            "Contraction amplitude (a.u.)": b.contraction_amplitude,
        }
        for p in cfg.percentages:
            record[f"{100 - p}-to-{100 - p} transient (ms)"] = b.percentage_durations_ms[p]
        records.append(record)
    return records