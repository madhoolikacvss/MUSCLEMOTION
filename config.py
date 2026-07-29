"""
config.py — every tunable MUSCLEMOTION parameter in one place.

This mirrors the wizard dialog options in the original ImageJ macro (see
UserManual sections 6 A-H) plus a couple of Python-specific additions
(e.g. an explicit manual reference frame index instead of an interactive
slider). Nothing in here does any computation — it's purely a typed,
documented settings object that every other module reads from.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


ReferenceFrameMode = Literal["autodetect", "manual", "first_frame"]


@dataclass
class MuscleMotionConfig:
    # ---- acquisition ----
    recorded_framerate: float = 26.0          # frames per second
    # ms per frame, derived — do not set directly
    sampling_time_ms: float = field(init=False)

    # ---- speed / general ----
    speed_window: int = 2                     # frames; shift used for speed + ARF scan
    gaussian_blur: bool = False               # sigma=10 blur applied before every diff

    # ---- SNR / ROI masking ("F. Noise Reduction" in the wizard) ----
    max_project: bool = True                  # "Yes, but keep it simple" -> True
    mp_start_range: int = 1                   # 1-indexed, inclusive (matches macro convention)
    mp_end_range: int = -1                    # -1 == "to the end of the recording"

    # ---- reference frame selection ("G. Automatic Reference Frame Detection") ----
    reference_frame_mode: ReferenceFrameMode = "autodetect"
    manual_reference_frame_index: Optional[int] = None   # required if mode == "manual"
    auto_detect_start: int = 1                # 1-indexed, inclusive
    auto_detect_stop: int = 9999              # will be clamped to (n_frames - speed_window - 1)
    low_value_n: int = 20                     # candidates kept after the "quiet" filter
    unity_selection_n: int = 10               # candidates kept after the "flat/unity" filter

    # ---- transient / peak analysis ("H. Transient Analysis") ----
    automatic_transient_detection: bool = True
    peak_detection_window: int = 20           # frames, ~0.75 * frames-per-beat-period
    peak_threshold_pct: float = 30.0          # % of (max - perc0) required to count as a peak
    percentages: List[int] = field(default_factory=lambda: [10, 20, 30, 50, 90])
    baseline_threshold_pct: float = 2.0       # % noise band (standard baseline mode only)
    baseline_number_of_points: int = 5        # frames averaged for the standard-mode baseline
    high_freq_baseline_detection: bool = True # True -> min-value-before-peak baseline mode

    def __post_init__(self):
        if self.recorded_framerate <= 0:
            raise ValueError("recorded_framerate must be > 0")
        self.sampling_time_ms = (1.0 / self.recorded_framerate) * 1000.0

        if self.reference_frame_mode == "manual" and self.manual_reference_frame_index is None:
            raise ValueError(
                "reference_frame_mode='manual' requires manual_reference_frame_index to be set"
            )

        if self.percentages != sorted(self.percentages):
            raise ValueError(
                "percentages should be listed in ascending order — the FIRST entry is used "
                "to define time-to-peak / relaxation-time / transient-duration, matching the "
                "original macro's behavior (see transients.py docstring)."
            )
