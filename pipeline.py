"""
pipeline.py — the orchestrator. Chains every stage together for one well
(load -> reference frame -> mask -> signals -> peaks -> baseline ->
transients), writes MUSCLEMOTION-equivalent output files, and provides a
simple batch runner across many wells/files.

This module is intentionally the ONLY place that imports from more than
one sibling stage module — every other file in this package only depends
on config.py/utils.py, so changing e.g. masking.py's internals never
requires touching peaks.py or baseline.py. This file is where they get
wired together, and it's the one place expected to need updating if a
stage's public interface changes.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from .config import MuscleMotionConfig
from .io_utils import load_stack
from .reference_frame import select_reference_frame, remove_reference_frame, ReferenceFrameResult
from .masking import compute_snr_mask, SNRMaskResult
from .signals import compute_signals, SignalResult
from .peaks import detect_peaks, PeakDetectionResult
from .baseline import compute_baselines, BaselineResult
from .transients import analyze_transients, beats_to_records, TransientAnalysisResult


@dataclass
class LegacyFlags:
    """
    Every documented macro quirk in one place, all defaulting to the
    corrected behavior. Use LegacyFlags.all_legacy() as a shortcut for
    "flip every one of these to True," for byte-for-byte validation
    against MUSCLEMOTION's own demo dataset.
    """
    reference_frame_offset_bug: bool = False   # reference_frame.py
    peak_index_bug: bool = False                # peaks.py (perc0 estimate)
    peak_padding_bug: bool = False                # peaks.py (zero/one-peak padding)
    baseline_first_peak_bug: bool = False          # baseline.py, Mode A only
    baseline_mutating_n_bug: bool = False           # baseline.py, Mode B only
    baseline_zero_bug: bool = False                  # baseline.py, Mode B only
    stale_percentage_crossing_bug: bool = False       # transients.py (provably inert, see transients.py)
    restrict_mean_to_mask: bool = False                 # signals.py (NOT a macro quirk — an opt-in improvement)
    peak_detection_method: str = "legacy"                # "legacy" or "scipy" (see peaks.py)

    @classmethod
    def all_legacy(cls) -> "LegacyFlags":
        return cls(
            reference_frame_offset_bug=True,
            peak_index_bug=True,
            peak_padding_bug=True,
            baseline_first_peak_bug=True,
            baseline_mutating_n_bug=True,
            baseline_zero_bug=True,
            stale_percentage_crossing_bug=True,
            restrict_mean_to_mask=False,   # not a macro behavior, always opt-in regardless
            peak_detection_method="legacy",
        )


@dataclass
class WellResult:
    well_name: str
    n_frames: int
    cfg: MuscleMotionConfig
    reference_frame_result: ReferenceFrameResult
    mask_result: Optional[SNRMaskResult]
    signals: SignalResult
    peak_result: Optional[PeakDetectionResult]
    baseline_result: Optional[BaselineResult]
    transient_result: Optional[TransientAnalysisResult]
    beat_records: Optional[List[dict]]
    speed_linearity_correlation: Optional[float]
    warnings: List[str]
    elapsed_seconds: float


def speed_linearity_qc(signals: SignalResult) -> Optional[float]:
    """
    Quantitative version of the macro's "Comparison calculated (red) and
    measured (black) speed.jpg" visual QC check: correlate the derivative
    of the Contraction signal against the actually-measured Speed signal.
    A value close to 1.0 means MUSCLEMOTION's internal consistency check
    passes (see UserManual); a low value suggests speedWindow, frame rate,
    or blur settings may need adjusting for this recording.
    """
    calculated_speed = np.abs(np.diff(signals.contraction))
    n = min(len(calculated_speed), len(signals.speed))
    if n < 2:
        return None
    a, b = calculated_speed[:n], signals.speed[:n]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_pipeline(
    stack_or_path,
    cfg: MuscleMotionConfig,
    well_name: str = "well",
    legacy: Optional[LegacyFlags] = None,
) -> WellResult:
    """
    Run every stage for ONE well/recording and return a WellResult.

    `stack_or_path` may be a path (str) — dispatched through io_utils.load_stack
    — or an already-loaded (n_frames, H, W) numpy array (handy for tests and
    for in-memory batch processing where you've loaded the stack yourself).
    """
    start = time.time()
    legacy = legacy or LegacyFlags()
    warnings: List[str] = []

    stack = load_stack(stack_or_path) if isinstance(stack_or_path, str) else np.asarray(stack_or_path)
    n_frames = stack.shape[0]

    if cfg.recorded_framerate < 50:
        warnings.append(
            f"Recorded framerate ({cfg.recorded_framerate} fps) is low — "
            "the UserManual recommends at least 60-75 fps for reliable results."
        )

    ref_result = select_reference_frame(stack, cfg, legacy_offset_bug=legacy.reference_frame_offset_bug)
    working_stack = remove_reference_frame(stack, ref_result)

    mask_result: Optional[SNRMaskResult] = None
    mask = None
    if cfg.max_project:
        mask_result = compute_snr_mask(working_stack, ref_result.frame, cfg)
        mask = mask_result.mask

    sig = compute_signals(
        working_stack, ref_result.frame, mask, cfg,
        restrict_mean_to_mask=legacy.restrict_mean_to_mask,
    )

    correlation = speed_linearity_qc(sig)
    if correlation is not None and correlation < 0.5:
        warnings.append(
            f"Speed-linearity QC correlation is low ({correlation:.2f}) — calculated and "
            "measured speed disagree; consider Gaussian blur, speed_window, or frame rate."
        )

    peak_result: Optional[PeakDetectionResult] = None
    baseline_result: Optional[BaselineResult] = None
    transient_result: Optional[TransientAnalysisResult] = None
    beat_records: Optional[List[dict]] = None

    if cfg.automatic_transient_detection:
        peak_kwargs = {"legacy_index_bug": legacy.peak_index_bug}
        if legacy.peak_detection_method == "legacy":
            peak_kwargs["legacy_padding_bug"] = legacy.peak_padding_bug

        peak_result = detect_peaks(
            sig.contraction, ref_result.index, cfg,
            method=legacy.peak_detection_method,
            **peak_kwargs,
        )
        baseline_result = compute_baselines(
            sig.contraction, peak_result.peaks, cfg,
            legacy_first_peak_bug=legacy.baseline_first_peak_bug,
            legacy_mutating_baseline_n_bug=legacy.baseline_mutating_n_bug,
            legacy_zero_baseline_bug=legacy.baseline_zero_bug,
        )
        warnings.extend(baseline_result.warnings)

        transient_result = analyze_transients(
            sig.contraction, peak_result.peaks, baseline_result.baselines, cfg,
            legacy_stale_percentage_crossing_bug=legacy.stale_percentage_crossing_bug,
        )
        beat_records = beats_to_records(transient_result, cfg)

    return WellResult(
        well_name=well_name,
        n_frames=n_frames,
        cfg=cfg,
        reference_frame_result=ref_result,
        mask_result=mask_result,
        signals=sig,
        peak_result=peak_result,
        baseline_result=baseline_result,
        transient_result=transient_result,
        beat_records=beat_records,
        speed_linearity_correlation=correlation,
        warnings=warnings,
        elapsed_seconds=time.time() - start,
    )


def save_well_outputs(result: WellResult, output_dir: str) -> dict:
    """
    Write the same set of output files MUSCLEMOTION itself produces (see
    UserManual section 7), named after `result.well_name`. Returns a dict
    of {kind: path} for whatever was actually written.
    """
    os.makedirs(output_dir, exist_ok=True)
    name = result.well_name
    paths = {}

    contraction_path = os.path.join(output_dir, f"{name}_Contraction.txt")
    with open(contraction_path, "w") as f:
        for t_ms, v in zip(result.signals.time_contraction_ms, result.signals.contraction):
            f.write(f"{t_ms}\t{v}\n")
    paths["contraction_txt"] = contraction_path

    speed_path = os.path.join(output_dir, f"{name}_Speed-of-contraction.txt")
    with open(speed_path, "w") as f:
        for t_ms, v in zip(result.signals.time_speed_ms, result.signals.speed):
            f.write(f"{t_ms}\t{v}\n")
    paths["speed_txt"] = speed_path

    if result.beat_records is not None:
        overview_path = os.path.join(output_dir, f"{name}_Overview-results.csv")
        if result.beat_records:
            fieldnames = list(result.beat_records[0].keys())
            with open(overview_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result.beat_records)
        else:
            with open(overview_path, "w") as f:
                f.write("# No peaks detected for this well.\n")
        paths["overview_csv"] = overview_path

    log_path = os.path.join(output_dir, f"{name}_Log_file.txt")
    with open(log_path, "w") as f:
        f.write(f"Well: {name}\n")
        f.write(f"n_frames: {result.n_frames}\n")
        f.write(f"reference_frame_index: {result.reference_frame_result.index} "
                f"(mode={result.reference_frame_result.mode})\n")
        f.write(f"speed_linearity_correlation: {result.speed_linearity_correlation}\n")
        if result.peak_result is not None:
            f.write(f"n_peaks_detected: {result.peak_result.diagnostics.get('n_real_peaks')}\n")
        if result.transient_result is not None and result.transient_result.bpm_estimate is not None:
            f.write(f"bpm_estimate: {result.transient_result.bpm_estimate:.2f}\n")
        f.write(f"elapsed_seconds: {result.elapsed_seconds:.2f}\n")
        f.write("\nConfig used:\n")
        f.write(json.dumps(asdict(result.cfg), indent=2, default=str))
        f.write("\n\nWarnings:\n")
        for w in result.warnings:
            f.write(f"- {w}\n")
    paths["log_txt"] = log_path

    return paths


def well_summary_row(result: WellResult) -> dict:
    """One row of plate-level summary — the natural aggregation unit across a 96-well batch."""
    n_peaks = result.peak_result.diagnostics.get("n_real_peaks") if result.peak_result else None
    bpm = result.transient_result.bpm_estimate if result.transient_result else None
    mean_amp = (
        float(np.mean([b.contraction_amplitude for b in result.transient_result.beats]))
        if result.transient_result and result.transient_result.beats
        else None
    )
    return {
        "well_name": result.well_name,
        "n_frames": result.n_frames,
        "n_peaks": n_peaks,
        "bpm_estimate": bpm,
        "mean_contraction_amplitude": mean_amp,
        "speed_linearity_correlation": result.speed_linearity_correlation,
        "n_warnings": len(result.warnings),
    }


def run_batch(
    inputs: List[tuple],
    cfg: MuscleMotionConfig,
    output_dir: str,
    legacy: Optional[LegacyFlags] = None,
    n_jobs: int = 1,
) -> List[WellResult]:
    """
    Run the pipeline across many wells and write a plate-level summary CSV.

    `inputs` is a list of (well_name, stack_or_path) tuples. Use n_jobs > 1
    to parallelize across CPU cores via ProcessPoolExecutor (only useful
    when stack_or_path values are file paths, not in-memory arrays, since
    large arrays are expensive to pickle across processes).
    """
    results: List[WellResult] = []

    if n_jobs <= 1:
        for well_name, stack_or_path in inputs:
            result = run_pipeline(stack_or_path, cfg, well_name=well_name, legacy=legacy)
            save_well_outputs(result, output_dir)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(run_pipeline, stack_or_path, cfg, well_name, legacy): well_name
                for well_name, stack_or_path in inputs
            }
            for future in as_completed(futures):
                result = future.result()
                save_well_outputs(result, output_dir)
                results.append(result)
        order = [n for n, _ in inputs]
        results.sort(key=lambda r: order.index(r.well_name))

    summary_path = os.path.join(output_dir, "plate_summary.csv")
    os.makedirs(output_dir, exist_ok=True)
    rows = [well_summary_row(r) for r in results]
    if rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return results