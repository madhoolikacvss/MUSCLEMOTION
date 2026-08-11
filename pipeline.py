"""
pipeline.py is the orchestrator. Chains every stage together for one well
(load -> reference frame -> mask -> signals -> peaks -> baseline ->
transients), writes MUSCLEMOTION-equivalent output files, and provides a
simple batch runner across many wells/files.

This module is intentionally the ONLY place that imports from more than
one sibling stage module, every other file in this package only depends
on config.py/utils.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from config import MuscleMotionConfig
from io_utils import load_stack
from reference_frame import select_reference_frame, remove_reference_frame, ReferenceFrameResult
from masking import compute_snr_mask, SNRMaskResult
from signals import compute_signals, SignalResult
from peaks import detect_peaks, PeakDetectionResult
from baseline import compute_baselines, BaselineResult
from transients import analyze_transients, beats_to_records, TransientAnalysisResult


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
    height: int
    width: int
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
    input_path: str = ""
    batch_mode: bool = False
    # Timing breakdown
    timing: Optional[dict] = None


def speed_linearity_qc(signals: SignalResult) -> Optional[float]:
    """
    Equivalent for "Comparison calculated (red) and measured (black) speed.jpg" 
    visual QC check: correlate the derivative of the Contraction signal against 
    the actually-measured Speed signal.
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


def _write_log_file(result: WellResult, output_dir: str) -> str:
    """
    Write log file matching the macro's format.
    """
    log_path = os.path.join(output_dir, f"{result.well_name}_Log_file.txt")
    
    with open(log_path, "w") as f:
        # Header
        now = datetime.now()
        f.write("Log started\n")
        f.write(f"Date: {now.day}-{now.month}-{now.year}\n")
        f.write(f"Time: {now.hour:02d}:{now.minute:02d}:{now.second:02d}\n")
        f.write("***\n")
        
        # Version info
        f.write("Algorithm tool version number: 1.0\n")
        
        # Core parameters
        cfg = result.cfg
        f.write(f"recordedFramerate: {cfg.recorded_framerate}\n")
        f.write(f"speedWindow: {cfg.speed_window}\n")
        f.write(f"referenceFrameSlice: 1\n")
        f.write(f"maxProject: {1 if cfg.max_project else 0}\n")
        f.write(f"MPstartRange: {cfg.mp_start_range}\n")
        f.write(f"MPendRange: {cfg.mp_end_range}\n")
        f.write(f"hideIntermediateResults: 1\n")
        f.write(f"checkClip: 1\n")
        f.write(f"checkSpeedlinearity: 1\n")
        
        # ARF parameters
        is_autodetect = cfg.reference_frame_mode == "autodetect"
        f.write(f"autodetectReferenceFrame: {1 if is_autodetect else 0}\n")
        if is_autodetect:
            f.write("***autodetectReferenceFrame parameters\n")
            f.write(f"*lowValueN: {cfg.low_value_n}\n")
            f.write(f"*unitySelectionN: {cfg.unity_selection_n}\n")
            f.write(f"*autoDetectStart: {cfg.auto_detect_start}\n")
            f.write(f"*autoDetectStop: {cfg.auto_detect_stop}\n")
            f.write("***\n")
        
        # Manual reference frame note
        f.write(f"manualReferenceFrame: {1 if cfg.reference_frame_mode == 'manual' else 0}\tNOTE:overruled if autodetectReferenceFrame is true\n")
        
        # Transient detection parameters
        f.write(f"automaticTransientDetection: {1 if cfg.automatic_transient_detection else 0}\n")
        if cfg.automatic_transient_detection:
            f.write("***automaticTransientDetection parameters\n")
            f.write(f"*PeakDetectionWindow: {cfg.peak_detection_window}\n")
            f.write(f"*peakThreshold: {cfg.peak_threshold_pct}\n")
            f.write(f"*drawPeaks: 1\n")
            f.write(f"*percentages: \n")
            f.write(", ".join(str(p) for p in cfg.percentages) + "\n")
            f.write(f"*baselineThreshold: {cfg.baseline_threshold_pct}\n")
            f.write(f"*baselineNumberOfPoints: {cfg.baseline_number_of_points}\n")
            f.write(f"*highFreqBaselineDetection: {1 if cfg.high_freq_baseline_detection else 0}\n")
            f.write(f"*guassianBlur10: {'Yes' if cfg.gaussian_blur else 'No'}\n")
            f.write(f"*tiffImSequence: No\n")
            f.write("***\n")
        
        # Batch info
        f.write(f"batchDirLoad: {1 if result.batch_mode else 0}\n")
        f.write(f"batchDirLoadVideo: {1 if result.batch_mode else 0}\n")
        if result.input_path:
            f.write(f"BatchDirVideo path={os.path.basename(result.input_path)}\n")
        
        f.write("\n")
        
        # Evaluation header
        f.write(f"----------------- Evaluating file:MUSCLEMOTION v1.0 -----------------\n")
        
        # Stack dimensions - NOW FROM THE ACTUAL STACK!
        f.write(f"Width: {result.width}\n")
        f.write(f"Height: {result.height}\n")
        f.write("Channels: 1\n")  # MUSCLEMOTION always works with grayscale
        f.write(f"Slices: {result.n_frames}\n")
        f.write("Frames: 1\n")  # In the macro, for a stack, frames is always 1
        
        # Warnings (macro-style)
        if cfg.recorded_framerate < 50:
            f.write("WARNING: Recorded framerate is low\n")
        
        # ARF warnings
        if cfg.auto_detect_stop >= result.n_frames - cfg.speed_window - 1:
            clamped_stop = result.n_frames - cfg.speed_window - 1
            f.write(f"WARNING: autoDetectStop set to {clamped_stop} since it should be smaller than stack number ({result.n_frames}) minus speedWindow ({cfg.speed_window}) minus 1 (Reference frame).\n")
        
        # Reference frame info
        ref_frame_1based = result.reference_frame_result.index + 1
        # Check if autodetect or manual
        if cfg.reference_frame_mode == "autodetect":
            f.write(f"Automatic detected reference frame: frame {ref_frame_1based}\n")
        else:
            f.write(f"Manual selected reference frame: frame {ref_frame_1based}\n")
        
        # Peaks detected
        if result.transient_result and result.transient_result.beats:
            peak_frames = [b.peak_index + 1 for b in result.transient_result.beats]
            f.write("Peaks detected at points (frames):\n")
            f.write(", ".join(str(p) for p in peak_frames) + "\n")
        else:
            f.write("Peaks detected at points (frames):\n")
            f.write("No peaks detected\n")
        
        # Saved plot values (macro-style)
        f.write(f"Saved plot values: {os.path.join(output_dir, result.well_name + '_Contraction.txt')}\n")
        f.write(f"Saved plot values: {os.path.join(output_dir, result.well_name + '_Speed-of-contraction.txt')}\n")
        
        # Elapsed time
        f.write(f"Elapsed time (s): {result.elapsed_seconds:.2f}\n")
        
        # Timing breakdown
        if result.timing:
            f.write("\n--- Timing Breakdown ---\n")
            for stage, duration in result.timing.items():
                f.write(f"{stage}: {duration:.2f}s\n")
    
    return log_path 


def run_pipeline(
    stack_or_path,
    cfg: MuscleMotionConfig,
    well_name: str = "well",
    legacy: Optional[LegacyFlags] = None,
    batch_mode: bool = False,
) -> WellResult:
    """
    Run every stage for ONE well/recording and return a WellResult.
    """
    total_start = time.time()
    timing = {}
    legacy = legacy or LegacyFlags()
    warnings: List[str] = []

    # ========================================================================
    # Stage 1: Load Stack
    # ========================================================================
    stage_start = time.time()
    input_path = stack_or_path if isinstance(stack_or_path, str) else ""
    stack = load_stack(stack_or_path) if isinstance(stack_or_path, str) else np.asarray(stack_or_path)
    n_frames, height, width = stack.shape
    timing["load_stack"] = time.time() - stage_start

    if cfg.recorded_framerate < 50:
        warnings.append(
            f"Recorded framerate ({cfg.recorded_framerate} fps) is low — "
            "the UserManual recommends at least 60-75 fps for reliable results."
        )

    # ========================================================================
    # Stage 2: Reference Frame Selection
    # ========================================================================
    stage_start = time.time()
    ref_result = select_reference_frame(stack, cfg, legacy_offset_bug=legacy.reference_frame_offset_bug)
    timing["reference_frame"] = time.time() - stage_start

    # ========================================================================
    # Stage 3: Remove Reference Frame
    # ========================================================================
    stage_start = time.time()
    working_stack = remove_reference_frame(stack, ref_result)
    timing["remove_reference"] = time.time() - stage_start

    # ========================================================================
    # Stage 4: Masking (SNR Improvement)
    # ========================================================================
    stage_start = time.time()
    mask_result: Optional[SNRMaskResult] = None
    mask = None
    if cfg.max_project:
        mask_result = compute_snr_mask(working_stack, ref_result.frame, cfg)
        mask = mask_result.mask
    timing["masking"] = time.time() - stage_start

    # ========================================================================
    # Stage 5: Signals (Contraction + Speed)
    # ========================================================================
    stage_start = time.time()
    sig = compute_signals(
        working_stack, ref_result.frame, mask, cfg,
        restrict_mean_to_mask=legacy.restrict_mean_to_mask,
    )
    timing["signals"] = time.time() - stage_start

    # ========================================================================
    # Stage 6: Speed Linearity QC
    # ========================================================================
    stage_start = time.time()
    correlation = speed_linearity_qc(sig)
    if correlation is not None and correlation < 0.5:
        warnings.append(
            f"Speed-linearity QC correlation is low ({correlation:.2f}), calculated and "
            "measured speed disagree; consider Gaussian blur, speed_window, or frame rate."
        )
    timing["speed_linearity"] = time.time() - stage_start

    # ========================================================================
    # Stage 7: Peak Detection, Baseline, Transients
    # ========================================================================
    peak_result: Optional[PeakDetectionResult] = None
    baseline_result: Optional[BaselineResult] = None
    transient_result: Optional[TransientAnalysisResult] = None
    beat_records: Optional[List[dict]] = None

    if cfg.automatic_transient_detection:
        # Peak Detection
        stage_start = time.time()
        peak_kwargs = {"legacy_index_bug": legacy.peak_index_bug}
        if legacy.peak_detection_method == "legacy":
            peak_kwargs["legacy_padding_bug"] = legacy.peak_padding_bug

        peak_result = detect_peaks(
            sig.contraction, ref_result.index, cfg,
            method=legacy.peak_detection_method,
            **peak_kwargs,
        )
        timing["peak_detection"] = time.time() - stage_start

        # Baseline Computation
        stage_start = time.time()
        baseline_result = compute_baselines(
            sig.contraction, peak_result.peaks, cfg,
            legacy_first_peak_bug=legacy.baseline_first_peak_bug,
            legacy_mutating_baseline_n_bug=legacy.baseline_mutating_n_bug,
            legacy_zero_baseline_bug=legacy.baseline_zero_bug,
        )
        warnings.extend(baseline_result.warnings)
        timing["baseline"] = time.time() - stage_start

        # Transient Analysis
        stage_start = time.time()
        transient_result = analyze_transients(
            sig.contraction, peak_result.peaks, baseline_result.baselines, cfg,
            legacy_stale_percentage_crossing_bug=legacy.stale_percentage_crossing_bug,
        )
        beat_records = beats_to_records(transient_result, cfg)
        timing["transients"] = time.time() - stage_start

    return WellResult(
        well_name=well_name,
        n_frames=n_frames,
        height=height,
        width=width,
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
        elapsed_seconds=time.time() - total_start,
        input_path=input_path,
        batch_mode=batch_mode,
        timing=timing,
    )


def _save_mask_image(result: WellResult, output_dir: str) -> Optional[str]:
    """
    Save the binary mask (pixels of interest) as a PNG image.
    Returns the path to the saved image, or None if no mask exists.
    """
    if result.mask_result is None or result.mask_result.mask is None:
        return None
    
    mask = result.mask_result.mask
    name = result.well_name
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    ax.set_title(f"Pixels of Interest Mask - {name}\n{np.sum(mask)} pixels ({100 * np.sum(mask) / mask.size:.2f}%)")
    ax.axis("off")
    fig.tight_layout()
    
    path = os.path.join(output_dir, f"{name}_mask.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return path


def _generate_plots(result: WellResult, output_dir: str) -> dict:
    """
    Generate and save all plots matching the macro's output.
    Called automatically by save_well_outputs().
    """
    os.makedirs(output_dir, exist_ok=True)
    name = result.well_name
    paths = {}
    
    # Access cfg from result
    cfg = result.cfg
    
    # 1. Contraction Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.signals.time_contraction_ms, result.signals.contraction, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Contraction (a.u.)')
    ax.set_title(f'Contraction Profile - {name}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{name}_Contraction_profile.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    paths["contraction_plot"] = path
    
    # 2. Speed of Contraction Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.signals.time_speed_ms, result.signals.speed, 'r-', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Speed of Contraction (a.u.)')
    ax.set_title(f'Speed of Contraction Profile - {name}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{name}_Speed_of_contraction_profile.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    paths["speed_plot"] = path
    
    # 3. Speed Linearity Comparison (if correlation available)
    if result.speed_linearity_correlation is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        measured_speed = result.signals.speed
        calculated_speed = np.abs(np.diff(result.signals.contraction))
        n = min(len(calculated_speed), len(measured_speed))
        
        if n > 1:
            calc_min = calculated_speed[:n].min()
            calc_max = calculated_speed[:n].max()
            meas_min = measured_speed[:n].min()
            meas_max = measured_speed[:n].max()
            
            calc_norm = (calculated_speed[:n] - calc_min) / (calc_max - calc_min + 1e-10)
            meas_norm = (measured_speed[:n] - meas_min) / (meas_max - meas_min + 1e-10)
            time_norm = result.signals.time_speed_ms[:n]
            
            ax.plot(time_norm, meas_norm, 'k-', linewidth=1.5, label='Measured speed')
            ax.plot(time_norm, calc_norm, 'r-', linewidth=1.5, label='Calculated from contraction')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Normalized speed (a.u.)')
            ax.set_title(f'Speed Linearity Check - {name}\nCorrelation: {result.speed_linearity_correlation:.3f}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, f"{name}_Speed_linearity_comparison.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            paths["linearity_plot"] = path
    
    # 4. Transient Analysis Plot (if peaks detected)
    if result.transient_result and result.transient_result.beats:
        fig, ax = plt.subplots(figsize=(14, 8))
        y_values = result.signals.contraction
        time_ms = result.signals.time_contraction_ms
        
        ax.plot(time_ms, y_values, 'b-', linewidth=1.5, label='Contraction')
        
        for i, beat in enumerate(result.transient_result.beats):
            peak = beat.peak_index
            baseline = beat.baseline_value
            peak_value = beat.peak_amplitude
            
            ax.plot(time_ms[peak], peak_value, 'ro', markersize=8, 
                   label='Peak' if i == 0 else '')
            ax.plot(time_ms[peak], baseline, 'go', markersize=6, 
                   label='Baseline' if i == 0 else '')
            ax.plot([time_ms[peak], time_ms[peak]], [baseline, peak_value], 
                   'g--', linewidth=1)
            ax.annotate(f'#{i+1}', (time_ms[peak], peak_value), 
                       textcoords="offset points", xytext=(5, 10), 
                       ha='left', fontsize=8)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Contraction (a.u.)')
        bpm_str = f", BPM: {result.transient_result.bpm_estimate:.1f}" if result.transient_result.bpm_estimate else ""
        ax.set_title(f'Transient Analysis - {name}{bpm_str}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_Transient_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths["transient_plot"] = path
    
    return paths


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

    # Generate and save plots (automatically)
    plot_paths = _generate_plots(result, output_dir)
    paths.update(plot_paths)

    if result.mask_result is not None:
        mask_path = _save_mask_image(result, output_dir)
        if mask_path:
            paths["mask_image"] = mask_path
    
    # Generate log file (matching macro format)
    log_path = _write_log_file(result, output_dir)
    paths["log_txt"] = log_path

    return paths


def well_summary_row(result: WellResult) -> dict:
    """One row of plate-level summary CSV for a single well."""
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
            result = run_pipeline(stack_or_path, cfg, well_name=well_name, legacy=legacy, batch_mode=True)
            save_well_outputs(result, output_dir)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(run_pipeline, stack_or_path, cfg, well_name, legacy, True): well_name
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