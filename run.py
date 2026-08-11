"""
Simple script to run the full MUSCLEMOTION pipeline on a single video or a folder of videos.

Usage:
    # Single video
    python run.py "C:\\path\\to\\video.avi" --output "C:\\output"
    
    # Batch folder
    python run.py --batch "C:\\path\\to\\videos" --output "C:\\output"
"""

import sys
import os
import time
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from config import MuscleMotionConfig
from pipeline import run_pipeline, LegacyFlags, save_well_outputs


def process_single_video(video_path: str, cfg: MuscleMotionConfig, base_output_dir: str, 
                         legacy: LegacyFlags = None, well_name: str = None) -> dict:
    """
    Process a single video and return timing information.
    Each video gets its own subfolder named after the video.
    """
    if well_name is None:
        well_name = Path(video_path).stem
    
    # Create a subfolder for this video
    video_output_dir = os.path.join(base_output_dir, well_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    start_time = time.time()
    # legacy_fixed = LegacyFlags()
    result = run_pipeline(
        video_path,
        cfg,
        well_name=well_name,
        legacy=legacy,
    )
    
    # Save outputs to the video-specific subfolder
    paths = save_well_outputs(result, video_output_dir)
    
    elapsed = time.time() - start_time
    
    return {
        "well_name": well_name,
        "video_path": video_path,
        "result": result,
        "paths": paths,
        "elapsed": elapsed,
        "n_frames": result.n_frames,
        "n_peaks": len(result.transient_result.beats) if result.transient_result else 0,
        "output_dir": video_output_dir,
    }


def process_batch(input_folder: str, cfg: MuscleMotionConfig, output_dir: str,
                  legacy: LegacyFlags = None, pattern: str = "*.avi", 
                  n_jobs: int = 1) -> List[dict]:
    """
    Process all videos in a folder. Each video gets its own subfolder.
    """
    # Find all video files
    video_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    
    # Also support other extensions
    for ext in [".tif", ".tiff", ".png"]:
        video_files.extend(sorted(glob.glob(os.path.join(input_folder, f"*{ext}"))))
    
    if not video_files:
        raise ValueError(f"No video files found in {input_folder}")
    
    print(f"\n{'-'*60}")
    print(f"BATCH PROCESSING: {len(video_files)} videos found")
    for f in video_files:
        print(f"  - {Path(f).name}")
    print(f"{'-'*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_start = time.time()
    
    if n_jobs <= 1:
        # Process sequentially
        for i, video_path in enumerate(video_files, 1):
            well_name = Path(video_path).stem
            print(f"\n[{i}/{len(video_files)}] Processing: {well_name}")
            
            result = process_single_video(
                video_path, cfg, output_dir, legacy, well_name
            )
            results.append(result)
            
            print(f"Completed in {result['elapsed']:.2f}s ({result['n_frames']} frames, {result['n_peaks']} peaks)")
            print(f"Output: {result['output_dir']}")
    
    else:
        # Process in parallel
        print(f"Using {n_jobs} parallel processes...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for video_path in video_files:
                well_name = Path(video_path).stem
                future = executor.submit(
                    process_single_video, 
                    video_path, 
                    cfg, 
                    output_dir, 
                    legacy,
                    well_name
                )
                futures[future] = well_name
            
            for i, future in enumerate(as_completed(futures), 1):
                well_name = futures[future]
                result = future.result()
                results.append(result)
                print(f"[{i}/{len(video_files)}] {well_name}: {result['elapsed']:.2f}s ({result['n_frames']} frames, {result['n_peaks']} peaks)")
                print(f"   Output: {result['output_dir']}")
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "-" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("-" * 60)
    print(f"Total videos processed: {len(results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time / len(results):.2f} seconds")
    print(f"Total frames processed: {sum(r['n_frames'] for r in results)}")
    print(f"Total peaks detected: {sum(r['n_peaks'] for r in results)}")
    print(f"Output directory: {output_dir}")
    
    # List all output subfolders
    print("\nOutput subfolders:")
    for r in results:
        print(f"   {r['well_name']}/")
    
    # Save batch summary
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    import csv
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'well_name', 'n_frames', 'n_peaks', 'elapsed_seconds', 
            'bpm', 'speed_linearity_correlation', 'n_warnings', 'output_folder'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'well_name': r['well_name'],
                'n_frames': r['n_frames'],
                'n_peaks': r['n_peaks'],
                'elapsed_seconds': f"{r['elapsed']:.2f}",
                'bpm': r['result'].transient_result.bpm_estimate if r['result'].transient_result else '',
                'speed_linearity_correlation': r['result'].speed_linearity_correlation,
                'n_warnings': len(r['result'].warnings),
                'output_folder': r['output_dir'],
            })
    print(f"\nBatch summary saved: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run MUSCLEMOTION pipeline on a single video or batch folder"
    )
    
    # Mutually exclusive group for single vs batch mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("video_path", type=str, nargs='?', default=None,
                       help="Path to a single video file")
    group.add_argument("--batch", "-b", type=str, 
                       help="Path to folder containing videos for batch processing")
    
    parser.add_argument("--output", "-o", type=str, default="./output", 
                       help="Output directory (default: ./output)")
    parser.add_argument("--pattern", "-p", type=str, default="*.avi",
                       help="File pattern for batch mode (default: *.avi)")
    parser.add_argument("--no-legacy", action="store_true", 
                       help="Disable legacy bugs (use corrected behavior)")
    parser.add_argument("--no-mask", action="store_true", 
                       help="Disable SNR mask (max_project=False)")
    parser.add_argument("--frame", type=int, default=None,
                       help="Manual reference frame (1-based). If not set, uses autodetect")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                       help="Number of parallel processes for batch mode (default: 1)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure the pipeline
    cfg = MuscleMotionConfig(
        recorded_framerate=26,
        speed_window=2,
        gaussian_blur=True,
        max_project=not args.no_mask,
        reference_frame_mode="manual" if args.frame else "autodetect",
        manual_reference_frame_index=(args.frame - 1) if args.frame else None,
        mp_start_range=1,
        mp_end_range=-1,
        auto_detect_start=1,
        auto_detect_stop=9999,
        low_value_n=20,
        unity_selection_n=10,
        automatic_transient_detection=True,
        peak_detection_window=20,
        peak_threshold_pct=30.0,
        baseline_threshold_pct=2.0,
        baseline_number_of_points=5,
        percentages=[10, 50, 90],
        high_freq_baseline_detection=True,
    )
    
    # Choose legacy flags
    if args.no_legacy:
        legacy = None
        print("Using CORRECTED behavior (legacy bugs DISABLED)")
    else:
        legacy = LegacyFlags.all_legacy()
        print("Using ALL legacy bugs ENABLED (matching original macro behavior)")
    
    # Run single or batch mode
    if args.batch:
        # Batch mode
        print(f"\nBatch mode: processing all videos in {args.batch}")
        print(f"File pattern: {args.pattern}")
        print(f"Parallel jobs: {args.jobs}")
        print(f"Output directory: {output_dir}")
        print(f"Each video will have its own subfolder named after the video")
        
        results = process_batch(
            args.batch,
            cfg,
            str(output_dir),
            legacy,
            args.pattern,
            args.jobs,
        )
        
        print(f"\nBatch processing complete! Results saved to: {output_dir}")
        print(f" Each video's results are in its own subfolder:")
        for r in results:
            print(f"   - {r['well_name']}/")
        
    elif args.video_path:
        # Single video mode - also creates a subfolder
        well_name = Path(args.video_path).stem
        video_output_dir = output_dir / well_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSingle video mode: {args.video_path}")
        print(f"Output directory: {video_output_dir}")
        
        if args.frame:
            print(f"Manual reference frame: {args.frame}")
        else:
            print("Reference frame: autodetect")
        
        # Process single video
        pipeline_start = time.time()
        result = run_pipeline(
            args.video_path,
            cfg,
            well_name=well_name,
            legacy=legacy,
        )
        pipeline_time = time.time() - pipeline_start
        
        # Save outputs to video-specific subfolder
        save_start = time.time()
        paths = save_well_outputs(result, str(video_output_dir))
        save_time = time.time() - save_start
        
        total_time = time.time() - pipeline_start
        
        # Print summary
        print("\n" + "-" * 60)
        print("PIPELINE COMPLETE")
        print("-" * 60)
        print(f"Well name: {result.well_name}")
        print(f"Video frames: {result.n_frames}")
        print(f"Reference frame: {result.reference_frame_result.index + 1} (1-based)")
        
        if result.transient_result:
            print(f"Number of peaks detected: {len(result.transient_result.beats)}")
            if result.transient_result.bpm_estimate:
                print(f"BPM estimate: {result.transient_result.bpm_estimate:.2f}")
        else:
            print("Number of peaks detected: 0")
        
        if result.speed_linearity_correlation is not None:
            print(f"Speed linearity correlation: {result.speed_linearity_correlation:.3f}")
        
        print(f"\nTiming:")
        print(f"Pipeline execution: {pipeline_time:.2f}s")
        print(f"File saving: {save_time:.2f}s")
        print(f"Total: {total_time:.2f}s")
        
        print(f"\nOutput folder: {video_output_dir}")
        print("\nOutput files saved:")
        for key, path in paths.items():
            if Path(path).exists():
                file_size = Path(path).stat().st_size / 1024
                print(f"  {key}: {Path(path).name} ({file_size:.1f} KB)")
        
        print(f"\nSingle video processing complete! Results saved to: {video_output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()