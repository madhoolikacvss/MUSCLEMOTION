"""
io_utils.py — get any of MUSCLEMOTION's supported input types into a single
numpy array of shape (n_frames, H, W), dtype float32 (grayscale).

Supported, matching UserManual section 3 ("Input files"):
    - uncompressed AVI
    - TIFF stacks
    - PNG or TIF image sequences (a folder of individually numbered frames)
"""

from __future__ import annotations

import glob
import os

import numpy as np
import tifffile
import cv2


def _to_gray_f32(frame: np.ndarray) -> np.ndarray:
    """Collapse an (H, W, 3) BGR/RGB frame to grayscale float32; pass 2D frames through."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


def load_tiff_stack(path: str) -> np.ndarray:
    """Load a multi-page TIFF stack directly into a (n_frames, H, W) float32 array."""
    stack = tifffile.imread(path)
    if stack.ndim == 2:
        # a single-frame "stack" — treat as one frame so downstream code doesn't special-case it
        stack = stack[np.newaxis, ...]
    if stack.ndim == 4:
        # (n_frames, H, W, C) — collapse channel dim
        stack = np.stack([_to_gray_f32(f) for f in stack], axis=0)
    return stack.astype(np.float32)


def load_avi(path: str) -> np.ndarray:
    """Load every frame of an (uncompressed, per the manual) AVI into a (n_frames, H, W) array."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open AVI file: {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_to_gray_f32(frame))
    cap.release()

    if not frames:
        raise IOError(f"No frames read from AVI file: {path}")
    return np.stack(frames, axis=0)


def load_image_sequence(folder: str, pattern: str = "*") -> np.ndarray:
    """
    Load a folder of individually numbered PNG/TIF frames, sorted alphabetically
    (matching the macro's "Image Sequence..." behavior).
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    paths = [p for p in paths if p.lower().endswith((".png", ".tif", ".tiff"))]
    if not paths:
        raise IOError(f"No PNG/TIF frames found in: {folder}")

    frames = [_to_gray_f32(cv2.imread(p, cv2.IMREAD_UNCHANGED)) for p in paths]
    return np.stack(frames, axis=0)


def load_stack(path: str) -> np.ndarray:
    """
    Dispatch on file type / whether `path` is a directory, and return a
    (n_frames, H, W) float32 grayscale stack — the single entry point every
    other module and the pipeline orchestrator should call.
    """
    if os.path.isdir(path):
        return load_image_sequence(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return load_tiff_stack(path)
    if ext == ".avi":
        return load_avi(path)

    raise ValueError(f"Unsupported file type for MUSCLEMOTION input: {path}")