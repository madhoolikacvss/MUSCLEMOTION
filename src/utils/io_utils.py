"""
io_utils.py, get any of MUSCLEMOTION's supported input types into a single
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
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Ensure values are in 0-255 range
    frame = frame.astype(np.float32)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return frame


def load_avi(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open AVI file: {path}")

    frames = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        
        # # ========== DEBUG: Print raw frame info ==========
        # if frame_count <= 3:
        #     print(f"DEBUG: Raw frame {frame_count} - dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
        
        frames.append(_to_gray_f32(frame))
    cap.release()

    if not frames:
        raise IOError(f"No frames read from AVI file: {path}")
    stack = np.stack(frames, axis=0)
    
    # # ========== DEBUG: Print stack info ==========
    # print(f"DEBUG: Stack - dtype={stack.dtype}, min={stack.min()}, max={stack.max()}")
    
    return stack

def load_avi(path: str) -> np.ndarray:
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
    stack = np.stack(frames, axis=0)
    return stack

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
    (n_frames, H, W) float32 grayscale stack
    """
    if os.path.isdir(path):
        return load_image_sequence(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return load_tiff_stack(path)
    if ext == ".avi":
        return load_avi(path)

    raise ValueError(f"Unsupported file type for MUSCLEMOTION input: {path}")