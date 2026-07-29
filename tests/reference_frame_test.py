"""
Quick synthetic smoke test — not a real validation against MUSCLEMOTION's
output (we'll do that against the demo_stack.tif dataset once more stages
exist), just a sanity check that the module runs and behaves sensibly:
a clearly-quiet stretch of frames should be picked over a clearly-beating
stretch.
"""

import numpy as np

from ..config import MuscleMotionConfig
from ..reference_frame import select_reference_frame, remove_reference_frame

rng = np.random.default_rng(0)

n_frames, H, W = 150, 32, 32
base = rng.normal(loc=100, scale=1.0, size=(H, W)).astype(np.float32)

stack = np.zeros((n_frames, H, W), dtype=np.float32)
# frames 0-49: quiet baseline (just sensor noise)
for i in range(50):
    stack[i] = base + rng.normal(scale=1.0, size=(H, W))
# frames 50-99: a "beat" — a blob brightens and fades
for i in range(50, 100):
    phase = np.sin((i - 50) / 50 * np.pi)  # 0 -> 1 -> 0
    blob = np.zeros((H, W), dtype=np.float32)
    blob[10:20, 10:20] = phase * 40
    stack[i] = base + blob + rng.normal(scale=1.0, size=(H, W))
# frames 100-149: quiet again
for i in range(100, 150):
    stack[i] = base + rng.normal(scale=1.0, size=(H, W))

cfg = MuscleMotionConfig(
    recorded_framerate=26,
    speed_window=2,
    gaussian_blur=False,
    reference_frame_mode="autodetect",
    auto_detect_start=1,
    auto_detect_stop=9999,   # will be clamped, exactly like your real config
    low_value_n=20,
    unity_selection_n=10,
)

result = select_reference_frame(stack, cfg)
print("Chosen reference frame index:", result.index)
print("Is it in a quiet region (expected 0-49 or 100-149)?",
      result.index < 50 or result.index >= 100)

working_stack = remove_reference_frame(stack, result)
print("Stack shape before/after removing reference frame:", stack.shape, "->", working_stack.shape)

# Also sanity-check manual and first_frame modes
cfg_manual = MuscleMotionConfig(reference_frame_mode="manual", manual_reference_frame_index=75)
r2 = select_reference_frame(stack, cfg_manual)
print("Manual mode index:", r2.index)

cfg_first = MuscleMotionConfig(reference_frame_mode="first_frame")
r3 = select_reference_frame(stack, cfg_first)
print("First-frame mode index:", r3.index)