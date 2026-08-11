# MUSCLEMOTION (Python port)

A modular Python reimplementation of the MUSCLEMOTION ImageJ macro
(van Meer & Sala, *Circulation Research*, 2017) for quantifying muscle/
organoid contraction from video/stack recordings.

**Goal of this codebase:** faithfully replicate MUSCLEMOTION's own
results first, with every known quirk in the original macro explicitly
documented and reproducible on demand, before any deliberate algorithmic
improvements are layered on top. Each pipeline stage lives in its own
file so it can be modified in isolation without affecting the others.


```
musclemotion/
└── src/
    ├── scripts/
    │   ├── pipeline.py
    │   └── run.py
    ├── stages/
    │   ├── baseline.py
    │   ├── masking.py
    │   ├── peaks.py
    │   ├── reference_frame.py
    │   ├── signals.py
    │   └── transients.py
    └── utils/
        ├── config.py
        ├── io_utils.py
        └── utils.py
```

---

## config.py

Holds every tunable MUSCLEMOTION parameter (acquisition settings, speed
window, masking, reference-frame-detection, and transient/peak-analysis
settings) in a single typed `MuscleMotionConfig` dataclass. Does no
computation itself , every other module reads its settings from this
object. Includes basic validation (e.g. `percentages` must be listed in
ascending order, since later stages rely on that ordering).

---

## utils.py

Small, stage-agnostic building blocks reused across multiple modules:
Gaussian blur (`blur_frame`/`blur_stack`, matching the macro's hardcoded
`sigma=10`), the shared `mean(|A-B|)` primitive nearly every stage is
built from, and a robust (median-based) baseline helper used in place of
a fragile single-frame lookup.

---

## io_utils.py

Single entry point (`load_stack`) for getting any of MUSCLEMOTION's
supported input types , TIFF stacks, uncompressed AVI, or a folder of
PNG/TIFF image-sequence frames , into one unified `(n_frames, H, W)`
float32 grayscale numpy array.

**TODOs / known limitations:**
- Does not verify that an AVI file is actually *uncompressed*, as the
  original UserManual requires for reliable results , it will load a
  compressed AVI without warning, even though compression artifacts can
  distort the resulting signal.

---

## reference_frame.py

**Step 1.** Selects the single frame every other frame gets compared
against for the Contraction signal. Three modes: `autodetect` (the
"quiet + stable" heuristic , finds a frame with low overall motion whose
motion is also not actively changing, i.e. not mid-beat), `manual`
(caller-supplied index), and `first_frame` (trivial fallback).

---

## masking.py

**Step 2.** Builds a binary "pixels of interest" mask so later stages can
restrict their averaging to pixels that actually move, rather than
diluting the signal with static background. Faithful port of the macro's
`pixelsOfInterest()`: running elementwise max of `|frame - reference|`
across a configurable frame range, thresholded at `mean + 1 std`.

**TODOs / known limitations:**
- The `mean + std` threshold is not adaptive to whether an ROI genuinely
  exists , it will always flag roughly the top 15-20% of *any* image's
  pixels as "of interest," even a fully static, non-contracting
  recording. This is reproduced (matching MUSCLEMOTION's own
  behavior) and is directly tested (`test_known_limitation_static_
  recording_still_produces_nonempty_mask`).

---

## signals.py

**Step 3.** Computes the two 1D signals everything downstream depends
on: **Contraction** (`mean(|frame_i - reference_frame|)` for every frame)
and **Speed of contraction** (`mean(|frame_i - frame_{i-speed_window}|)`,
a rolling comparison). Both are vectorized across the whole stack at
once rather than looped frame-by-frame like the macro.

**TODOs / known limitations:**
- `restrict_mean_to_mask` flag (default `False` = macro-faithful): the
  original macro multiplies the diff image by the mask but then averages
  over the *entire* frame (denominator = total pixel count), diluting the
  signal in proportion to how much of the frame is masked out as
  background. `restrict_mean_to_mask=True` computes a proper masked mean
  instead (denominator = masked-in pixel count only), likely a better
  behaved signal, kept here as an explicit opt-in rather than the default
  so a literal port can be validated first.

---

## peaks.py

**Step 4.** Finds beat/contraction peaks in the Contraction signal. Two
detectors behind one interface: `detect_peaks_legacy` (faithful port of
the macro's hand-rolled windowed local-max detector) and
`detect_peaks_scipy` (a `scipy.signal.find_peaks`-based alternative,
already available as a drop-in swap).

**TODOs / known limitations:**
- `legacy_index_bug` flag (default `False` = corrected): the macro reads
  a crude baseline estimate (`perc0`) using `referenceFrameSlice` as an
  index , but that index refers to the *original* stack's frame numbering,
  while the Contraction signal array is one frame shorter (the reference
  frame was already removed from it). This silently reads the wrong
  sample. Corrected default uses a robust median instead.
- `legacy_padding_bug` flag (default `False` = corrected): the macro
  initializes its peak list as the scalar `0`, not an empty list. When
  **zero** real peaks are found, its own "pad if fewer than 2 peaks"
  safeguard concatenates onto that stray `0`, producing a fake peak at
  index 0 even though nothing was ever detected , a real, independent
  source of false positives on flat/non-contracting recordings, confirmed
  directly in tests.
- The windowed local-max check itself is a fairly naive non-max-
  suppression; `detect_peaks_scipy` is provided as a more robust
  alternative once literal-port validation is complete.

---

## baseline.py

**Step 5.** For each detected peak, estimates the local resting level
("baseline") it rose from. Two modes matching
`cfg.high_freq_baseline_detection`: Mode A (minimum value in a pre-peak
window) and Mode B (average of the last few genuinely-flat points found
in that window).

**TODOs / known limitations:**
- `legacy_first_peak_bug` flag (Mode A only, default `False` = corrected):
  the macro has a dead-code branch (literal commented-out lines suggest
  an unfinished edit) that skips the baseline search entirely for the
  very *first* peak, leaving its baseline equal to its own peak value ,
  meaning the first beat's reported contraction amplitude is always
  exactly 0 in this mode. Confirmed directly in tests.
- `legacy_mutating_baseline_n_bug` flag (Mode B only, default `False` =
  corrected): if a peak doesn't have enough flat points, the macro
  permanently overwrites the *shared* `baseline_number_of_points` setting
  with the smaller count found , silently changing behavior for every
  later peak in the same recording, not just the current one.
- `legacy_zero_baseline_bug` flag (Mode B only, default `False` =
  corrected): if a peak has ≤1 qualifying flat point, the macro's
  baseline for that peak becomes exactly 0 (and can even divide-by-zero
  in the original macro). Corrected default falls back to the window
  minimum instead of silently reporting 0.
- `compute_speed_max_per_peak` gracefully returns zeros for recordings
  with fewer than 2 real peaks (the macro's underlying per-peak-spacing
  logic is undefined in that case) rather than crashing, which the
  single-peak-recording tests specifically guard against.

---

## transients.py

**Step 6.** Finds where the Contraction signal crosses each requested
percentage level (e.g. 10%, 50%, 90%) on the way up and back down for
every peak, then assembles the final per-beat metrics: time-to-peak,
relaxation time, transient duration, per-percentage durations,
peak-to-peak time, amplitudes, and a derived BPM estimate.

**TODOs / known limitations:**
- Only `cfg.percentages[0]` (the smallest, listed first) determines
  time-to-peak, relaxation time, and overall transient/contraction
  duration , this is inherent to how the macro works, not a bug, but it
  means the *order* of `percentages` changes what these headline metrics
  mean. Documented in `config.py` as well.
- `legacy_stale_percentage_crossing_bug` flag exists for structural
  parity with the macro's array-reuse pattern, but was proven
  mathematically **unreachable** in practice (given `percentages` must
  ascend, a successful primary-percentage crossing guarantees every
  larger percentage's crossing also succeeds within the same window) ,
  confirmed by an explicit equivalence test. Kept in the code for
  completeness, not because it changes any output.
- For a recording with exactly one real peak, there's no neighboring
  peak to size a percentage-crossing search window from. Rather than
  reproduce the macro's degenerate arithmetic for this edge case (which
  involves an internal `false`-as-zero coercion), this module falls back
  to searching (almost) the entire signal , a deliberate, documented
  choice rather than a silently-replicated edge-case bug.
- `bpm_estimate` and `n_peaks` are convenience metrics derived on top of
  MUSCLEMOTION's own output , **not** native MUSCLEMOTION outputs
  themselves. Documented clearly so they aren't mistaken for something
  the original macro reports directly.

---

## pipeline.py

The orchestrator. Chains every stage together for one well/recording
(load → reference frame → mask → signals → peaks → baseline →
transients), writes the same output files MUSCLEMOTION itself produces
(`Contraction.txt`, `Speed-of-contraction.txt`, `Overview-results.csv`,
`Log_file.txt`), and provides a batch runner across many wells with
optional multiprocessing and a plate-level summary CSV. This is
intentionally the *only* file that imports across sibling stage modules ,
every other file only depends on `config.py`/`utils.py`.

Also includes `speed_linearity_qc`, a quantitative version of the macro's
visual "calculated vs. measured speed" comparison plot (a Pearson
correlation, auto-flagged as a warning when low), and `LegacyFlags`, a
single dataclass collecting every quirk flag from every stage above ,
`LegacyFlags.all_legacy()` flips every one on at once, which is what you'd
use to validate this port against MUSCLEMOTION's own `demo_stack.tif`/
`demo_results`.

**TODOs / known limitations:**
- `run_batch`'s `n_jobs > 1` parallel path pickles each well's stack
  across process boundaries, which is only efficient when `inputs`
  contains file paths (loaded independently in each worker) rather than
  already-loaded large in-memory arrays.
