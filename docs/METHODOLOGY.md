# Methodology

## 1. Research question

The project asks a narrow, testable question:

> On identical emergency-vehicle arrivals at a four-approach intersection, does
> a confirmed detection plus protected priority green reduce emergency-vehicle
> waiting time compared with a fixed-time signal, and what does it cost normal
> traffic?

It does **not** claim real-world safety, emissions or response-time benefits.
The controller is simulated and the evaluation is offline plus signal-level.

## 2. System under test

### 2.1 Detector

- Architecture: YOLOv8n, fine-tuned from the COCO checkpoint.
- Input: 448 × 448, batch 16, CPU, 30 epochs requested, patience 10.
- Classes: ambulance, fire_engine, police_vehicle, normal_vehicle.
- Augmentation: HSV, translation, scale, shear, horizontal flip, mosaic, mixup.
- Operating threshold in the pipeline: 0.55 for detection/display; priority
  confirmation uses a stricter 0.60.
- Evaluation: Ultralytics validation on the held-out test split at
  `conf=0.001`, `iou=0.60` for mAP, plus a separate fixed-threshold pass for
  TP/FP/FN.

The test split is disjoint at the image level: no test image or label is used
for training, early stopping, threshold tuning or lane calibration. A later
provenance audit found perceptual near-duplicates of training frames inside the
test split (consecutive video frames), so a leakage-screened re-evaluation is
reported alongside the full-split result; see
`conference_package/experiments/leakage_screened/`.

### 2.2 Tracker

The default tracker is a class-aware IoU tracker:

- match threshold 0.30;
- maximum missed frames 30;
- minimum hits 2;
- history 40 frames;
- persistent integer IDs.

ByteTrack is selectable but not used in the reported run because the optional
`lap` package could not be installed in the environment. This is stated rather
than silently substituting a different tracker.

### 2.3 Lane and approach logic

Lane assignment uses the bounding-box centre against normalised polygons. A
5-frame majority vote reduces flicker. A track outside every approach polygon
is `UNKNOWN` and cannot trigger priority. The last lane is retained only while
the vehicle is inside the central junction zone.

Approach is defined by a decreasing normalised distance to the junction centre
over an 8 s window, or by being queued within 0.42 normalised distance. This
works for all four camera directions, unlike a fixed x-direction heuristic.

### 2.4 Priority confirmation

The seven conditions are listed in `docs/ARCHITECTURE.md`. The important design
choice is that confirmation is **multi-frame and multi-signal**. A single
high-confidence detection is not enough. This reduces false activations at the
cost of a small delay (at least 3 processed frames and 0.5 s of track age).

### 2.5 Controller and baseline

The proposed controller is a guarded FSM with an all-red preparation, a
protected green, exit-based release, clearance all-red, restoration and
cooldown. The baseline is a fixed-time controller with the same green/yellow/
all-red durations and no priority.

For a given EV arrival `(lane, t_arrival)`, the baseline wait is:

```text
baseline_wait = next_fixed_green(lane, t_arrival) − t_arrival
```

The proposed wait is:

```text
proposed_wait = actual_priority_green_onset − t_arrival
```

The same arrival times are used in both. The baseline is an idealised
fixed-time plan, not a field-optimised plan.

## 3. Metrics

### 3.1 Detection

For each class and for all classes:

- precision `TP / (TP + FP)`;
- recall `TP / (TP + FN)`;
- F1 `2PR / (P + R)`;
- mAP@50 and mAP@50-95 from Ultralytics validation;
- TP/FP/FN at a fixed operating threshold with greedy IoU matching
  (`IoU ≥ 0.5`, same class).

An auxiliary image-level ROC is reported only if the test split contains both
positive and negative images. If not, ROC is marked “Not evaluated” and no AUC
is invented.

### 3.2 Waiting time and control

- emergency mean and maximum waiting time;
- normal mean and maximum waiting time;
- emergency vehicles served / total;
- throughput in vehicles per hour;
- priority activations;
- priority response time (`green − request`);
- priority duration (green held until release).

### 3.3 Runtime

- FPS (EMA of wall-clock frame rate);
- total frame latency;
- YOLO inference latency;
- tracking latency;
- end-to-end priority latency from first lane-assigned EV detection to green.

Runtime is measured on the actual CPU used for the run. It is not a hardware
benchmark and will differ on other machines.

## 4. Experimental design

### 4.1 Held-out detector evaluation

One run of `scripts/evaluate.py` on the 143-image test split. No test-time
augmentation. The evaluation writes both aggregate and per-class values, plus
the fixed-threshold FP/FN counts. The fire-engine class has only 20 test boxes,
so its per-class numbers are reported with that caveat.

### 4.2 Signal-level simulation

`simulation.py` builds a seeded scenario with Poisson normal arrivals at
420/420/380/380 vehicles per hour and five scheduled emergency vehicles. Both
controllers process the same arrival list. The simulation advances at
`dt = 0.25 s` for 600 s.

Schedule C was selected **before** the final comparison from a sweep of
candidate schedules, using the criterion “five distinct priority activations
with a non-trivial baseline wait.” It was not selected after inspecting the
final mean to maximise the reported effect. The event-level results are
reported in full, including events where priority is slower than the baseline.

A density sweep repeats the comparison at:

| Density | Arrival rates per lane per hour |
|---|---|
| Low | 180, 180, 160, 160 |
| Medium | 420, 420, 380, 380 |
| High | 900, 900, 820, 820 |

with seeds 42, 43 and 44. This gives nine runs and a mean ± sample standard
deviation. The sweep is deterministic for a given seed.

### 4.3 Live pipeline demonstration

`main.py` runs the actual detector, tracker, lane logic, priority engine and
controller on real emergency-response footage. The demo is qualitative and
provides measured FPS/latency for one session; it is not a controlled benchmark
and is not used to estimate real-world safety.

## 5. Why the event-level result matters

The simulation's proposed controller pays a fixed confirmation delay (2 s) and,
when the approach is not already safely green, an all-red preparation (1.5 s).
An EV that would have arrived on green under the baseline can therefore wait
longer under priority control. In the reported run, three events improve sub-
stantially and two already-green events incur the 3.5 s confirmation/preparation
cost. The mean still falls because the three red-arrival events dominate, but
the correct conclusion is conditional:

> Priority helps most when the EV arrives on red; it can add a small delay when
> the EV already has a green. A deployed system would need a “green-path”
> fast path that avoids unnecessary re-phasing.

This nuance is preferable to claiming a uniform improvement that the event log
does not support.

## 6. Reproducibility

- Seed: 42 for the main run; 42/43/44 for the sweep.
- Python, Torch, Ultralytics and OpenCV versions are recorded in
  `docs/ENVIRONMENT.md` and in the run summary.
- Exact commands are in `docs/EXPERIMENTS.md` and `run_all.py`.
- Every generated table is derived from result files by
  `scripts/generate_results.py`.
- The dataset audit records SHA-1 duplicates, missing labels, corrupt images,
  invalid boxes and unknown classes.

## 7. Statistical caution

The test split is small and the simulation has five scheduled EVs. Means and
standard deviations are descriptive, not inferential. No p-values are reported
because the sample is too small and the simulation is deterministic. The
arrival-rate sweep is a sensitivity analysis, not a population estimate.
