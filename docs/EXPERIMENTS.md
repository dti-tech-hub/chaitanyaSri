# Experiments and results

All values in this document come from generated artifacts under `results/`.
If a run has not happened, the entry says **Not evaluated** instead of showing a
placeholder number.

## 0. Environment

| Item | Value |
|---|---|
| OS | Windows (local workstation) |
| Python | 3.14.5 |
| Torch | 2.14.1+cpu |
| Ultralytics | 8.4.174 |
| OpenCV | 5.0.0 |
| Device | CPU only |
| Seed | 42 (main), 42/43/44 (sweep) |

## 1. Dataset audit

Command:

```powershell
python scripts/validate_dataset.py
```

Result file: `results/metrics/dataset_validation.json`

| Split | Images | Boxes | Ambulance | Fire engine | Police | Normal |
|---|---:|---:|---:|---:|---:|---:|
| Train | 998 | 1,719 | 563 | 132 | 344 | 680 |
| Val | 285 | 501 | 148 | 35 | 116 | 202 |
| Test | 143 | 226 | 73 | 20 | 57 | 76 |

Validation status: **ok** — 0 duplicate images, 0 missing labels, 0 corrupt
images, 0 invalid boxes, 0 unknown class IDs.

## 2. Detector training

Command used for the reported run:

```powershell
python scripts/train_yolo.py --epochs 30 --imgsz 448 --batch 16 --workers 4 --patience 10 --name ev_yolo_448
```

Artifacts:

- `results/training/ev_yolo_448/` — Ultralytics run, curves, confusion matrix
- `results/training/training_metadata.json` — exact parameters and elapsed time
- `results/training/results.csv` — per-epoch metrics
- `models/yolo/best.pt` — best checkpoint copied by the script

Training was performed on CPU. A one-epoch smoke run was used only to verify the
pipeline and is stored separately in `results/training/smoke_yolo/`; its metrics
are not reported as the final model.

Final training/test values are inserted into `results/report_tables.md` by
`scripts/generate_results.py` after `scripts/evaluate.py` runs. This avoids
freezing a partially trained checkpoint into the report.

## 3. Held-out detector evaluation

Command:

```powershell
python scripts/evaluate.py
```

Outputs:

- `results/metrics/detection_metrics.csv`
- `results/metrics/detection_metrics.json`
- `results/metrics/evaluation_run.json`
- `results/figures/detection_metrics.png`
- `results/figures/precision_recall_curve.png`, `f1_curve.png`,
  `precision_curve.png`, `recall_curve.png`
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png` only if the test split contains both classes

The JSON contains aggregate P/R/F1/mAP, per-class P/R/F1/mAP, operating-point
TP/FP/FN, per-class FP/FN, validation speed and the ROC availability flag.

**Status:** see `results/metrics/detection_metrics.json`. If the file is absent,
the result is “Not evaluated — model unavailable”.

## 4. Fixed-time vs priority simulation

Command:

```powershell
python simulation.py --sweep --seeds 42,43,44
```

Configuration:

- 600 s, `dt = 0.25 s`, seed 42;
- 287 vehicles, 5 emergency vehicles;
- normal arrivals 420/420/380/380 per lane per hour;
- scheduled EVs: north ambulance 40 s, east fire engine 170 s, south police 300 s, west ambulance 430 s, north fire engine 560 s;
- 2 s detection-confirmation proxy;
- 6 s priority hold before release simulation.

### 4.1 Aggregate result (seed 42)

| Metric | Fixed-time baseline | Emergency priority | Change |
|---|---:|---:|---:|
| Emergency mean wait | 10.00 s | **3.50 s** | −6.50 s |
| Emergency maximum wait | 26.00 s | **3.50 s** | −22.50 s |
| Normal mean wait | 9.71 s | 8.97 s | −0.74 s |
| Normal maximum wait | 28.17 s | 52.61 s | +24.44 s |
| Emergency served | 5 / 5 | 5 / 5 | — |
| Throughput | 1,704 veh/h | 1,710 veh/h | +6 veh/h |
| Priority activations | 0 | 5 | — |
| Priority response | not applicable | 1.50 s | — |

### 4.2 Event-level result (seed 42)

| # | Class | Lane | Arrival | Green | Proposed wait | Baseline wait | Difference |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | ambulance | north | 40.0 | 43.5 | 3.5 | 8.0 | **−4.5 s** |
| 2 | fire_engine | east | 170.0 | 173.5 | 3.5 | 0.0 | +3.5 s |
| 3 | police_vehicle | south | 300.0 | 303.5 | 3.5 | 0.0 | +3.5 s |
| 4 | ambulance | west | 430.0 | 433.5 | 3.5 | 26.0 | **−22.5 s** |
| 5 | fire_engine | north | 560.0 | 563.5 | 3.5 | 16.0 | **−12.5 s** |

This is the honest outcome: priority control is not uniformly better at the
individual-event level. Events 2 and 3 arrived on a baseline green and paid the
2 s confirmation + 1.5 s preparation cost. The mean improves because the two
large red-arrival savings (events 4 and 5) dominate. A production design would
need a fast path for an EV that already has a safe green.

### 4.3 Density sweep

Mean ± sample standard deviation across seeds 42, 43 and 44:

| Density | Baseline EV wait (s) | Proposed EV wait (s) | Baseline normal wait (s) | Proposed normal wait (s) | Baseline throughput | Proposed throughput | Priority events |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 10.00 ± 0.00 | 3.50 ± 0.00 | 8.72 ± 0.13 | 7.42 ± 0.88 | 724 ± 56 | 714 ± 52 | 15 |
| Medium | 10.00 ± 0.00 | 3.50 ± 0.00 | 8.50 ± 1.07 | 8.53 ± 0.46 | 1,682 ± 130 | 1,684 ± 137 | 15 |
| High | 10.00 ± 0.00 | 3.50 ± 0.00 | 8.37 ± 0.27 | 8.80 ± 0.50 | 3,516 ± 131 | 3,492 ± 101 | 15 |

The EV wait is identical across seeds because the EV schedule is fixed relative
to the signal cycle and the confirmation/preparation delays are deterministic.
Normal-traffic values vary with the Poisson arrivals. This is a property of the
signal-level simulator, not evidence that EV wait is independent of congestion
in reality.

At high density the proposed normal mean wait is 0.43 s higher and throughput is
24 veh/h lower in this sample; the difference is small relative to the
seed-to-seed spread and should not be over-interpreted. The project does not
claim a throughput benefit.

## 5. Live pipeline demonstration

Command (example):

```powershell
python main.py --source demo --max-frames 400 --no-display --save-video
```

Outputs:

- `results/videos/demo_annotated.mp4`
- `results/events/events.csv`, `events.jsonl`
- `results/metrics/performance.csv`, `performance_summary.json`
- `results/metrics/pipeline_summary.json`
- `results/figures/fps.png`, `latency.png`

Measured FPS/latency and priority events are session-specific. They are written
to `results/metrics/pipeline_summary.json` and summarized in
`results/report_tables.md`. No live-session number is claimed until that run has
actually happened.

## 6. Test suite

Command:

```powershell
python -m pytest -q
```

The suite covers:

- legal and illegal signal-state transitions;
- priority request/activate/release/cooldown/max-duration;
- lane assignment, unknown-lane fail-safe and exit detection;
- tracker birth, matching and loss;
- simulation accounting, baseline waits and event counts;
- comparison-table generation.

The exact pass count and duration are recorded by `run_all.py`; run the command
to reproduce them on your machine.

## 7. Dashboard check

```powershell
python scripts/check_dashboard.py
```

This runs the Streamlit app through `AppTest`, checks that all five tabs render
and that no exception is raised. A separate health check is performed by
`python run_all.py --check-dashboard`.

## 8. What was not evaluated

| Item | Reason |
|---|---|
| Real traffic-signal deployment | Out of scope and unsafe; controller is simulated |
| Real-world response-time benefit | Requires field trials and emergency-service data |
| Tracking MOTA/IDF1 | No video-level track annotations |
| ByteTrack/BoT-SORT comparison | Optional `lap` dependency unavailable |
| Night/rain/occlusion robustness | No labelled subsets for those conditions |
| Queue spillback / pedestrian phases | Not modelled by the signal-level simulator |
| Fire-engine deployment accuracy | Only 20 test boxes; too few for a strong claim |
| Image-level ROC | May be unavailable if the test split has no negative images |
