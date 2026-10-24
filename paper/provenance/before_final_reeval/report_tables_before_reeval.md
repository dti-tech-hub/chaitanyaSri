# Generated result tables

Generated from files under `results/` at 2026-10-22T02:51:53+00:00.

> Every value below is read from an actual run artifact. `Not evaluated` means the corresponding experiment did not produce a value; it is never replaced with an estimate.

## 1. Dataset audit

Validation status: **ok** · classes: ambulance, fire_engine, police_vehicle, normal_vehicle

| Split | Images | Boxes | ambulance | fire_engine | police_vehicle | normal_vehicle |
|---|---|---|---|---|---|---|
| train | 998 | 1,719 | 563 | 132 | 344 | 680 |
| val | 285 | 501 | 148 | 35 | 116 | 202 |
| test | 143 | 226 | 73 | 20 | 57 | 76 |

Errors: 0.

## 2. Training run

| Field | Value |
|---|---|
| Device | cpu |
| Image size | 512 |
| Batch | 8 |
| Epochs requested | 1 |
| Elapsed | 00:10:24 |
| Recorded epochs | 1 |
| Best epoch | 2 |

Best validation checkpoint (epoch 2): precision 0.409, recall 0.552, mAP@50 0.509, mAP@50-95 0.200.

## 3. Held-out detector evaluation

Model: `B:\chaitanyasri\emergency-vehicle-smart-signal\models\yolo\best.pt` · split: `test` · dataset: `B:\chaitanyasri\emergency-vehicle-smart-signal\data\dataset\data.yaml`

| Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 0.354 | 0.622 | 0.451 | 0.511 | 0.196 |

### Per-class test metrics

| Class | Instances | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| ambulance | 73 | 0.202 | 0.630 | 0.306 | 0.344 | 0.138 |
| fire_engine | 20 | 0.466 | 0.392 | 0.426 | 0.407 | 0.222 |
| police_vehicle | 57 | 0.128 | 0.930 | 0.225 | 0.624 | 0.176 |
| normal_vehicle | 76 | 0.620 | 0.537 | 0.576 | 0.668 | 0.248 |

### Operating point

| Conf | IoU | Images | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.50 | 143 | 73 | 71 | 153 | 0.507 | 0.323 | 0.395 |

Auxiliary image-level ROC AUC: **0.828** (image-level presence, not a detection mAP metric).

Validation speed (ms/image): preprocess=1.1, inference=70.9, loss=0.0, postprocess=3.7.

## 4. Simulated fixed-time vs emergency-priority control

Seed 42 · duration 600.0 s · 287 vehicles · 5 emergency vehicles. Signal-level simulation only.

| Metric | Fixed-time baseline | Emergency priority | Change |
|---|---|---|---|
| Emergency mean wait (s) | 10.00 | 3.50 | -6.50 |
| Emergency max wait (s) | 26.00 | 3.50 | -22.50 |
| Normal mean wait (s) | 9.71 | 8.97 | -0.74 |
| Normal max wait (s) | 28.17 | 52.61 | 24.44 |
| Emergency vehicles served | 5 / 5 | 5 / 5 | — |
| Throughput (veh/h) | 1704 | 1710 | 6 |
| Priority activations | 0 | 5 | — |
| Priority response (s) | Not applicable | 1.50 | — |

### Measured priority events

| # | Class | Lane | Arrival (s) | Green (s) | Wait (s) | Baseline wait (s) | Duration (s) |
|---|---|---|---|---|---|---|---|
| 1 | ambulance | north | 40.00 | 43.50 | 3.50 | 8.00 | 6.00 |
| 2 | fire_engine | east | 170.00 | 173.50 | 3.50 | 0.00 | 6.00 |
| 3 | police_vehicle | south | 300.00 | 303.50 | 3.50 | 0.00 | 6.00 |
| 4 | ambulance | west | 430.00 | 433.50 | 3.50 | 26.00 | 6.00 |
| 5 | fire_engine | north | 560.00 | 563.50 | 3.50 | 16.00 | 6.00 |

The normal-vehicle maximum wait can rise even when its mean falls: serving an EV out of sequence necessarily delays at least one conflicting movement.

## 5. Arrival-rate sweep

| Density | Baseline EV wait (s) | Proposed EV wait (s) | Baseline normal wait (s) | Proposed normal wait (s) | Baseline throughput | Proposed throughput | Priority events (all seeds) |
|---|---|---|---|---|---|---|---|
| low | 10.00 ± 0.00 | 3.50 ± 0.00 | 8.72 ± 0.13 | 7.42 ± 0.88 | 724.00 ± 55.75 | 714.00 ± 52.31 | 15 |
| medium | 10.00 ± 0.00 | 3.50 ± 0.00 | 8.50 ± 1.07 | 8.53 ± 0.46 | 1682.00 ± 130.40 | 1684.00 ± 136.86 | 15 |
| high | 10.00 ± 0.00 | 3.50 ± 0.00 | 8.37 ± 0.27 | 8.80 ± 0.50 | 3516.00 ± 130.77 | 3492.00 ± 100.94 | 15 |

Mean ± sample standard deviation across seeds [42, 43, 44].

## 6. Live computer-vision pipeline

Not evaluated — run `python main.py --source video --input <video> --save-video`.
