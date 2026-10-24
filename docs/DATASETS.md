# Datasets and licences

## 1. Training data

The detector was trained exclusively on labelled, real images from two public
Roboflow Universe projects. No synthetic images or model-generated labels were
used.

| # | Project | Workspace | Version | Original classes | Licence |
|---|---|---|---|---|---|
| 1 | `emergency-vehicle-crigt` | `jd-s39ap` | 1 | ambulance, fire brigade, police | CC BY 4.0 |
| 2 | `vehicle-detc.` | `emergency-vehicle-detection-sabir` | 1 | ambulance, fire-truck, vehicle | CC BY 4.0 |

Source pages:

- <https://universe.roboflow.com/jd-s39ap/emergency-vehicle-crigt/dataset/1>
- <https://universe.roboflow.com/emergency-vehicle-detection-sabir/vehicle-detc./dataset/1>

Both `data.yaml` files record `license: CC BY 4.0` and were downloaded from
those pages. The original metadata is preserved in
`data/raw/repos/AI-Traffic-Management-System/project/datasets/emergency_vehicles/`
and `data/raw/repos/Emergency-Vehicle-and-Density-Detection/vehicle-detc/`.

### 1.1 Changes made (CC BY 4.0 attribution requirement)

The preparation script `scripts/prepare_dataset.py`:

1. merged the two source datasets;
2. remapped class IDs to the canonical four-class scheme;
3. renamed output files with a source prefix to make provenance visible;
4. preserved the original train/val/test split membership (no re-shuffling);
5. dropped exact duplicate images by SHA-1;
6. copied every image and rewrote every YOLO label line;
7. recorded the source path, SHA-1, box count and empty-label flag for every
   output image in `data/dataset/metadata.csv`.

Canonical classes:

```text
0 ambulance
1 fire_engine
2 police_vehicle
3 normal_vehicle
```

The second dataset's `vehicle` class became `normal_vehicle`. The first
dataset's `fire brigade` and the second dataset's `fire-truck` both became
`fire_engine`.

### 1.2 Final split composition

| Split | Images | Boxes | Ambulance | Fire engine | Police | Normal |
|---|---:|---:|---:|---:|---:|---:|
| Train | 998 | 1,719 | 563 | 132 | 344 | 680 |
| Val | 285 | 501 | 148 | 35 | 116 | 202 |
| Test | 143 | 226 | 73 | 20 | 57 | 76 |

The held-out test split is never used for training, early stopping, threshold
selection or lane calibration.

### 1.3 Provenance counts before merging

| Source | Split | Images | Boxes | Ambulance | Fire engine | Police | Normal |
|---|---|---:|---:|---:|---:|---:|---:|
| emergency-vehicle-crigt | train | 745 | 771 | 409 | 18 | 344 | — |
| emergency-vehicle-crigt | val | 213 | 224 | 104 | 4 | 116 | — |
| emergency-vehicle-crigt | test | 107 | 108 | 50 | 1 | 57 | — |
| vehicle-detc. | train | 253 | 948 | 154 | 114 | — | 680 |
| vehicle-detc. | val | 72 | 277 | 44 | 31 | — | 202 |
| vehicle-detc. | test | 36 | 118 | 23 | 19 | — | 76 |

### 1.4 Known limitations of the data

- **Fire engines are scarce**: only 132 train, 35 val and 20 test boxes. Any
  fire-engine metric has a wide confidence interval and should not be treated
  as deployment-grade.
- The two source datasets were collected independently. Their cameras,
  countries, lighting, resolution and annotation conventions differ, which
  helps generalisation but also creates domain shift.
- `normal_vehicle` comes only from the second dataset. It is not a complete
  taxonomy of road users (no pedestrians, cyclists, buses or trucks as separate
  classes).
- Labels are bounding boxes, not instance masks or tracks. Tracking performance
  is therefore evaluated indirectly through the pipeline tests and demo, not
  with a MOT benchmark.
- The test split is small (143 images). Report exact counts alongside every
  metric; avoid over-interpreting differences of a few percentage points.

### 1.5 Excluded data

The Open Images Dataset (OID) subset present under `data/raw/oid/` was **not**
used. Its downloaded annotations contained only ambulance boxes, not the
fire-engine and police labels required for a four-class model. Including it
would have taught the model that fire engines and police vehicles do not exist.
This exclusion is recorded here rather than hidden.

## 2. Demo footage

`data/raw/ambu-test/` contains externally sourced clips used only for
demonstration and qualitative inspection:

- aerial intersection clips `9784626` and `9784628`;
- additional response/compilation clips whose filenames are self-describing.

These clips are **not** part of the training, validation or test data. They are
not relicensed by this project; their original licences/terms continue to
apply. Pexels-hosted clips (numeric filenames) retain the Pexels licence.
Compilation clips retain the terms of their original uploaders. Do not
redistribute them as part of a commercial package.

The default demo geometry is a schematic overhead four-way intersection. It is
not a calibrated reconstruction of the location in the footage. For a real
camera, run `tools/calibrate_lanes.py` and re-check every approach and exit ROI.

## 3. What would be needed for a stronger dataset

1. More fire-engine examples across cameras and countries (at least several
   hundred test instances).
2. Video-level sequences with persistent IDs to evaluate tracking (MOT-style
   MOTA/IDF1) rather than only per-frame detection.
3. Night, rain, glare and occlusion subsets.
4. A separate negative set of normal traffic for image-level false-alarm
   analysis.
5. Calibrated camera geometry and, ideally, naturalistic driving trajectories
   for a traffic microsimulation.

None of these were available in the submitted environment. The report states
that explicitly instead of extrapolating.
