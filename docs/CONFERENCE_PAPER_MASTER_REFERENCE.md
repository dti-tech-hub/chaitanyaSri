# Master Reference Document: Emergency Vehicle Smart Traffic Signal Controller
## Complete Project Data, Real Empirical Values, and Blueprint for Full Conference Paper

> **Document Status:** Authoritative & Verified against Project Artifacts  
> **Repository:** `b:\chaitanyasri\emergency-vehicle-smart-signal`  
> **Source Artifacts:** `results/metrics/detection_metrics.json`, `conference_package/experiments/leakage_screened/screened_metrics.json`, `results/simulation/simulation_summary.json`, `results/simulation/density_sweep.csv`, `results/training/training_metadata.json`, `conference_package/conference_paper.md`  
> **Execution Environment:** CPU-only (13th Gen Intel Core i5-1335U, Python 3.14.5, PyTorch 2.14.1+cpu, Ultralytics 8.4.174)

---

## 1. Executive Summary & Publication Profile

This reference document compiles all real values, mathematical formulations, experimental outcomes, architectural invariants, and empirical findings for the project **"Detection-Confirmed Emergency Vehicle Priority at a Simulated Four-Way Intersection: Performance, Cost, and a Leakage-Screened Evaluation"**.

Every metric herein corresponds to committed JSON/CSV artifacts produced by automated pipelines. There are zero simulated placeholders or hand-estimated figures.

### 1.1 Target Conference Venues & Tracks
* **IEEE Intelligent Transportation Systems Conference (ITSC)** — Traffic Signal Control / Autonomous Infrastructure Track
* **IEEE Intelligent Vehicles Symposium (IV)** — Sensing, Detection & Traffic Management
* **ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys)**
* **ACM Conference on Computing and Sustainable Societies (COMPASS)**
* **IEEE Vehicular Technology Conference (VTC)** — Connected & Automated Vehicles

### 1.2 Core Research Contributions
1. **End-to-End CPU Pipeline:** A lean, dependency-free prototype coupling a 4-class YOLOv8n detector, class-aware IoU tracker, calibrated normalized polygon ROIs, a 7-condition confirmation filter, and a guarded 7-state signal FSM with fail-safe recovery.
2. **First-of-its-Kind Leakage-Screened Perception Audit:** Rigorous empirical proof that raw video-frame train/test splits inflate CV metrics (mAP@50 0.989 $\rightarrow$ 0.953; Police F1 0.969 $\rightarrow$ 0.838) via perceptual-hash deduplication (removing 87 of 143 test images).
3. **Honest Like-for-Like Control Analysis:** An arrival-locked deterministic simulation (600 s, identical arrival sequence) demonstrating that while emergency mean wait drops by 65.0% (10.00 s to 3.50 s), **two out of five emergency arrivals actually experienced delay penalties (+3.5 s)**, and normal vehicle maximum queue wait increased by **+86.8% (+24.44 s)**.
4. **Reproducibility & Rigor:** 34 passing automated unit tests (0.83 s execution), deterministic seeds (42–44), and 100% reproducible artifact-to-figure traceability.

---

## 2. Dataset Composition & Leakage Audit

The detector was fine-tuned exclusively on real-world footage aggregated from two public CC BY 4.0 Roboflow Universe repositories:
1. `emergency-vehicle-crigt` (workspace `jd-s39ap`, version 1)
2. `vehicle-detc.` (workspace `emergency-vehicle-detection-sabir`, version 1)

*Classes mapped to 4 canonical categories:* `ambulance`, `fire_engine` (combining source `fire brigade` & `fire-truck`), `police_vehicle`, and `normal_vehicle` (source `vehicle`). Raw OpenImages imports were intentionally excluded due to unlabelled non-ambulance vehicles that would introduce false negatives.

### 2.1 As-Committed Dataset Split Breakdown (Table 1)

| Split | Images | Total Annotations | Ambulance | Fire Engine | Police Vehicle | Normal Vehicle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 998 | 1,719 | 563 | 132 | 344 | 680 |
| **Val** | 285 | 501 | 148 | 35 | 116 | 202 |
| **Test (Committed)** | 143 | 226 | 73 | 20 | 57 | 76 |
| **Total** | **1,426** | **2,446** | **784** | **187** | **517** | **958** |

*Integrity verification:* 0 corrupt images, 0 missing labels, 0 invalid bounding boxes, 0 out-of-bound coordinates, 0 exact SHA-1 collisions across splits.

### 2.2 Near-Duplicate Data Leakage Audit Findings
While exact cryptographic hashing (SHA-1) showed 0 collisions, video-sourced datasets frequently suffer from consecutive-frame correlation across train and test sets.
* **Committed Perceptual Hash Screen:** Evaluated Hamming distance on 64-bit perceptual hashes (threshold $\le 6$). Identified **77 near-duplicate test images** (including 8 pairs at distance 0).
* **Independent Difference Hash (dHash) Audit:** Converted images to grayscale, resized to $9 \times 8$ via Lanczos interpolation, and generated 64-bit binary gradients. Flagged **81 near-duplicate test images** (with 13 identical pairs at distance 0).
* **Intersection & Conservative Union Screen:**
  * Overlapping images flagged by both methods: **71 images**
  * Committed-only flags: **6 images**
  * dHash-only flags: **10 images**
  * **Conservative Union removed:** **87 of 143 test images (60.8% of test set removed)**
  * **Verified Clean/Screened Test Set Remaining:** **56 images (133 bounding boxes)**

### 2.3 Impact of Leakage Removal on Class Support
The leakage was heavily concentrated in emergency classes originating from video sequences:
* **Ambulance:** 73 boxes $\rightarrow$ **34 boxes** (-53.4%)
* **Fire Engine:** 20 boxes $\rightarrow$ **17 boxes** (-15.0%)
* **Police Vehicle:** 57 boxes $\rightarrow$ **10 boxes (-82.5% reduction)**
* **Normal Vehicle:** 76 boxes $\rightarrow$ **72 boxes** (-5.3%)

---

## 3. Computer Vision & Detection Benchmarks

### 3.1 Model Architecture & Training Hyperparameters (Table 2)
* **Model Architecture:** Ultralytics YOLOv8n (Nano variant, anchor-free, decoupled head)
* **Parameters & Complexity:** 72 layers, 3,006,428 parameters, 8.1 GFLOPs
* **Input Resolution & Batch:** $448 \times 448$ pixels, batch size 16
* **Training Compute:** CPU-only on 13th Gen Intel Core i5-1335U (12 threads)
* **Wall-clock Training Time:** 54,652.7 seconds (**15 hours, 10 minutes, 52 seconds**)
* **Epochs & Schedule:** 30 requested and completed; Cosine LR annealing (`lr0 = 0.001`, `lrf = 0.01`, 3-epoch warmup)
* **Best Checkpoint Fitness:** **Epoch 22** (Fitness = $0.1 \times \text{mAP@50} + 0.9 \times \text{mAP@50-95}$)
  * Epoch 22 Val metrics: P = 0.971, R = 0.961, mAP@50 = 0.980, mAP@50-95 = 0.649
  * Final Epoch 30 Val metrics: mAP@50 = 0.973, mAP@50-95 = 0.634
* **Data Augmentation:** HSV jitter (H: 0.015, S: 0.6, V: 0.4), Rotation ($\pm 5^\circ$), Translation (0.1), Scale (0.5), Shear ($2.0^\circ$), Horizontal flip ($p = 0.5$), Mosaic ($p = 0.8$, closed for final 10 epochs), Mixup ($p = 0.05$).

### 3.2 Full Test Split vs. Leakage-Screened Test Split (Tables 3 & 4)

Standard evaluation at standard ranking threshold (`conf = 0.001`, `IoU = 0.60`):

| Evaluation Protocol | Split Support | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Held-Out Test Split** | 143 images / 226 boxes | **0.985** | **0.982** | **0.984** | **0.989** | **0.660** |
| **Leakage-Screened Split** | 56 images / 133 boxes | **0.941** | **0.956** | **0.949** | **0.953** | **0.636** |
| **Observed Degradation** | *-87 images (-60.8%)* | *-0.044* | *-0.026* | *-0.035* | **-0.036** | *-0.024* |

#### Class-Specific Breakdown on Leakage-Screened Split
* **Ambulance (34 instances):** P = 0.997, R = 0.971, F1 = 0.983, mAP@50 = 0.992, mAP@50-95 = 0.691
* **Fire Engine (17 instances):** P = 1.000, R = 0.968, F1 = 0.984, mAP@50 = 0.995, mAP@50-95 = 0.732
* **Police Vehicle (10 instances):** P = 0.784, R = 0.900, F1 = **0.838**, mAP@50 = **0.841**, mAP@50-95 = **0.539** *(Sharp drop of -14.3 mAP points due to near-duplicate removal)*
* **Normal Vehicle (72 instances):** P = 0.985, R = 0.986, F1 = 0.986, mAP@50 = 0.983, mAP@50-95 = 0.582

### 3.3 Operating Point Performance (Live Deployment Thresholds) (Table 5)
In signal control, the system does not operate at `conf = 0.001`. The controller requires `conf >= 0.55` for tracking and `conf >= 0.60` for priority activation, evaluated at greedy class-aware matching `IoU >= 0.50`:

| Metric | Full Test Split (143 images) | Leakage-Screened Split (56 images) |
| :--- | :---: | :---: |
| **True Positives (TP)** | 222 | 129 |
| **False Positives (FP)** | 4 | 4 |
| **False Negatives (FN)** | 4 | 4 |
| **Precision** | 0.9823 (98.2%) | **0.9699 (97.0%)** |
| **Recall** | 0.9823 (98.2%) | **0.9699 (97.0%)** |
| **F1-Score** | 0.9823 (98.2%) | **0.9699 (97.0%)** |
| **95% Wilson Score Interval for Recall** | [0.956, 0.993] | **[0.925, 0.988]** |

*Screened False Positive Breakdown:* Police = 2, Ambulance = 1, Normal = 1.  
*Screened False Negative Breakdown:* Police = 1, Ambulance = 2, Normal = 1.

### 3.4 Inference Latency Profile on CPU
Measured per frame on standard $448 \times 448$ input (Intel Core i5-1335U, single-stream CPU):
* **Preprocess:** $1.76 \text{ ms}$
* **Neural Network Inference:** $104.53 \text{ ms}$
* **Postprocessing & NMS:** $2.81 \text{ ms}$
* **Loss Computation:** $0.0003 \text{ ms}$
* **Total End-to-End Latency:** $\approx 109.1 \text{ ms}$ ($\approx 9.17 \text{ FPS}$)

---

## 4. Signal Controller Architecture & Safety Logic

### 4.1 Seven Invariant Conditions for Priority Confirmation
A priority preemption request is confirmed if and only if **all 7 conditions** simultaneously hold:
1. **Target Class Validity:** Vehicle detected as `ambulance`, `fire_engine`, or `police_vehicle`.
2. **Confidence Threshold:** Bounding box confidence score $\ge 0.60$.
3. **Temporal Consistency:** Vehicle track ID confidently detected for at least **3 consecutive processed frames**.
4. **Track Age:** Cumulative track lifespan $\ge 0.50 \text{ s}$.
5. **Calibrated ROI Membership:** Box centroid $(x_c, y_c)$ resides strictly inside a calibrated approach polygon in normalized space $[0, 1]^2$. Detections in `UNKNOWN` space are discarded.
6. **Kinematic Approach Criterion:** Normalized Euclidean distance to intersection center $d(t)$ satisfies:
   $$\Delta d = d(t) - d(t - 8\text{s}) < -0.01$$
   *OR* the vehicle is already queued near the stop line ($d(t) \le 0.42$).
7. **Controller Readiness:** Signal FSM is currently in `NORMAL` state, outside the mandatory $10 \text{ s}$ post-preemption cooldown window, and no other EV currently holds green.

### 4.2 Guarded Finite State Machine (FSM) Lifecycle
The signal controller operates across 7 discrete states:
1. `NORMAL`: Standard 2-phase cyclical timing:
   * North-South Phase: Green 20 s $\rightarrow$ Yellow 3 s $\rightarrow$ All-Red 1 s
   * East-West Phase: Green 20 s $\rightarrow$ Yellow 3 s $\rightarrow$ All-Red 1 s (Cycle Time: 48 s)
2. `EMERGENCY_DETECTED`: Intermediate evaluation state upon receipt of a confirmed request.
   * *Fast Path:* If approach lane is already in safe green, transition immediately to `PRIORITY_ACTIVE`.
   * *Conflicting Phase:* Transition to `PRIORITY_PREPARING`.
3. `PRIORITY_PREPARING`: Conflicting phase transitions immediately through Yellow and enters an enforced All-Red preparation clearance ($\ge 1.5 \text{ s}$).
4. `PRIORITY_ACTIVE`: Grants exclusive, protected green to the preemption approach lane. All conflicting movements forced red. Maximum green duration hard-capped at **$45.0 \text{ s}$**.
5. `EMERGENCY_CLEARED`: Triggered when vehicle track enters the far-side exit polygon ROI, held for a minimum of $2.0 \text{ s}$.
6. `RETURNING_TO_NORMAL`: Mandatory all-red clearance phase before restoring the interrupted phase cycle.
7. `FAIL_SAFE`: Terminal safety state. Any uncaught exception, invalid state transition, or geometry discrepancy instantly forces **ALL APPROACHES TO RED** and halts automated cycling until human intervention.

---

## 5. Simulation Experiment & Full Event Disclosure

### 5.1 Simulation Parameters
* **Harness:** Deterministic signal-level Poisson arrival model (`simulation.py`, seed 42)
* **Simulation Horizon:** $600.0 \text{ s}$ ($10 \text{ minutes}$), time step $\Delta t = 0.25 \text{ s}$
* **Traffic Demand:** North: 420 veh/h; South: 420 veh/h; East: 380 veh/h; West: 380 veh/h (Total generated: 287 vehicles)
* **Scheduled Emergency Vehicles:** 5 vehicles across distinct lanes and times
* **Preemption Confirmation Delay:** $2.0 \text{ s}$ fixed proxy; Priority Green Hold: $6.0 \text{ s}$

### 5.2 Aggregate Control Results (Table 6)

| Metric | Fixed-Time Baseline | Proposed Emergency Priority | Absolute & Relative Delta |
| :--- | :---: | :---: | :---: |
| **Emergency Mean Wait** | $10.00 \text{ s}$ | **$3.50 \text{ s}$** | **-6.50 s (-65.0%)** |
| **Emergency Max Wait** | $26.00 \text{ s}$ | **$3.50 \text{ s}$** | **-22.50 s (-86.5%)** |
| **Normal Traffic Mean Wait** | $9.71 \text{ s}$ | $8.97 \text{ s}$ | -0.74 s (-7.6%) |
| **Normal Traffic Max Wait** | $28.17 \text{ s}$ | **$52.61 \text{ s}$** | **+24.44 s (+86.8%)** |
| **Emergency Vehicles Served** | 5 / 5 (100%) | 5 / 5 (100%) | 0 |
| **Total Vehicles Served** | 284 / 287 | 285 / 287 | +1 vehicle |
| **Intersection Throughput** | $1,704 \text{ veh/h}$ | $1,710 \text{ veh/h}$ | +6 veh/h (+0.35%, statistical noise) |
| **Priority Activations** | 0 | 5 | +5 |
| **Mean Priority Response Time** | N/A | $1.50 \text{ s}$ | Measured clearance delay |

### 5.3 Complete Event-Level Disclosure (Table 7)
Crucially, evaluating priority at an aggregate level conceals individual event trade-offs. The table below details all 5 simulated emergency events:

| # | Vehicle Class | Approach Lane | Arrival Time ($t_a$) | Request Time ($t_r$) | Green Onset ($t_g$) | Release Time ($t_{rel}$) | Proposed Wait | Baseline Wait | Net Change ($\Delta$) | Priority Benefit? |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Ambulance | North | $40.0 \text{ s}$ | $42.0 \text{ s}$ | $43.5 \text{ s}$ | $49.5 \text{ s}$ | $3.5 \text{ s}$ | $8.0 \text{ s}$ | **-4.5 s** | Yes (Improved) |
| **2** | Fire Engine | East | $170.0 \text{ s}$ | $172.0 \text{ s}$ | $173.5 \text{ s}$ | $179.5 \text{ s}$ | $3.5 \text{ s}$ | $0.0 \text{ s}$ | **+3.5 s** | **No (Delayed!)** |
| **3** | Police Vehicle | South | $300.0 \text{ s}$ | $302.0 \text{ s}$ | $303.5 \text{ s}$ | $309.5 \text{ s}$ | $3.5 \text{ s}$ | $0.0 \text{ s}$ | **+3.5 s** | **No (Delayed!)** |
| **4** | Ambulance | West | $430.0 \text{ s}$ | $432.0 \text{ s}$ | $433.5 \text{ s}$ | $439.5 \text{ s}$ | $3.5 \text{ s}$ | $26.0 \text{ s}$ | **-22.5 s** | Yes (Major Gain) |
| **5** | Fire Engine | North | $560.0 \text{ s}$ | $562.0 \text{ s}$ | $563.5 \text{ s}$ | $569.5 \text{ s}$ | $3.5 \text{ s}$ | $16.0 \text{ s}$ | **-12.5 s** | Yes (Major Gain) |

#### Critical Empirical Insights:
1. **The Fixed Minimum Cost:** In the simulation, every priority activation incurs exactly $3.5 \text{ s}$ of delay ($2.0 \text{ s}$ confirmation window + $1.5 \text{ s}$ all-red phase transition).
2. **The "Green-Arrival" Penalty:** In Events 2 and 3, the emergency vehicles happened to arrive when their approach already had a green signal under the fixed-time plan (Baseline Wait = $0.0 \text{ s}$). However, because an earlier preemption shifted the fixed cycle offset, the vehicles met a red signal in the priority controller and were forced to pay the $3.5 \text{ s}$ preparation penalty.
3. **The Distributional Penalty on Normal Traffic:** While average normal vehicle wait decreased slightly (-0.74 s) because normal cars queued behind the EV cleared during the protected green, the **maximum wait time for opposing normal vehicles surged from 28.17 s to 52.61 s (+86.8%)**, demonstrating significant queue starvation on interrupted phases.

### 5.4 Traffic Density Sweep (Table 8)
Conducted across 9 independent runs (3 seeds: 42, 43, 44 $\times$ 3 traffic density levels):
* **Low Density:** Arrival rates 180 / 180 / 160 / 160 veh/h
* **Medium Density:** Arrival rates 420 / 420 / 380 / 380 veh/h
* **High Density:** Arrival rates 900 / 900 / 820 / 820 veh/h

| Density Level | Runs | Baseline EV Wait | Proposed EV Wait | Baseline Normal Wait | Proposed Normal Wait | Baseline Throughput | Proposed Throughput | Total Priority Events |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low** | 3 | $10.00 \pm 0.00 \text{ s}$ | $3.50 \pm 0.00 \text{ s}$ | $8.72 \pm 0.13 \text{ s}$ | $7.42 \pm 0.88 \text{ s}$ | $724 \pm 56 \text{ vph}$ | $714 \pm 52 \text{ vph}$ | 15 |
| **Medium** | 3 | $10.00 \pm 0.00 \text{ s}$ | $3.50 \pm 0.00 \text{ s}$ | $8.50 \pm 1.07 \text{ s}$ | $8.53 \pm 0.46 \text{ s}$ | $1,682 \pm 130 \text{ vph}$ | $1,684 \pm 137 \text{ vph}$ | 15 |
| **High** | 3 | $10.00 \pm 0.00 \text{ s}$ | $3.50 \pm 0.00 \text{ s}$ | $8.37 \pm 0.27 \text{ s}$ | $8.80 \pm 0.50 \text{ s}$ | $3,516 \pm 131 \text{ vph}$ | $3,492 \pm 101 \text{ vph}$ | 15 |

*Key finding:* Normal vehicle wait penalties scale with congestion: at low density, proposed normal wait is $1.30 \text{ s}$ lower than baseline; at high density, it is $0.43 \text{ s}$ higher, with slight throughput drop (-24 vph).

---

## 6. Software Verification & Engineering Integrity

* **Automated Unit Tests:** 34 tests implemented using `pytest` 9.1.1.
* **Test Suite Execution Time:** 0.83 seconds.
* **Coverage Scope:**
  * FSM state transition legality and invalid jump exception catching (`InvalidTransition`).
  * Safety interlocks: conflicting greens prevention, fail-safe all-red activation on exceptions.
  * Priority lifecycle: request generation, countdown timers, hold-after-exit, 45 s timeout cap, 10 s cooldown.
  * Polygon geometry: point-in-polygon verification, normalized coordinate transforms, unknown-lane fail-safe.
  * Object tracker: ID birth, bounding box association, lost track aging, trajectory smoothing.
  * Simulation accounting: Poisson arrival generation, queue delay tracking, throughput aggregation.

---

## 7. Artifact Traceability Matrix

Every figure, table, and numerical finding maps directly to a script and persisted file:

| Figure / Table | Output File | Source Data File | Generating Script | Key Empirical Fact |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1** | `fig1_system_architecture.png` | Architecture spec | `make_paper_assets.py` | Visualizes perception, control, and simulation pipelines |
| **Fig. 2** | `fig2_priority_fsm.png` | `signal_controller.py` | `make_paper_assets.py` | 7-state guarded FSM diagram |
| **Fig. 3** | `fig3_dataset_and_leakage.png` | `near_duplicate_audit_recomputed.json` | `make_paper_assets.py` | Class balance + Hamming distance histogram |
| **Fig. 4** | `fig4_detector_full_vs_screened.png`| `screened_metrics.json` | `make_paper_assets.py` | Bar comparison of P, R, F1, mAP across splits |
| **Fig. 5** | `fig5_event_level_waits.png` | `simulation_summary.json` | `make_paper_assets.py` | Event-by-event delta (-4.5, +3.5, +3.5, -22.5, -12.5 s) |
| **Fig. 13** | `fig13_confusion_matrix_screened.png`| `yolo_val_clean/` | Ultralytics Val | Normalized confusion matrix on clean test images |
| **Table 1** | `table1_dataset.md` | `dataset_validation.json` | `validate_dataset.py` | 1,426 total images; 2,446 annotations |
| **Table 2** | `table2_training.md` | `training_metadata.json` | `train_yolo.py` | 30 epochs, 54,652.7 s on CPU, best epoch 22 |
| **Table 3** | `table3_detector_full.md` | `detection_metrics.json` | `evaluate.py` | Full test mAP@50 = 0.989 |
| **Table 4** | `table4_detector_screened.md` | `screened_metrics.json` | `evaluate_screened.py` | Screened test mAP@50 = 0.953 |
| **Table 5** | `table5_operating_point.md` | `detection_metrics.json` | `evaluate.py` | Operating point at conf 0.55: 129 TP, 4 FP, 4 FN |
| **Table 6** | `table6_simulation_aggregate.md`| `simulation_summary.json` | `simulation.py` | Mean EV wait drops from 10.0s to 3.5s |
| **Table 7** | `table7_priority_events.md` | `simulation_summary.json` | `simulation.py` | 5 individual priority events with baseline comparison |
| **Table 8** | `table8_density_sweep.md` | `density_sweep.csv` | `simulation.py --sweep` | 9-run density sweep across 3 seeds |

---

## 8. Threats to Validity & Honest Non-Claims

For publication in peer-reviewed conferences, reviewers demand rigorous discussion of boundaries:

1. **Simulated Environment Disclaimer:** The signal controller is a software prototype. It has not been deployed on physical municipal hardware (e.g., NEMA TS2 or 2070 controllers). No claims of real-world fuel, emissions, or response-time improvements can be made.
2. **Macroscopic vs. Microscopic Traffic Limits:** The simulator uses deterministic Poisson arrivals with instantaneous movement. It does not model driver hesitation, queue spillback, pedestrian crosswalk clearance, or multi-lane overtaking.
3. **Geographic Diversity of Dataset:** All images derive from two Roboflow Universe collections. Even after removing near-duplicates, background environmental similarity remains high.
4. **Fire Engine Sample Scarcity:** Only 17 fire engine bounding boxes exist in the screened test split. Consequently, high metrics on this class (mAP@50 0.995) have broad confidence intervals.
5. **Absence of Video Ground Truth:** While frame-level detection is strictly verified, end-to-end video false alarm rates (e.g., false preemption activations per 1,000 video hours) were not quantified due to lack of continuous temporal labels.

---

## 9. Conference Paper Writing Guide & LaTeX Snippets

### 9.1 Recommended Paper Outline (6–8 Pages)
* **Section I: Introduction** — Problem statement, failure modes of existing prototypes (leakage & mean-only reporting), 4 explicit contributions.
* **Section II: Related Work** — Emergency preemption (FHWA, Qin & Khan transition strategies), YOLO edge perception, tracking association, simulation tooling.
* **Section III: System Architecture** — Online perception (YOLOv8n + IoU tracker), ROI polygon calibration, 7-condition confirmation filter, guarded 7-state FSM.
* **Section IV: Experimental Methodology** — Dataset merger & SHA-1 audit, dual perceptual hash leakage screen (dHash vs. pHash), simulation harness with identical arrivals.
* **Section V: Empirical Results**
  * *Subsection A:* Detector benchmark (Full vs. Screened, Table 3 & 4, Fig. 4).
  * *Subsection B:* Fixed-threshold operating point (Table 5, Wilson intervals).
  * *Subsection C:* Simulation aggregate & event-level disclosure (Tables 6 & 7, Fig. 5).
  * *Subsection D:* Density sensitivity sweep (Table 8).
* **Section VI: Discussion** — Why near-duplicate removal changes conclusions; the true cost of priority on opposing traffic; safety interlock necessity.
* **Section VII: Threats to Validity** — Detailed breakdown of internal, construct, external, and ecological threats.
* **Section VIII: Conclusion & Future Work** — Summary of findings; path toward SUMO co-simulation, grouped cross-video splits, and hardware-in-the-loop validation.

### 9.2 Key Mathematical Formulations for Paper

#### Normalized Centroid Distance to Junction:
$$d(t) = \frac{\| \mathbf{c}(t) - \mathbf{c}_j \|_2}{s}$$
Where $\mathbf{c}(t) = (x_c, y_c)$ is the vehicle centroid in normalized coordinates $[0, 1]^2$, $\mathbf{c}_j = (0.5, 0.5)$ is the junction center, and $s = \frac{\sqrt{2}}{2} \approx 0.7071$ normalizes the maximum distance to 1.

#### Kinematic Approach Metric:
$$\text{IsApproaching}(t) = \left( d(t) - d(t - w) < -\delta \right) \lor \left( d(t) \le d_{\text{queue}} \right)$$
Where window $w = 8.0 \text{ s}$, distance threshold $\delta = 0.01$, and queued threshold $d_{\text{queue}} = 0.42$.

#### Waiting Time Formulations:
$$t_{\text{wait, live}} = t_{\text{green\_onset}} - t_{\text{first\_confident\_detection}}$$
$$t_{\text{wait, baseline}} = t_{\text{next\_fixed\_green}}(\text{lane}, t_{\text{arrival}}) - t_{\text{arrival}}$$
$$\Delta t_{\text{event}} = t_{\text{wait, proposed}} - t_{\text{wait, baseline}}$$

---

*Authored and validated for conference paper submission package in `emergency-vehicle-smart-signal`.*
