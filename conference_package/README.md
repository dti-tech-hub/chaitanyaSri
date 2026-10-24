# Conference paper package — Emergency Vehicle Detection & Simulated Smart Signal

This folder contains everything needed to turn the repository into a conference submission:
a complete paper draft, every figure at 300 dpi, every table as Markdown and CSV, the
re-evaluation experiment, the provenance audits, and a submission checklist.

**One-line scope:** a CPU-only, four-class YOLOv8n detector plus a guarded, *simulated*
emergency-priority signal controller, evaluated with a leakage-screened detector test and a
deterministic event-level control experiment.

> **Safety and scope.** The signal controller is software only. It is not connected to a
> traffic light and was not validated on public roads. Nothing in this package supports a
> real-world safety, response-time, emissions or congestion claim.

---

## 1. Start here

| If you want to… | Open |
|---|---|
| Read or edit the paper | [`conference_paper.md`](conference_paper.md) |
| See what must be filled/checked before submission | [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) |
| Verify a number | [`TRACEABILITY.md`](TRACEABILITY.md) |
| Inspect the raw evidence | `artifacts/` and `provenance/` |
| Rebuild figures and tables | `python conference_package/scripts/make_paper_assets.py` |

The paper is 6,661 words including tables and references — roughly 9 pages in IEEE
two-column format. The abstract is 234 words. Trim to the venue limit (Section C of the
checklist suggests where).

---

## 2. Headline results (all real, all traceable)

### 2.1 Detector, full held-out test split (143 images, 226 boxes)

| Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---:|---:|---:|---:|---:|
| 0.985 | 0.982 | 0.984 | 0.989 | 0.660 |

### 2.2 Detector after the near-duplicate screen (56 images, 133 boxes)

| Precision | Recall | F1 | mAP@50 | mAP@50-95 | Operating-point P/R |
|---:|---:|---:|---:|---:|---:|
| 0.941 | 0.956 | 0.949 | 0.953 | 0.636 | 0.970 (129 TP, 4 FP, 4 FN) |

The screen is essential: 77 test images were already flagged by the committed audit, an
independent dHash pass flags 81, and their union removes **87 of 143**. Leakage is
concentrated in emergency frames: police test instances fall 57 → 10, ambulance 73 → 34,
fire engine 20 → 17, normal 76 → 72. The as-committed mAP is inflated; the screened values
are the ones the paper treats as primary.

### 2.3 Simulated control, 600 s, seed 42, 287 vehicles, 5 EVs

| Metric | Fixed-time | Priority | Change |
|---|---:|---:|---:|
| Emergency mean wait | 10.00 s | 3.50 s | −6.50 s (−65%) |
| Emergency max wait | 26.00 s | 3.50 s | −22.50 s (−86.5%) |
| Normal mean wait | 9.71 s | 8.97 s | −0.74 s |
| Normal max wait | 28.17 s | 52.61 s | **+24.44 s** |
| Throughput | 1,704 veh/h | 1,710 veh/h | +6 (noise) |
| EVs served | 5/5 | 5/5 | — |
| Priority activations | 0 | 5 | — |

Per event: −4.5 s, **+3.5 s**, **+3.5 s**, −22.5 s, −12.5 s. Priority is not uniformly
better; the mean is carried by two red-arrival events, and the 3.5 s cost is the fixed
2.0 s confirmation + 1.5 s all-red preparation.

---

## 3. Folder contents

```text
conference_package/
├── README.md                        ← this file
├── conference_paper.md              ← full paper draft (abstract → appendix)
├── SUBMISSION_CHECKLIST.md          ← fill-in list, venue decisions, ethics/formatting
├── TRACEABILITY.md                  ← every table/figure/headline number → artifact → script
├── ASSETS_MANIFEST.md               ← figure/table sources
├── figures/                         ← 14 PNGs at 300 dpi (5 drawn, 9 copied from real runs)
├── tables/                          ← 9 tables, each as .md and .csv
├── artifacts/                       ← self-contained copies of the evidence JSON/CSV
├── provenance/                      ← committed audit + reproducible dHash audit + summary
├── experiments/leakage_screened/    ← cleaned test split, data yaml, metrics, Ultralytics run
└── scripts/
    ├── audit_near_duplicates.py     ← reproducible 64-bit dHash audit (81 flags)
    ├── evaluate_screened.py         ← union screen, re-evaluation of the same checkpoint
    └── make_paper_assets.py         ← rebuilds every generated figure and table
```

`artifacts/` is a convenience copy. The canonical artifacts live under `results/` in the
repository and are never edited by hand. Seven figures are cited in the paper (1–5, 9, 13);
the rest are suitable supplementary material.

---

## 4. Reproduce the whole package

From the repository root:

```bash
python conference_package/scripts/audit_near_duplicates.py
python conference_package/scripts/evaluate_screened.py
python conference_package/scripts/make_paper_assets.py
```

Observed cost on the reported CPU: the audit takes seconds, the screened evaluation took 30.0 s
(recorded in `screened_metrics.json` → `protocol.elapsed_s`), and asset generation takes a few
seconds. The screened evaluation writes only inside
`conference_package/experiments/leakage_screened/` and does not touch `results/`.

To reproduce the underlying project artifacts (not needed to rebuild this package):

```bash
python scripts/validate_dataset.py
python scripts/evaluate.py
python simulation.py --sweep --seeds 42,43,44
python -m pytest -q
```

---

## 5. What is real and what is modelled

**Real measurements**
- Dataset counts and audit (images, boxes, classes, corrupt/missing/invalid = 0).
- Training cost and hyperparameters, best epoch, per-epoch validation curve.
- Held-out detector precision/recall/F1/mAP and fixed-threshold TP/FP/FN on real labelled images.
- Near-duplicate counts under two perceptual hash methods.
- 34 passing tests in 0.83 s; end-to-end runner status `ok`.

**Modelled**
- Signal timing, arrivals, waiting times, throughput and priority events come from a
  deterministic signal-level simulator, not from road measurements. The 2 s confirmation
  delay is a proxy for detection/tracking latency.
- Normal-vehicle arrivals are Poisson; real arrivals are correlated and time-varying.
- The intersection geometry is schematic. The demo footage is not calibrated to it.

**Not measured**
- Live end-to-end FPS/latency: `results/metrics/performance.csv` contains only a header and
  the event log contains only `session_start`. No live number is claimed.
- Tracking MOTA/IDF1: no video-level track annotations.
- Cross-city generalisation: both source datasets are from the same two collections.
- Real-world safety, response time, emissions or congestion.

---

## 6. Known inconsistencies in the repository (read before submitting)

These do not affect the paper — the paper reads the correct artifacts — but they will confuse
a reviewer who opens the repository.

1. **`results/report_tables.md` has a history of being stale.** Until this push it still showed
   the 1-epoch smoke checkpoint (P 0.354, R 0.622, mAP@50 0.511; "epochs requested 1"). It has
   now been regenerated from the final 30-epoch evaluation (P 0.985, R 0.982, mAP@50 0.989;
   best epoch 22). Re-run `python scripts/generate_results.py` after any new evaluation so it
   cannot drift again. The `paper/provenance/before_final_reeval/` copies preserve the old values
   for audit only.
2. **`docs/METHODOLOGY.md` says the test split is “disjoint by source image.”** That is true at
   the image level but not at the near-duplicate level; see the audit. Reword if you cite it.
3. **`paper/experiments/leakage_screened/` and `paper/experiments/sensitivity/` are empty
   placeholders.** This package's `experiments/leakage_screened/` is the implemented version.
4. **The PR curve was never copied into `results/figures/`.** `src/evaluation/detection_metrics.py`
   looks for `PR_curve.png`, while Ultralytics writes `BoxPR_curve.png`. The package copies
   `results/metrics/yolo_val_test/BoxPR_curve.png` directly.
5. **The committed near-duplicate audit has no generating script.** This package adds a
   reproducible dHash implementation; the two audits disagree on 16 images, so the evaluation
   removes their union. If you cite the audit, cite the package script for reproducibility.

---

## 7. Recommended submission decisions

1. **Lead with the screened detector result**, with the full split as an ablation. Reporting
   0.985 as the headline invites a leakage objection that the paper can pre-empt.
2. **Keep the per-event table.** It is the most credible part of the control result and the
   basis for the green-path recommendation.
3. **Keep the non-claims.** The simulated-controller disclaimer belongs in the abstract,
   introduction and threats to validity, not only in a footnote.
4. **State the small-sample statistics plainly.** The screened split has 56 images and 133
   boxes; fire engines 17, police 10. Do not add significance tests to a deterministic
   simulation with five EVs.
5. **If double-blind**, remove the repository URL, Windows paths, workspace names and demo
   footage references until after review.

---

## 8. Suggested data/code availability statement

> Code, configuration, generated tables and the trained checkpoint are available at
> [repository URL]. The two source datasets are public under CC BY 4.0 [12]. Demo clips are
> not redistributed and retain their original licences. The leakage-screened test split is
> generated deterministically by the included script; the audit lists both hash passes and
> their union.

## 9. Licence and attribution

Project code is MIT (`LICENSE` in the repository). Training data are CC BY 4.0 from the two
Roboflow Universe projects listed in `docs/DATASETS.md`; attribution is required if you
redistribute the merged dataset. Ultralytics YOLOv8 is used under its own licence. Demo clips
keep their original licences and are not part of the dataset.
