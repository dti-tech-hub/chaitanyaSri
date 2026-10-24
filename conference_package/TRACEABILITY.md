# Traceability: every paper number to its artifact

Each row says where a value comes from and which script writes it. If a number in
`conference_paper.md` is not in this file, it was typed by hand and must be checked or removed.

## 1. Figures

| Paper element | File | Source | How it is produced |
|---|---|---|---|
| Fig. 1 architecture | `figures/fig1_system_architecture.png` | `docs/ARCHITECTURE.md`, `src/` module boundaries | drawn by `make_paper_assets.py` |
| Fig. 2 priority FSM | `figures/fig2_priority_fsm.png` | `src/control/signal_controller.py`, `docs/ARCHITECTURE.md` | drawn by `make_paper_assets.py` |
| Fig. 3 dataset + leakage | `figures/fig3_dataset_and_leakage.png` | `artifacts/dataset_validation.json`, `provenance/near_duplicate_audit_recomputed.json`, `provenance/near_duplicate_audit_summary.json` | `make_paper_assets.py` |
| Fig. 4 full vs screened | `figures/fig4_detector_full_vs_screened.png` | `artifacts/detection_metrics.json`, `artifacts/screened_metrics.json` | `make_paper_assets.py` |
| Fig. 5 event waits | `figures/fig5_event_level_waits.png` | `artifacts/simulation_summary_trimmed.json` | `make_paper_assets.py` |
| Fig. 6 dataset samples | `figures/fig6_dataset_samples.png` | `results/figures/dataset_samples.png` | copied real plot |
| Fig. 7 full confusion | `figures/fig7_confusion_matrix_full.png` | `results/figures/confusion_matrix_normalized.png` | copied real plot |
| Fig. 8 full PR curve | `figures/fig8_precision_recall_full.png` | `results/metrics/yolo_val_test/BoxPR_curve.png` | copied real plot |
| Fig. 9 simulation aggregate | `figures/fig9_simulation_aggregate.png` | `results/figures/simulation_waiting_time.png` | copied real plot |
| Fig. 10 EV wait distribution | `figures/fig10_ev_wait_distribution.png` | `results/figures/simulation_wait_distribution.png` | copied real plot |
| Fig. 11 metric bars | `figures/fig11_detection_metrics_plot.png` | `results/figures/detection_metrics.png` | copied real plot |
| Fig. 12 auxiliary ROC | `figures/fig12_auxiliary_roc.png` | `results/figures/roc_curve.png` | copied real plot |
| Fig. 13 screened confusion | `figures/fig13_confusion_matrix_screened.png` | `experiments/leakage_screened/yolo_val_clean/confusion_matrix_normalized.png` | Ultralytics validation |
| Fig. 14 screened PR curve | `figures/fig14_precision_recall_screened.png` | `experiments/leakage_screened/yolo_val_clean/BoxPR_curve.png` | Ultralytics validation |

## 2. Tables

| Paper table | Package file | Source artifact | Script |
|---|---|---|---|
| Table 1 dataset | `tables/table1_dataset.md` | `artifacts/dataset_validation.json` → `splits.*.{images,boxes,class_counts}` | `scripts/validate_dataset.py` |
| Table 2 training | `tables/table2_training.md` | `artifacts/training_metadata.json`; `results/training/ev_yolo_448/results.csv` | `scripts/train_yolo.py` |
| Table 3 full test | `tables/table3_detector_full.md` | `artifacts/detection_metrics.json` | `scripts/evaluate.py` |
| Table 4 screened | `tables/table4_detector_screened.md` | `artifacts/screened_metrics.json` | `conference_package/scripts/evaluate_screened.py` |
| Table 5 operating point | `tables/table5_operating_point.md` | both JSONs → `operating_point` | both evaluation scripts |
| Table 6 control aggregate | `tables/table6_simulation_aggregate.md` | `artifacts/simulation_summary_trimmed.json` → `baseline`, `proposed` | `simulation.py` |
| Table 7 events | `tables/table7_priority_events.md` | `artifacts/simulation_summary_trimmed.json` → `priority_events` | `simulation.py` |
| Table 8 sweep | `tables/table8_density_sweep.md` | `artifacts/density_sweep.csv` | `simulation.py --sweep` |
| Table A1 parameters | in paper | `config/config.yaml`, `config/intersection.yaml` | hand-maintained config |
| Table A2 environment | `tables/table9_environment.md` | `docs/ENVIRONMENT.md`, evaluation log | recorded, not generated |
| 34 tests | in paper §5.6 | `python -m pytest -q` | pytest 9.1.1 |

## 3. Headline numbers

| Value in paper | Artifact | Key | Written by |
|---|---|---|---|
| 143 test images / 226 boxes | `artifacts/dataset_validation.json` | `splits.test.images`, `splits.test.boxes` | `validate_dataset.py` |
| 56 images / 133 boxes (screened) | `artifacts/screened_metrics.json` | `split.images_kept`; sum of `per_class[].instances` | `evaluate_screened.py` |
| Full P 0.985288 → 0.985 | `artifacts/detection_metrics.json` | `overall.precision` | `evaluate.py` |
| Full R 0.982266 → 0.982 | same | `overall.recall` | same |
| Full F1 0.983774 → 0.984 | same | `overall.f1` | same |
| Full mAP@50 0.988837 → 0.989 | same | `overall.mAP50` | same |
| Full mAP@50-95 0.660483 → 0.660 | same | `overall.mAP50-95` | same |
| Full TP/FP/FN 222/4/4 | same | `operating_point.{tp,fp,fn}` | same |
| Full operating P/R 0.9823 | same | `operating_point.precision`, `.recall` | same |
| Screened P 0.941486 → 0.941 | `artifacts/screened_metrics.json` | `overall.precision` | `evaluate_screened.py` |
| Screened R 0.956296 → 0.956 | same | `overall.recall` | same |
| Screened F1 0.948833 → 0.949 | same | `overall.f1` | same |
| Screened mAP@50 0.952714 → 0.953 | same | `overall.mAP50` | same |
| Screened mAP@50-95 0.636006 → 0.636 | same | `overall.mAP50-95` | same |
| Screened TP/FP/FN 129/4/4 | same | `operating_point.{tp,fp,fn}` | same |
| Screened operating P/R 0.9699 → 0.970 | same | `operating_point.precision`, `.recall` | same |
| 77 committed near-duplicates | `provenance/near_duplicate_audit_summary.json` | `committed_flagged` | committed audit (no script) |
| 81 dHash near-duplicates | same | `recomputed_flagged` | `audit_near_duplicates.py` |
| 71 overlap / 6+10 disagreement | same | `overlap`, `committed_only`, `recomputed_only` | same |
| 87 union, 56 kept | same; `artifacts/screened_metrics.json` | `union`; `split.images_flagged`, `split.images_kept` | both |
| 13 identical dHash pairs | `provenance/near_duplicate_audit_recomputed.json` | count of `pairs[].hamming == 0` | `audit_near_duplicates.py` |
| Class counts 73→34, 20→17, 57→10, 76→72 | `artifacts/dataset_validation.json` vs `screened_metrics.json` | `splits.test.class_counts`, `per_class[].instances` | validation + screened eval |
| 54,652.7 s training | `artifacts/training_metadata.json` | `elapsed_s` | `train_yolo.py` |
| 30 epochs, 448 px, batch 16 | same | `epochs_requested`, `imgsz`, `batch` | same |
| Best epoch 22 | `results/training/ev_yolo_448/results.csv` | argmax of 0.1·mAP@50 + 0.9·mAP@50-95 | Ultralytics fitness |
| Best-val P/R/mAP 0.971/0.961/0.980/0.649 | same | row `epoch=22` | same |
| YOLOv8n 3,006,428 params, 8.1 GFLOPs | evaluation/training log | model summary | Ultralytics |
| Emergency mean 10.00 → 3.50 s | `artifacts/simulation_summary_trimmed.json` | `baseline.emergency_waiting_time_s`, `proposed...` | `simulation.py` |
| Emergency max 26.00 → 3.50 s | same | `*.emergency_max_wait_s` | same |
| Normal mean 9.710 → 8.973 s | same | `*.normal_waiting_time_s` | same |
| Normal max 28.171 → 52.614 s | same | `*.normal_max_wait_s` | same |
| Throughput 1,704 → 1,710 veh/h | same | `*.throughput_vph` | same |
| 5/5 EVs served | same | `*.emergency_served`, `.emergency_total` | same |
| 5 activations, 1.5 s response | same | `proposed.priority_events`, `priority_response_time_s` | same |
| Event deltas −4.5/+3.5/+3.5/−22.5/−12.5 | same | `priority_events[].waiting_time_s` − `baseline_wait_s` | same |
| Sweep values (Table 8) | `artifacts/density_sweep.csv` | mean ± sample SD across seeds | `simulation.py --sweep` |
| 34 tests, 0.83 s | pytest output (`34 passed in 0.83s`) | test suite | `python -m pytest -q` |
| 95% Wilson interval [0.925, 0.988] | computed from 129/133 | Wilson score interval, z=1.96 | derived (not in artifact) |
| 95% Wilson interval [0.956, 0.993] | computed from 222/226 | same | derived |

Percent changes, the best-epoch argmax and the two Wilson intervals are derived from stored
counts rather than stored directly. They are simple arithmetic or closed-form statistics on
values in the table above; if the venue requires reproducibility for them, add a short script
to `scripts/`.

## 4. Deprecated values — do not cite

`results/report_tables.md` and `results/metrics/report_tables.json` were previously generated
on 2026-10-22 02:51 UTC from the **1-epoch smoke checkpoint**, before the final 30-epoch model
was evaluated. They contained P 0.354, R 0.622, F1 0.451, mAP@50 0.511 and "epochs requested
1". They have now been regenerated from `results/metrics/detection_metrics.json` and report the
final values (P 0.985, best epoch 22). The old numbers must not appear in the paper; the
pre-regeneration copies are preserved under `paper/provenance/before_final_reeval/`.

`paper/provenance/before_final_reeval/` preserves the old values for audit purposes only.

## 5. Regeneration order

```bash
python conference_package/scripts/audit_near_duplicates.py   # -> provenance/*.json
python conference_package/scripts/evaluate_screened.py       # -> experiments/.../screened_metrics.json
python conference_package/scripts/make_paper_assets.py       # -> figures/*, tables/*
```

`make_paper_assets.py` reads the JSONs above plus `results/` and rewrites all generated assets.
Re-run it after any change and re-check Table 4 and Fig. 4.
