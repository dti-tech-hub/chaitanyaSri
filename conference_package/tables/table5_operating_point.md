# Table 5. Fixed-threshold operating point used by the live pipeline (conf = 0.55, IoU >= 0.50)

Sources: `results/metrics/detection_metrics.json`, `conference_package/experiments/leakage_screened/screened_metrics.json`.

| Split | Images | Instances | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| Full test split | 143 | 226 | 222 | 4 | 4 | 0.982 | 0.982 | 0.982 |
| Leakage-screened | 56 | 133 | 129 | 4 | 4 | 0.970 | 0.970 | 0.970 |
