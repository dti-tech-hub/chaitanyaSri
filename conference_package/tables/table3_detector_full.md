# Table 3. Detector performance on the full held-out test split (143 images)

Source: `results/metrics/detection_metrics.json`. This split contains near-duplicate frames of training images (see Table 4 and Fig. 3b).

| Class | Images | Instances | P | R | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|---|
| Overall | - | - | 0.985 | 0.982 | 0.984 | 0.989 | 0.660 |
| ambulance | 72 | 73 | 0.998 | 0.986 | 0.992 | 0.994 | 0.667 |
| fire_engine | 20 | 20 | 1.000 | 0.973 | 0.987 | 0.995 | 0.738 |
| police_vehicle | 57 | 57 | 0.957 | 0.982 | 0.969 | 0.984 | 0.659 |
| normal_vehicle | 23 | 76 | 0.986 | 0.987 | 0.987 | 0.982 | 0.579 |
