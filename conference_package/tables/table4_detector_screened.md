# Table 4. Detector performance after removing all near-duplicate test images (56 images)

Source: `conference_package/experiments/leakage_screened/screened_metrics.json`. Same checkpoint, same thresholds (mAP: conf=0.001, IoU=0.60; operating point: conf=0.55, IoU>=0.50).

| Class | Images | Instances | P | R | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|---|
| Overall | - | - | 0.941 | 0.956 | 0.949 | 0.953 | 0.636 |
| ambulance | 33 | 34 | 0.997 | 0.971 | 0.983 | 0.992 | 0.691 |
| fire_engine | 17 | 17 | 1.000 | 0.968 | 0.984 | 0.995 | 0.732 |
| police_vehicle | 10 | 10 | 0.784 | 0.900 | 0.838 | 0.841 | 0.539 |
| normal_vehicle | 21 | 72 | 0.985 | 0.986 | 0.986 | 0.983 | 0.582 |
