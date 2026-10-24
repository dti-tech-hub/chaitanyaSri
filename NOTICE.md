# Attribution and third-party material

## Code
Original project code is released under the MIT License (see `LICENSE`).

## Labelled training data
The four-class dataset in `data/dataset/` is derived from two public Roboflow
Universe datasets, both marked **CC BY 4.0** at download time:

1. `emergency-vehicle-crigt`, workspace `jd-s39ap`, version 1
   <https://universe.roboflow.com/jd-s39ap/emergency-vehicle-crigt/dataset/1>
   Original classes: ambulance, fire brigade, police.
2. `vehicle-detc.`, workspace `emergency-vehicle-detection-sabir`, version 1
   <https://universe.roboflow.com/emergency-vehicle-detection-sabir/vehicle-detc./dataset/1>
   Original classes: ambulance, fire-truck, vehicle.

The preparation script records the exact source file for every output image in
`data/dataset/metadata.csv` and the source-level counts in
`data/dataset/dataset_report.json`. Class names were remapped to the four
canonical classes; images were not synthesised. See `docs/DATASETS.md` for
split counts and the attribution requirements of CC BY 4.0.

## Demo footage
`data/raw/ambu-test/` contains externally sourced stock/compilation clips used
only to demonstrate the pipeline. They are **not** part of the training,
validation, or test splits, and they are not relicensed by this project. Their
own licences/terms continue to apply. Pexels-hosted clips retain their Pexels
licence; the compilation clips retain the terms of their original uploaders.

## Pretrained weights
`models/yolo/yolov8n.pt` is the Ultralytics YOLOv8n COCO checkpoint, used only
as the initialisation for fine-tuning. Ultralytics software and pretrained
weights are subject to the Ultralytics licence (AGPL-3.0 by default).
