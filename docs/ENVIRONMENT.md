# Environment

## Tested machine

| Component | Version / value |
|---|---|
| OS | Windows 10/11 (local workstation) |
| Python | 3.14.5 |
| Torch | 2.14.1+cpu |
| Torchvision | installed as an Ultralytics dependency |
| Ultralytics | 8.4.174 |
| OpenCV | 5.0.0 |
| NumPy | 2.4.4 |
| pandas | 3.0.1 |
| Streamlit | 1.65.0 |
| Matplotlib | 3.10.8 |
| scikit-learn | 1.8.0 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| GPU | none (CPU-only run) |

## Install

```powershell
python -m pip install -r requirements.txt
```

The project also works with newer compatible versions. For an exact replica of
the reported run, use the versions above.

## Optional dependencies

- `lap` — required only for ByteTrack/BoT-SORT. Not installed in the reported
  environment; the built-in IoU tracker is used.
- `jupyter` / `notebook` — only for opening `notebooks/analysis.ipynb`.
- A CUDA build of Torch — optional; `model.device: auto` will detect it.

## Reproducibility notes

- Training on CPU took several hours for 30 epochs at 448 px. The script records
  the actual elapsed time in `results/training/training_metadata.json`.
- Ultralytics may write progress bars with terminal control characters to
  `results/training/train_console.log`. The structured metrics are in
  `results/training/results.csv`.
- Matplotlib uses the `Agg` backend for all generated figures so the pipeline
  works headlessly.
- OpenCV display is optional (`--no-display`); video writing and the dashboard
  do not require a GUI.
