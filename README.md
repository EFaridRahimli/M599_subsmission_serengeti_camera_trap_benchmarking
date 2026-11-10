# Serengeti Camera-Trap Benchmark

Compares VGG-19, ResNet-50 and MobileNetV2 (frozen ImageNet backbones, new classifier head) for classifying animal species in Snapshot Serengeti Season 8 camera-trap images.

## Results

Validation set: 1,867 images, 16 classes, 20 epochs per model. Final-epoch numbers come from the `*_log.csv` files. Inference time is the notebook's batch-of-32 timing on the author's GPU, averaged over 10 batches.

| Model | Val. accuracy | Val. loss | Inference (ms/image) |
|---|---|---|---|
| ResNet-50 | 81.09% | 0.6535 | 6.12 |
| VGG-19 | 81.04% | 0.6830 | 7.53 |
| MobileNetV2 | 78.36% | 0.7315 | 5.23 |

- ResNet-50 is the best model: highest accuracy, lowest loss, and faster than VGG-19.
- MobileNetV2 is the fastest but gives up about 2.7 points of accuracy.
- ResNet-50 per-class results are in the notebook's classification report. Weakest classes by F1: warthog (0.68), wildebeest (0.72), hyena (0.76), empty (0.78).

| Accuracy comparison | Per-class accuracy (ResNet-50) |
|---|---|
| ![Accuracy comparison](imgs/accuracy_comparison.png) | ![Per-class accuracy](imgs/per_class_accuracy.png) |

![Confusion matrix (ResNet-50)](imgs/confusion_matrix.png)

## Problem

Camera traps produce far more images than people can label. This project asks which standard CNN backbone gives the best trade-off between accuracy and inference speed for species classification, including a lightweight option for edge devices.

## Data

- Source: Snapshot Serengeti, Season 8 (LILA BC public dataset).
- Only daytime images are used. `daytime_training_list.txt` lists 21,421 image paths to download.
- Species labels come from the dataset's metadata JSON. The notebook maps them to 16 classes (for example male, female and cub lion all become `lion`).
- After filtering, the split is 80/10/10 (seed 42). Training has 14,980 images and validation has 1,867. The test split is created but not evaluated.

## Approach

- Transfer learning: ImageNet-pretrained backbone, frozen, because the dataset is small relative to the backbones and it keeps training cheap.
- Same head for all three models (global average pooling, Dense 256 ReLU, Dropout 0.5, softmax), so differences come from the backbone.
- Same training setup for all: Adam, categorical cross-entropy, 224x224 inputs, batch size 32, 20 epochs, each model's own `preprocess_input`.
- Evaluated on a fixed validation split with accuracy, per-class precision/recall/F1 and a confusion matrix.
- Latency measured per image on the same validation batches for each model.

## Repo contents

| Path | Description |
|---|---|
| `benchmarking.ipynb` | Label mapping, split, training, evaluation and timing for all three models |
| `daytime_training_list.txt` | Daytime image paths used by the notebook (relative to the image root; copy to `data2/` before running) |
| `vgg19_training_log.csv`, `resnet50_baseline_log.csv`, `mobilenet_baseline_log.csv` | Per-epoch training logs |
| `imgs/` | Result charts and model architecture diagrams |

## How to run

This is notebook-only; there is no script or requirements file.

1. Get the data. The notebook expects these paths:
   - Images: download the Season 8 image zip file(s) from LILA BC that contain the images in `daytime_training_list.txt`. The repo does not confirm which part zip(s) hold them all; the earlier `part1` name was not verified. Unzip them into `data2/snapshotserengeti-unzipped/`, so that paths such as `data2/snapshotserengeti-unzipped/S8/P06/P06_R3/S8_P06_R3_IMAG1151.JPG` exist. If the archive extracts into an extra top-level folder, move its contents so the `S8/` folder sits directly under `snapshotserengeti-unzipped/`.
   - Image list: copy `daytime_training_list.txt` into `data2/` (the notebook reads `data2/daytime_training_list.txt`). It has 21,421 lines, one image path each, relative to `snapshotserengeti-unzipped/`.
   - Metadata: download https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip, unzip it, and rename the extracted `.json` file to `metadata.json` in the repo root (the notebook reads `metadata.json`).

2. Install packages and open the notebook:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn tqdm split-folders jupyter
jupyter notebook benchmarking.ipynb
```

3. The first cell does not import everything the notebook uses. Before running, add: `import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns` and `from tensorflow.keras import models, layers, callbacks`.

The download and unzip layout above was not re-run when this README was written; it is inferred from the notebook's paths.

## Limitations and next steps

- Backbones are frozen; fine-tuning the top layers would likely improve accuracy.
- Single run with one seed and no test-set evaluation, so the 0.05-point gap between ResNet-50 and VGG-19 is not meaningful.
- Some classes have very few images (leopard has 3 validation images, cheetah 25), so their scores are unreliable. Class balance is uneven.
- Latency was measured on one GPU with Keras `predict`; edge-device speed was not tested.

Coursework for M599.
