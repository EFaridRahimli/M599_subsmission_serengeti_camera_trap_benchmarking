# Snapshot Serengeti Wildlife Classification

This project implements a **deep learning benchmarking framework** to automatically classify animal species from the **Snapshot Serengeti (Season 8)** dataset. The study compares three distinct **Convolutional Neural Network (CNN)** architectures—**VGG-19**, **ResNet-50**, and **MobileNetV2**—to evaluate their **classification performance** and **computational efficiency** for ecological monitoring and conservation use cases.

---

## Performance Summary

Based on experimental results, **MobileNetV2** and **ResNet-50** achieved the highest validation accuracies, while **VGG-19** served as a computationally heavy baseline.

| Model        | Accuracy (%) | Parameters | Loss   | Best Use Case |
|--------------|--------------|------------|--------|---------------|
| **MobileNetV2** | **81.09%** | 3.4M  | 0.4987 | Battery-powered edge devices / Real-time inference |
| **ResNet-50**   | 81.04%     | 25M   | 0.5191 | High-stakes wildlife census requiring maximum robustness |
| **VGG-19**      | 78.36%     | 144M  | 0.7315 | Legacy research baseline (not recommended for new projects) |

---

## Getting Started

### 1. Prerequisites

Ensure the following are installed on your system:

- Python 3.9  
- TensorFlow / Keras  
- Microsoft **AzCopy**

---

### 2. Data Acquisition (AzCopy)

To optimize storage usage, this project uses a **targeted download strategy**. Instead of downloading the full **~450 GB** Snapshot Serengeti dataset, only **daytime imagery** is retrieved using a curated file list (`daytime_training_list.txt`), reducing the footprint to approximately **17 GB**.

Metadata can be dowloaded from "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip"

#### Download Command

```bash
# Syntax for downloading specific files listed in your .txt file
azcopy copy \
"https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-v-2-0/SnapshotSerengeti_S08_v2_0_part1.zip" \
"./data2/snapshotserengeti-unzipped" \
--list-from "daytime_training_list.txt"
