# Marine Debris Classification

Deep Learning project, group G10.

One network that looks at the sea from four viewpoints (underwater, shore, drone and satellite), recognises which viewpoint a photo comes from, and says whether it contains debris and which kinds. We compare a **baseline CNN trained from scratch** with **MobileNetV2 transfer learning** (ImageNet weights), using the same heads, data and training setup, so only the backbone differs.

The full story (design, data problems, every training run, results and lessons) is in [`Project_Journey_Report.pdf`](Project_Journey_Report.pdf). The slides are in [`Marine Debris Classification.pptx`](Marine%20Debris%20Classification.pptx).

## Model

- **Backbone:** a 4-block CNN (1.42M parameters) or MobileNetV2 (6.18M parameters), followed by global average and max pooling.
- **Heads:** 1 viewpoint head (softmax, 4 outputs) and 4 multi-label debris heads (underwater 16 classes, shore 13, drone 7, satellite 7). Each debris head has one extra "trash present?" neuron.
- **Head size:** bigger heads where there is more training data (shore 512→256→128, underwater 384→192→96, drone 256→128, satellite 128→64).
- **Training:** focal loss, loss masking (each head learns only from its own dataset), balanced batches across the four datasets, AdamW, and a satellite lever that freezes that head when it stops improving. MobileNetV2 trains in two stages: frozen backbone, then fine-tuning from `block_13_expand` with BatchNorm frozen.

## Data

Four public datasets, one per viewpoint (download links in [`Datasets/Dataset_links.txt`](Datasets/Dataset_links.txt)):

| Viewpoint | Dataset | Why |
|---|---|---|
| Underwater | SeaClear (8,610 seabed images) | Real robot and diver footage with precise outlines |
| Shore | BePLi v2 (3,722 beach images) | Richest annotated beach-plastic set |
| Drone | UCWD (3,169 coastal images kept) | Coastal drone imagery |
| Satellite | MADOS (Sentinel-2 patches) | Labels debris separately from look-alikes (algae, foam, ships) |

All four were built for detection, so `1_prepare_data.py` turns them into classification tiles: it merges rare classes, cuts photos into tiles (underwater 3×2, shore 3×3, drone ~640 px, satellite whole), labels each tile from the original outlines, and splits by video or scene so near-identical frames never end up in both training and test. The raw datasets and tiles are not in this repository (tens of GB). The tile lists (`Datasets/prepared/manifests`) and class list are included.

## Results (test set)

Macro scores per head, in %. Thresholds are tuned on validation and applied once to test.

| Head | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Underwater | CNN | 89.7 | 18.5 | 42.2 | 21.8 | 84.2 |
| | MobileNetV2 | 93.7 | 29.7 | 41.3 | 27.1 | 84.7 |
| Shore | CNN | 82.4 | 24.2 | 52.9 | 31.5 | 79.8 |
| | MobileNetV2 | 84.0 | 24.9 | 49.0 | 31.9 | 80.1 |
| Drone | CNN | 97.7 | 61.3 | 56.3 | 58.0 | 95.6 |
| | MobileNetV2 | 97.9 | 67.9 | 55.1 | 60.0 | 95.8 |
| Satellite | CNN | 89.0 | 53.8 | 62.0 | 54.9 | 85.5 |
| | MobileNetV2 | 86.9 | 46.2 | 50.2 | 45.2 | 80.0 |

"Is there trash in this tile?" (the trash neuron):

| Head | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Underwater | CNN | 72.7 | 60.6 | 80.3 | 69.1 | 81.7 |
| | MobileNetV2 | 79.4 | 69.2 | 81.9 | 75.1 | 87.8 |
| Shore | CNN | 81.0 | 79.5 | 94.4 | 86.3 | 90.0 |
| | MobileNetV2 | 79.7 | 78.5 | 93.7 | 85.4 | 89.3 |
| Drone | CNN | 95.7 | 87.5 | 91.3 | 89.4 | 98.7 |
| | MobileNetV2 | 95.4 | 86.7 | 90.9 | 88.7 | 98.4 |
| Satellite | CNN | 82.7 | 70.1 | 66.7 | 68.3 | 86.8 |
| | MobileNetV2 | 77.1 | 59.2 | 58.3 | 58.8 | 79.8 |

Viewpoint accuracy: CNN 99.5%, MobileNetV2 99.6%. Confusion matrices and ROC curves are in `Models/Evaluations/`.

**Verdict:** MobileNetV2 wins underwater, the CNN wins satellite, and shore and drone are a tie. The biggest gain in the project came from fixing the tile labels (underwater trash ranking +17–20 points), not from the architecture: the remaining limits come from the data.

## Repository layout

```
Datasets/                     dataset links, data spec sheet, label-problem note, tile lists (prepared/)
Models/                       training logs and histories (model files: see Releases)
  Evaluations/eval_<model>/   test results, thresholds, confusion matrices, ROC curves
Notebook training codes/
  1_prepare_data.py           datasets -> labelled tiles and leak-free splits
  2_train_cnn.ipynb           baseline CNN
  3_train_mobilenetv2.ipynb   MobileNetV2 transfer learning
  4_continue_cnn.ipynb        continues the CNN from its best checkpoint
  5_evaluate.ipynb            all metrics, confusion matrices, ROC curves
webapp/app.py                 upload a photo and see what the model finds
```

Final models: `cnn_adamw_v3_long_best.keras` (baseline CNN) and `mnv2_adamw_v2_best.keras` (MobileNetV2). `cnn_adamw_v3_best.keras` is the CNN at 60 epochs, which `4_continue_cnn` starts from.

## How to run

1. Install Python 3.10 or newer, PyTorch with CUDA (see the comment in `requirements.txt`), then `pip install -r requirements.txt`.
2. Download the trained models from the repository's **Releases** page and put them in `Models/` (they are too large for a normal git push).
3. **Try the models:** `python webapp/app.py`, then open http://127.0.0.1:7860.
4. **Re-evaluate:** set `MODEL_NAME` in `5_evaluate.ipynb` and run all cells (needs the prepared tiles).
5. **Rebuild everything from scratch:** download the four datasets into `Datasets/`, run `python "Notebook training codes/1_prepare_data.py"`, then run notebooks 2 to 5.

The notebooks and `1_prepare_data.py` have the project folder path at the top. Change it if your copy is somewhere else. The web app finds its files on its own.
