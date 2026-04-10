# OOD Histopathology Patch Classification

**CentraleSupelec -- DLMI 2026 Kaggle Challenge**

Khelil Tabbane, Amine Soukane

## Task

Binary tumour/non-tumour classification of WSI patches under domain shift.
Train on 3 hospital centres, validate on a 4th, test on a 5th.

## Approach

Frozen Phikon (ViT-B/16, iBOT, 40M TCGA tiles) as feature extractor.
Lightweight linear heads trained on precomputed 768-dim CLS token embeddings.
Feature-space augmentation (FroFA: Mixup + Dropout + Noise) and multi-seed ensembling.

## Results

| Configuration | Val Acc (%) |
|---|---|
| DINOv2 + linear probe | 88.29 |
| Phikon + MLP | 97.40 |
| Phikon + linear + no aug | 97.12 |
| Phikon + linear + FroFA (best seed) | **97.28** |
| Phikon + linear + FroFA (5-seed mean) | 97.11 +/- 0.09 |

## Files

- `train_ood_kaggle.ipynb` -- Main notebook: feature extraction, baseline + FroFA training (5 seeds each), visualisations, ensemble prediction.
- `Experiments.md` -- Full experiment log with motivations and literature references.
- `report/` -- LaTeX report (MIDL template).

## How to run

1. Upload `train_ood_kaggle.ipynb` to Kaggle with the competition dataset and Phikon weights as input datasets.
2. Run all cells. Training takes ~2 min on cached features; feature extraction ~30 min on first run.
3. Download `submission.csv` and figure PDFs from `/kaggle/working/`.
