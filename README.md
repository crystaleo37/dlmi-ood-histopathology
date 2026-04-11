# OOD Histopathology Patch Classification

**CentraleSupelec DLMI 2026 -- Kaggle Challenge**

Khelil Tabbane, Amine Soukane

## Task

Binary tumour/non-tumour classification of WSI patches under domain shift.
Train on 3 hospital centres, validate on a 4th, test on a 5th.

## Approach

Frozen Phikon (ViT-B/16, iBOT, 40M TCGA tiles) as feature extractor.
Linear heads trained on precomputed 768-dim CLS embeddings with feature-space augmentation (FroFA: mixup + dropout + noise) and 5-seed ensemble.

## Results

| Configuration | Val Acc (%) |
|---|---|
| DINOv2 + linear probe | 88.29 |
| Phikon + MLP | 97.40 |
| Phikon + linear + FroFA (best seed) | **97.28** |
| Phikon + linear + FroFA (5-seed mean) | 97.11 |

## Repository structure

```
ablation_study.ipynb          Backbone/head/augmentation ablation (Table 1)
train_ood_kaggle.ipynb        Final model: FroFA + 5-seed ensemble + submission
report/
  report.pdf                  Compiled report (MIDL template)
  report.tex                  LaTeX source
  references.bib              Bibliography
  figures/                    Figures (generated from training logs)
```

## Reproducing

1. Upload `train_ood_kaggle.ipynb` to Kaggle with the competition dataset and Phikon weights.
2. Run all cells. Feature extraction takes ~30 min on first run (cached afterwards), head training ~2 min.
3. Download `submission.csv` from `/kaggle/working/`.
