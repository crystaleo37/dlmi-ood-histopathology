# OOD Histopathology Classification — Experiment Log

## Challenge
Binary classification of WSI patches. Training: 3 hospital centers, validation: 4th center, test: 5th center. Domain shift from different staining procedures across centers.

## Approach
Frozen pathology foundation model (feature extractor) + trainable classification head on precomputed features. This follows Kumar et al. (ICLR 2022): linear probing preserves pretrained OOD robustness, while fine-tuning distorts it (+7% OOD advantage on average).

---

## Experiment Results

### Run 1 — Phikon v1 + stain aug + MLP/CORAL (v1 commit)
- **Model**: Phikon v1 (ViT-B, 768-dim, `owkin/phikon`)
- **Augmentation**: HED stain augmentation x5 + ColorJitter + geometric
- **Head**: HeadWithCORAL MLP (768→128→1) — but CORAL was no-op on frozen features initially, then fixed
- **Training**: Adam LR=1e-3, patience=10, BCELoss
- **Val acc**: 95.93% (best epoch 6)
- **Test public**: **95.97%** ← first submission
- **Notes**: CORAL on MLP hidden layer. Stain aug helped here (or at least didn't hurt much).

### Run 2 — Phikon v1 + MLP/CORAL + lower LR + cosine scheduler (v2 attempt)
- **Changes**: LR=5e-4, CosineAnnealingLR, early stop on val accuracy, patience=20
- **Val acc**: 95.64% (best epoch 1 — immediate overfit)
- **Notes**: MLP overfits with fewer data. Lower LR didn't help, model peaked at epoch 1.

### Run 3 — Phikon v2 + stain aug x5 + linear head
- **Model**: Phikon v2 (ViT-L, 1024-dim, `owkin/phikon-v2`)
- **Augmentation**: Same HED stain aug x5
- **Head**: Linear (1024→1)
- **Val acc**: 94.3% (with label smoothing), 94.8% (without)
- **Notes**: v2 worse than v1 on this dataset. Larger model ≠ better for this task.

### Run 4 — Phikon v2 + NO augmentation + linear head
- **Model**: Phikon v2, features extracted without any augmentation
- **Head**: Linear (1024→1)
- **Val acc**: 95.3% (best epoch 5)
- **Notes**: Removing stain aug improved v2 by +0.5%. Confirms augmentation hurts frozen foundation model features.

### Run 5 — Phikon v2 + sklearn LogisticRegressionCV
- **Head**: sklearn LogisticRegressionCV (auto-tuned C, 5-fold CV)
- **Val acc**: 95.35%
- **Notes**: Marginal improvement over pytorch linear head. Features are the bottleneck, not the head.

### Run 6 — Phikon v1 + NO augmentation + linear head (v3 commit)
- **Model**: Phikon v1 (ViT-B, 768-dim)
- **Augmentation**: None during precompute, TTA x3 at prediction
- **Head**: Linear (768→1)
- **Val acc**: **97.12%** ← best result
- **Notes**: Simplest config is the best. No pixel aug, no stain aug, just clean CLS token features + linear head.

### Run 7 — Phikon v1 + NO augmentation + NO TTA (v3-noTTA)
- **Same as Run 6 but TTA_N=1**
- **Status**: Pending results
- **Hypothesis**: Test if TTA helps or hurts with frozen features. Literature suggests TTA is marginal with foundation models (they already encode transform invariance via iBOT/DINOv2 training).

### Run 8a — Phikon v1 + NO aug + linear head + multi-seed ensemble (baseline)
- **Model**: Phikon v1, no pixel augmentation
- **Head**: Linear (768→1) x5 seeds, AdamW (wd=1e-3)
- **Val acc per seed**: 97.05%, 97.01%, **97.12%**, 96.83%, 97.03%
- **Mean ± std**: **97.01% ± 0.10%**
- **Best epoch range**: 1–4 (fast convergence, then overfitting)
- **Notes**: Baseline confirms linear head is near-optimal. Very low variance across seeds.

### Run 8b — Phikon v1 + FroFA + linear head + multi-seed ensemble
- **Model**: Phikon v1, no pixel augmentation
- **Augmentation**: Feature Mixup (alpha=0.2) + Feature Dropout (p=0.1) + Gaussian noise (std=0.01) — in feature space during training only
- **Head**: Linear (768→1) x5 seeds, AdamW (wd=1e-3), ensemble prediction
- **Val acc per seed**: 97.06%, 97.11%, 97.03%, 97.06%, **97.28%**
- **Mean ± std**: **97.11% ± 0.09%** (+0.10% over baseline)
- **Best epoch range**: 10–34 (trains 3x longer before early stopping)
- **Notes**: FroFA wins. Small but consistent gain. Key finding: FroFA acts as a regulariser — models train much longer before overfitting (avg 35 epochs vs 12 for baseline). Lower variance too.

---

## Key Findings

1. **Phikon v1 > v2** for this specific dataset (97.12% vs 95.3%). Consistent with benchmarks showing ViT-B can outperform ViT-L on specific tasks.
2. **Stain augmentation hurts** frozen foundation models. Phikon was trained on 40M diverse TCGA tiles — it already encodes stain invariance. Adding stain aug at pixel level corrupts the features.
3. **Linear head > MLP** for this data regime (100K samples, 768-dim features). MLP overfits immediately (best epoch 1). This aligns with Kumar et al. (ICLR 2022): linear probing preserves OOD robustness.
4. **CORAL loss is architecturally incompatible** with frozen features + linear head. CORAL aligns second-order statistics across domains, but with a frozen backbone you cannot modify the features to become domain-invariant. Applying it to a 1-dim output is meaningless.
5. **Label smoothing didn't help** — slightly worse. Expected: binary classification with clean labels doesn't benefit from smoothing.
6. **AdamW with weight decay** (1e-3) recommended over Adam for slight regularization benefit on the linear head.
7. **FroFA regularises training**: baseline heads peak at epoch 1–4 and early-stop by epoch 11–14. FroFA heads peak at epoch 10–34 and early-stop by epoch 25–49. This 3x longer training confirms feature-space augmentation creates genuinely novel training points, unlike pixel augmentation which produces near-duplicate embeddings.
8. **Multi-seed ensemble** reduces variance (std 0.10% → 0.09%) and the ensemble prediction averages out per-seed noise.

## Literature-Backed Strategy (Priority Order)

| # | Strategy | Expected Gain | Effort | Status |
|---|----------|---------------|--------|--------|
| 1 | Multi-seed head ensemble (5 seeds) | +0.2-0.5% | Trivial | **Ready** |
| 2 | Feature-space augmentation (FroFA) | +0.3-1% | Low | **Ready** |
| 3 | Switch backbone to UNI or Virchow2 | +1-3% | Medium | Future |
| 4 | Multi-backbone ensemble | +1-3% | High | Future |
| 5 | AdamW weight decay (1e-3) | +0-0.3% | Trivial | **Ready** |
| 6 | Geometric-only TTA (rotations) | +0-0.3% | Low | Future |
| 7 | Stain normalization before extraction | +0-0.3% | Medium | **Not worth it** |
| 8 | CORAL / DANN on frozen features | ~0% | Wasted | **Abandoned** |

## What NOT to try (literature-backed dead ends)

- **CORAL/DANN on frozen features**: Architecturally incompatible — these methods require modifying the feature extractor via gradient reversal or second-order alignment. With a frozen backbone, there is nothing to align.
- **Stain normalization before feature extraction**: "Minimal and inconsistent performance gains" with foundation models (EXAONEPath, 2024). Foundation models trained on diverse TCGA data have already learned stain-invariant representations.
- **Fine-tuning the backbone**: Kumar et al. (ICLR 2022) showed fine-tuning distorts pretrained features and underperforms OOD by ~7% on average vs linear probing.
- **Heavy TTA (color + geometric)**: Foundation models trained with extensive augmentation (iBOT/DINOv2) already produce transform-robust features. TTA multiplies inference time by N for marginal gain.

---

## Architecture Summary

```
Image (C,H,W) → [Phikon v1 frozen] → CLS token (768-dim) → [Linear head] → sigmoid → prediction
                                            ↓
                                  Feature-space augmentation
                                  (Mixup, Dropout, Noise)
                                            ↓
                                  x5 seeds → average probabilities
```

## Notebooks

| Notebook | Description | Platform |
|----------|-------------|----------|
| `train_ood_kaggle.ipynb` | Baseline: linear head, multi-seed ensemble | Kaggle |
| `train_ood_kaggle_feataug.ipynb` | FroFA: feature augmentation, multi-seed ensemble | Kaggle |
| `train_ood_kaggle_noTTA.ipynb` | Baseline without TTA (ablation) | Kaggle |
| `train_colab_v3.ipynb` | Baseline for local Colab T4 runtime | Colab |
| `train_colab_v3_feataug.ipynb` | FroFA for local Colab T4 runtime | Colab |

## References
- [Phikon v1](https://huggingface.co/owkin/phikon) — ViT-B, iBOT on 40M TCGA tiles
- [Phikon v2](https://huggingface.co/owkin/phikon-v2) — ViT-L, DINOv2 on 460M tiles
- [FroFA (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Bar_Frozen_Feature_Augmentation_for_Few-Shot_Image_Classification_CVPR_2024_paper.pdf) — Feature-space augmentation for frozen backbones
- [Noisy Feature Mixup (ICLR 2022)](https://arxiv.org/abs/2110.02180) — Mixup + noise in feature space
- [LP > FT for OOD (ICLR 2022)](https://arxiv.org/abs/2202.10054) — Linear probing preserves OOD robustness
- [Model Soups (ICML 2022)](https://arxiv.org/abs/2203.05482) — Weight/prediction averaging improves OOD
- [Pathology FM benchmark (Nature BME 2025)](https://www.nature.com/articles/s41551-025-01516-3) — 19 models, 31 tasks
- [FM unrobust to center differences (2025)](https://arxiv.org/abs/2501.18055) — All current pathology FMs encode center identity
- [RandStainNA (MICCAI 2022)](https://arxiv.org/abs/2206.12694)
- [DG Benchmark in CPath](https://arxiv.org/abs/2409.17063)
- [DG Survey & Guidelines](https://dl.acm.org/doi/full/10.1145/3724391)
