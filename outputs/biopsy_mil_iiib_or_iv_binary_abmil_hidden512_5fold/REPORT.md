# IIIB or IV Binary ABMIL 5-Fold Report

## Objective

Run the same biopsy MIL binary test with `IIIb` and major label `IV` treated as
the positive class. All other known Bethesda labels are treated as negative.

I interpreted `IIIB and IV` as:

- positive exact labels: `IIIb`
- positive major labels: `IV`, which includes `IV`, `IVa`, `IVb`, `IVc`, and
  `IVd`
- negative labels: all other known labels, including `V` and `VI`

## Dataset

- raw rows: 970
- used rows: 951
- unique cases: 775
- positives: 370
- negatives: 581
- skipped rows: 2 non-target labels, 17 below `--min-patches`

Positive label counts:

| label | count |
| --- | ---: |
| IIIb | 248 |
| IV | 33 |
| IVa | 65 |
| IVb | 11 |
| IVc | 11 |
| IVd | 2 |

## Command

```bash
.venv/bin/python scripts/train_biopsy_mil.py \
  --labels-csv /mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv \
  --output-dir outputs/biopsy_mil_iiib_or_iv_binary_abmil_hidden512_5fold \
  --target-mode binary \
  --binary-positive-labels IIIB \
  --binary-positive-majors IV \
  --models abmil \
  --folds 5 \
  --epochs 20 \
  --patience 5 \
  --dropout 0.25 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --max-patches 1024 \
  --eval-max-patches 1024 \
  --normalizer-max-patches 2048 \
  --attention-topk 20 \
  --cache-in-memory
```

`--hidden-dim 512 --attention-dim 256` are now script defaults.

## Results

Five-fold test metrics using validation-selected balanced-accuracy thresholds:

| metric | mean | std |
| --- | ---: | ---: |
| ROC AUC | 0.9122 | 0.0134 |
| Average precision | 0.8367 | 0.0351 |
| Accuracy | 0.8391 | 0.0298 |
| Balanced accuracy | 0.8545 | 0.0258 |
| F1 | 0.8175 | 0.0284 |
| Precision | 0.7373 | 0.0503 |
| Recall / sensitivity | 0.9242 | 0.0694 |
| Specificity | 0.7849 | 0.0705 |

Aggregate thresholded test confusion counts across the five folds:

| TN | FP | FN | TP |
| ---: | ---: | ---: | ---: |
| 456 | 125 | 28 | 342 |

Fold-level test summary:

| fold | best epoch | threshold | AUC | AP | balanced accuracy | recall | specificity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.2120 | 0.9034 | 0.8193 | 0.8551 | 0.9324 | 0.7778 |
| 2 | 3 | 0.4959 | 0.9347 | 0.8942 | 0.8813 | 0.9178 | 0.8448 |
| 3 | 3 | 0.6931 | 0.9026 | 0.8111 | 0.8278 | 0.8108 | 0.8448 |
| 4 | 8 | 0.5000 | 0.9143 | 0.8464 | 0.8789 | 0.9733 | 0.7845 |
| 5 | 1 | 0.4679 | 0.9060 | 0.8128 | 0.8295 | 0.9865 | 0.6724 |

## Artifacts

- `metrics_summary.csv`: five-fold mean/std metrics
- `metrics_by_fold.csv`: fold-level metrics and selected thresholds
- `test_predictions_all.csv`: thresholded test predictions with per-row
  threshold column
- `roc.png` and `precision_recall.png`: test-set curves
- `abmil/fold_*/best_model.pt`: fold checkpoints
- `abmil/fold_*/test_attention_top_patches.csv`: top attention patches
