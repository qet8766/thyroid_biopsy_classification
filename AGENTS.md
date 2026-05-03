# Repository Agent Notes

## Default Binary Biopsy MIL Setting

Use `hidden512_attn256` as the default ABMIL configuration for binary biopsy
carcinoma classification (`V/VI` positive vs `I/II/III/IV` negative).

This replaces the earlier baseline capacity:

- old baseline: `--hidden-dim 256 --attention-dim 128`
- default now: `--hidden-dim 512 --attention-dim 256`

Keep the rest of the baseline training protocol unless there is a specific
reason to run a new sweep:

- grouped 5-fold CV with `--seed 2026`
- `--selection-metric roc_auc`
- `--threshold-metric balanced_accuracy`
- `--dropout 0.25`
- `--lr 0.0001`
- `--weight-decay 0.0001`
- `--max-patches 1024`
- `--eval-max-patches 1024`
- `--normalizer-max-patches 2048`
- `--cache-in-memory`

The sweep supporting this default is documented in:

```text
outputs/biopsy_mil_binary_abmil_hparam_sweep_20260503/REPORT.md
outputs/biopsy_mil_binary_abmil_hparam_sweep_20260503/aggregate_scoring.csv
```

Compared with `biopsy_mil_binary_abmil_5fold_checkpoints`, the
`hidden512_attn256` run improved thresholded accuracy, balanced accuracy, F1,
precision, and specificity, while giving up some recall and not improving AUC.

## Recommended Command

```bash
.venv/bin/python scripts/train_biopsy_mil.py \
  --labels-csv /mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv \
  --output-dir outputs/biopsy_mil_binary_abmil_hidden512_final \
  --models abmil \
  --folds 5 \
  --epochs 20 \
  --patience 5 \
  --hidden-dim 512 \
  --attention-dim 256 \
  --dropout 0.25 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --max-patches 1024 \
  --eval-max-patches 1024 \
  --normalizer-max-patches 2048 \
  --attention-topk 20 \
  --cache-in-memory
```

Use `--attention-topk 0 --no-plots` only for exploratory sweeps where full
artifacts are not needed.

## Custom Binary Targets

For non-default binary tasks, use `--binary-positive-labels` for exact Bethesda
labels and `--binary-positive-majors` for major-label groups. Example: to treat
exact `IIIb` plus the full `IV` group (`IV`, `IVa`, `IVb`, `IVc`, `IVd`) as
positive and all other known labels as negative:

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

## Metric Tradeoff

Five-fold reported-threshold means:

| setting | AUC | AP | accuracy | balanced accuracy | F1 | recall | specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `256/128` | 0.9855 | 0.9767 | 0.9370 | 0.9376 | 0.9080 | 0.9391 | 0.9360 |
| default `512/256` | 0.9848 | 0.9747 | 0.9495 | 0.9429 | 0.9237 | 0.9231 | 0.9626 |

For ranking-oriented experiments rather than thresholded classification,
`--lr 0.0002` is a useful follow-up candidate because it improved AUC/AP in the
2026-05-03 sweep, but it is not the default thresholded classifier setting.
