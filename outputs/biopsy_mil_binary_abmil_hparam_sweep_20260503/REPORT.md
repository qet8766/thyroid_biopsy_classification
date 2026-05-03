# Binary ABMIL Hyperparameter Revisit

Run date: 2026-05-03 UTC

Baseline artifact: `biopsy_mil_binary_abmil_5fold_checkpoints`

Sweep artifact root: `outputs/biopsy_mil_binary_abmil_hparam_sweep_20260503`

Aggregate scoring table: `outputs/biopsy_mil_binary_abmil_hparam_sweep_20260503/aggregate_scoring.csv`

## Scope

All runs used the same binary target, labels CSV, grouped 5-fold split seed
(`2026`), normalization, patch cap, and ABMIL code path as the saved baseline
unless noted. Sweep runs disabled plots and attention export
(`--no-plots --attention-topk 0`) to keep the comparison focused on
hyperparameters.

Tested first-stage knobs:

- checkpoint/threshold metrics: balanced accuracy, F1
- dropout: `0.10`, `0.40`
- learning rate: `0.0002`
- patch cap: train/eval `2048`
- capacity: hidden/attention `512/256` and `128/64`

Tested second-stage combinations:

- hidden/attention `512/256` plus balanced-accuracy or F1 selection
- learning rate `0.0002` plus balanced-accuracy or F1 selection

## Main Results

`reported` means the fold-specific threshold stored by the training run.
`fixed0.5` means a post-hoc fixed 0.5 decision threshold applied consistently
to the saved test predictions.

| setting | decision | AUC | AP | accuracy | balanced accuracy | F1 | recall | specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| saved baseline | reported | 0.9855 | 0.9767 | 0.9370 | 0.9376 | 0.9080 | 0.9391 | 0.9360 |
| saved baseline | fixed0.5 | 0.9855 | 0.9767 | 0.9411 | 0.9381 | 0.9122 | 0.9295 | 0.9467 |
| hidden512_attn256 | reported | 0.9848 | 0.9747 | 0.9495 | 0.9429 | 0.9237 | 0.9231 | 0.9626 |
| lr_2e-4 | reported | 0.9872 | 0.9787 | 0.9527 | 0.9386 | 0.9258 | 0.8975 | 0.9797 |
| lr_2e-4 | fixed0.5 | 0.9872 | 0.9787 | 0.9474 | 0.9428 | 0.9208 | 0.9294 | 0.9561 |
| lr_2e-4_sel_f1 | fixed0.5 | 0.9875 | 0.9786 | 0.9495 | 0.9485 | 0.9256 | 0.9456 | 0.9515 |

## Readout

Best reported thresholded run:

- `hidden512_attn256`
- Change: `--hidden-dim 512 --attention-dim 256`
- Kept: `lr=0.0001`, `dropout=0.25`, `weight_decay=0.0001`,
  train/eval patch cap `1024`, ROC-AUC checkpoint selection, validation
  balanced-accuracy thresholding
- Compared with the saved baseline, this improves accuracy, balanced accuracy,
  F1, and specificity. It gives up some recall.

Best practical ranking run:

- `lr_2e-4`
- Change: `--lr 0.0002`
- This has the best average precision and near-best ROC AUC. The absolute
  highest ROC AUC was `lr_2e-4_sel_f1` by a small margin (`0.9875` vs
  `0.9872`), but its reported thresholded metrics were weaker. With the
  reported validation-selected thresholds, plain `lr_2e-4` is
  specificity-heavy. With a fixed 0.5 threshold it is a more balanced tradeoff.

Best retrospective fixed-threshold result:

- `lr_2e-4_sel_f1` scored at fixed threshold `0.5`
- This had the highest fixed-threshold balanced accuracy in the sweep, but it
  was not chosen by validation fixed-threshold metrics, so treat it as a
  follow-up candidate rather than the default recommendation.

## Recommendation

Use `hidden512_attn256` if the goal is to improve the current saved run under
the existing validation-threshold protocol. It is the cleanest thresholded
hyperparameter improvement.

Use `lr_2e-4` if the goal is ranking quality or a conservative,
specificity-heavy operating point. If using this model operationally, evaluate
both the stored validation-selected thresholds and a fixed 0.5 threshold on a
locked validation set before choosing the decision rule.

Do not adopt the 2048 patch cap or the smaller `128/64` model from this sweep.
Neither improved the headline metrics. Combining the winning single knobs with
balanced-accuracy/F1 checkpoint selection also overfit validation and did not
improve held-out fold means.

## Re-run Commands

Full artifact run for the thresholded recommendation:

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

Full artifact run for the ranking recommendation:

```bash
.venv/bin/python scripts/train_biopsy_mil.py \
  --labels-csv /mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv \
  --output-dir outputs/biopsy_mil_binary_abmil_lr2e4_final \
  --models abmil \
  --folds 5 \
  --epochs 20 \
  --patience 5 \
  --dropout 0.25 \
  --lr 0.0002 \
  --weight-decay 0.0001 \
  --max-patches 1024 \
  --eval-max-patches 1024 \
  --normalizer-max-patches 2048 \
  --attention-topk 20 \
  --cache-in-memory
```
