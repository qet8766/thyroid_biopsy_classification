# II-to-VI Creative Follow-up Results

Run date: 2026-05-03 UTC

Target classes: `II`, `III`, `IV`, `V`, `VI`

Baseline artifact: `outputs/biopsy_mil_major_II_to_VI_profiled`

New artifacts:

- `outputs/biopsy_mil_major_II_to_VI_creative_posthoc`
- `outputs/biopsy_mil_major_II_to_VI_creative_posthoc_accuracy`
- `outputs/biopsy_mil_major_II_to_VI_creative_posthoc_balacc`
- `outputs/biopsy_mil_major_II_to_VI_feature_baselines`

## Best Tradeoffs

| goal | method | accuracy | balanced accuracy | macro F1 | within-one accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw baseline | raw ABMIL | 0.6949 +/- 0.0382 | 0.4880 +/- 0.0332 | 0.4826 +/- 0.0381 | 0.9513 +/- 0.0102 |
| Macro F1 | ABMIL + validation F1 bias | 0.6240 +/- 0.0799 | 0.5645 +/- 0.0400 | 0.5295 +/- 0.0545 | 0.9354 +/- 0.0132 |
| Exact accuracy | ABMIL + validation accuracy bias | 0.7097 +/- 0.0304 | 0.5193 +/- 0.0370 | 0.5137 +/- 0.0403 | 0.9491 +/- 0.0170 |
| Balanced accuracy | ABMIL + validation balanced-accuracy bias | 0.5636 +/- 0.1098 | 0.5944 +/- 0.0339 | 0.5043 +/- 0.0778 | 0.9290 +/- 0.0220 |
| Balanced tradeoff | ABMIL malignant cascade, balanced-accuracy selected | 0.6505 +/- 0.0673 | 0.5406 +/- 0.0574 | 0.5181 +/- 0.0624 | 0.9481 +/- 0.0204 |
| Ordinal decision | ABMIL ordinal risk, F1 selected | 0.6981 +/- 0.0232 | 0.5227 +/- 0.0408 | 0.5260 +/- 0.0365 | 0.9502 +/- 0.0061 |

## Interesting Re-reads

Collapsing the five-class probabilities back to malignant `V/VI` versus `II/III/IV` gives much higher clinical binary performance:

| method | AUC | AP | accuracy | balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABMIL + validation F1 bias, V/VI binary | 0.9852 +/- 0.0077 | 0.9778 +/- 0.0098 | 0.9439 +/- 0.0161 | 0.9337 +/- 0.0306 | 0.9130 +/- 0.0295 |
| raw ABMIL, V/VI binary | 0.9874 +/- 0.0054 | 0.9803 +/- 0.0073 | 0.9386 +/- 0.0156 | 0.9207 +/- 0.0338 | 0.9016 +/- 0.0322 |

The ordered-label view is also important. Raw ABMIL exact macro F1 is modest, but within-one Bethesda major class accuracy is `0.9513 +/- 0.0102`, and top-2 accuracy is `0.9206 +/- 0.0154`.

Confidence triage gives a useful operating mode. With thresholds selected from validation confidence:

| method | target coverage | actual coverage | accuracy | macro F1 | within-one accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw ABMIL | 60% | 58.3% | 0.8087 | 0.4885 | 0.9870 |
| ABMIL + validation F1 bias | 60% | 60.2% | 0.7306 | 0.5770 | 0.9743 |
| ABMIL + validation F1 bias | 80% | 81.8% | 0.6723 | 0.5578 | 0.9622 |

## Feature Baseline Check

Pooled embedding-statistic models did not beat ABMIL. Best CPU-only feature baseline:

| method | accuracy | balanced accuracy | macro F1 | within-one accuracy |
| --- | ---: | ---: | ---: | ---: |
| SGD logistic-style classifier on mean+std pooled UNI features | 0.6409 +/- 0.0255 | 0.4695 +/- 0.0168 | 0.4638 +/- 0.0181 | 0.8930 +/- 0.0107 |

This suggests the attention model is preserving slide signal that simple mean/std pooling loses.

## Readout

The strongest headline depends on the metric:

- For macro F1: use `ABMIL + validation F1 bias`.
- For exact accuracy: use `ABMIL + validation accuracy bias`.
- For rare-class balanced recall: use `ABMIL + validation balanced-accuracy bias`, accepting the accuracy hit.
- For a less extreme balanced tradeoff: use the balanced-accuracy-selected malignant cascade.
- For clinical malignant screening: collapse to `V/VI` binary and tune the threshold on validation.

All deployable post-hoc methods above tune only on fold validation predictions, then apply fixed decisions to the held-out test fold.
