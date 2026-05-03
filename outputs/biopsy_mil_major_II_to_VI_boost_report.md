# II-to-VI Metric Boost Experiments

Run date: 2026-05-02 UTC

Target classes: `II`, `III`, `IV`, `V`, `VI`

## Baseline

Baseline artifact: `outputs/biopsy_mil_major_II_to_VI_profiled`

| model | AUC | AP | accuracy | balanced accuracy | macro F1 | macro precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABMIL raw | 0.8850 +/- 0.0179 | 0.5913 +/- 0.0367 | 0.6949 +/- 0.0382 | 0.4880 +/- 0.0332 | 0.4826 +/- 0.0381 | 0.5383 +/- 0.0551 |
| meanpool raw | 0.8377 +/- 0.0247 | 0.5233 +/- 0.0391 | 0.6621 +/- 0.0308 | 0.4647 +/- 0.0422 | 0.4606 +/- 0.0510 | 0.4721 +/- 0.0539 |

## Technique 1: Validation-Only Class-Bias Calibration

This tunes one additive logit bias per class using validation predictions only, then
applies those fixed biases to the fold's test predictions. It does not use test labels.

F1-optimized artifact: `outputs/biopsy_mil_major_II_to_VI_profiled/calibrated_bias_f1`

| model | AUC | AP | accuracy | balanced accuracy | macro F1 | macro precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABMIL + F1 bias | 0.8847 +/- 0.0195 | 0.5851 +/- 0.0373 | 0.6208 +/- 0.0815 | 0.5783 +/- 0.0425 | 0.5330 +/- 0.0542 | 0.5509 +/- 0.0519 |
| meanpool + F1 bias | 0.8362 +/- 0.0248 | 0.5220 +/- 0.0400 | 0.5943 +/- 0.0647 | 0.4736 +/- 0.0395 | 0.4455 +/- 0.0521 | 0.4720 +/- 0.0664 |

Balanced-accuracy-optimized artifact:
`outputs/biopsy_mil_major_II_to_VI_profiled/calibrated_bias_balacc`

| model | accuracy | balanced accuracy | macro F1 |
| --- | ---: | ---: | ---: |
| ABMIL + bal-acc bias | 0.5572 +/- 0.1210 | 0.5830 +/- 0.0441 | 0.4980 +/- 0.0896 |
| meanpool + bal-acc bias | 0.4969 +/- 0.1158 | 0.4829 +/- 0.0506 | 0.4204 +/- 0.0783 |

Best headline tradeoff: `ABMIL + F1 bias`.

ABMIL F1-bias aggregate confusion, rows=true and columns=predicted:

| true \ pred | II | III | IV | V | VI |
| --- | ---: | ---: | ---: | ---: | ---: |
| II | 61 | 31 | 0 | 8 | 0 |
| III | 41 | 244 | 95 | 25 | 5 |
| IV | 4 | 57 | 61 | 0 | 0 |
| V | 5 | 3 | 1 | 20 | 17 |
| VI | 7 | 4 | 2 | 53 | 200 |

## Technique 2: F1 Selection + Built-In F1 Class-Bias Calibration

This reruns ABMIL with validation macro-F1 as the checkpoint-selection metric and
stores fold-specific validation-tuned class biases in `class_bias_calibration.json`.

Artifact: `outputs/biopsy_mil_major_II_to_VI_abmil_calibrated_f1`

| model | AUC | AP | accuracy | balanced accuracy | macro F1 | macro precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABMIL calibrated retrain | 0.8845 +/- 0.0187 | 0.5804 +/- 0.0402 | 0.6357 +/- 0.0902 | 0.5567 +/- 0.0324 | 0.5159 +/- 0.0308 | 0.5284 +/- 0.0287 |

Aggregate confusion, rows=true and columns=predicted:

| true \ pred | II | III | IV | V | VI |
| --- | ---: | ---: | ---: | ---: | ---: |
| II | 60 | 28 | 0 | 12 | 0 |
| III | 48 | 254 | 83 | 15 | 10 |
| IV | 4 | 58 | 60 | 0 | 0 |
| V | 10 | 2 | 1 | 12 | 21 |
| VI | 9 | 3 | 4 | 36 | 214 |

Per-class recall moved as follows:

| class | raw ABMIL recall | calibrated retrain recall | F1-bias post-hoc recall |
| --- | ---: | ---: | ---: |
| II | 0.4800 | 0.6000 | 0.6100 |
| III | 0.8171 | 0.6195 | 0.5951 |
| IV | 0.1803 | 0.4918 | 0.5000 |
| V | 0.0217 | 0.2609 | 0.4348 |
| VI | 0.9398 | 0.8045 | 0.7519 |

## Recommendation

Use `ABMIL + F1 bias` when the goal is macro-F1/macro-balanced performance. It gives the
largest macro-F1 gain and recovers many more `V` cases, with a clear accuracy tradeoff.

Use the raw ABMIL model when overall accuracy is more important than rare-class recall.
