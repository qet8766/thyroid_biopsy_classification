# FastPATH UNI Inference

Extract UNI embeddings from level-0 H&E FastPATH patches that contain at least
10% tissue according to the existing `plugin_results/tis_seg.json` tissue
segmentation output.

FastPATH source tiles are `512 x 512`. The runner splits each source tile into
four `256 x 256` patches, filters those patches by tissue fraction, then resizes
each selected patch to the UNI model input size (`224 x 224`) for inference.

The script skips slides whose `.fastpath` directory name contains `__IHC` and,
by default, skips any slide under an `ignore` directory.

## Setup

```bash
export HF_TOKEN=your_huggingface_token
```

The token must have access to `MahmoodLab/UNI`.

Required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

## Dry Run

Count eligible tiles without loading UNI:

```bash
python3 fastpath_uni_infer.py \
  --input-root /mnt/fastpath_d/thyroid \
  --output-dir outputs/uni_thyroid_patch256 \
  --dry-run \
  --limit 5
```

## Run Inference

```bash
python3 fastpath_uni_infer.py \
  --input-root /mnt/fastpath_d/thyroid \
  --output-dir outputs/uni_thyroid_patch256 \
  --case-list "/mnt/fastpath_d/thyroid/biopsy list.txt" \
  --batch-size 128 \
  --decode-workers 8 \
  --patch-size 256 \
  --output-dtype float16 \
  --preprocess gpu \
  --prefetch-batches 1
```

Each completed slide writes one `.npz` file containing:

- `embeddings`: `float16` or `float32`, shape `[n_patches, embedding_dim]`
- `tile_rc`: `int32`, shape `[n_patches, 2]`, row and column on the 256-patch grid
- `tile_xy`: `int32`, shape `[n_patches, 2]`, level-0 pixel x and y of the 256 patch
- `source_tile_rc`: `int32`, shape `[n_patches, 2]`, parent 512 FastPATH tile row and column
- `subtile_rc`: `int32`, shape `[n_patches, 2]`, quadrant row and column inside the parent tile
- `tissue_fraction`: `float32`, approximate tissue fraction per 256 patch
- `tile_size` / `patch_size`: scalar `int32`, the inference patch size (`256`)
- `source_tile_size`: scalar `int32`, the FastPATH source tile size (`512`)
- `model_input_size`: scalar `int32`, the UNI tensor size (`224`)

`manifest.csv` records slide-level status, tile counts, output paths, and errors.

`--case-list` is optional. When supplied, it should be a text file with one
case/patient ID or relative path prefix per line. The runner processes only
matching H&E non-IHC FastPATH slides.

`--preprocess gpu` uses OpenCV JPEG decode plus batched tensor resize/normalize
and is the fast path. Use `--preprocess pil` to use the exact timm/PIL transform
path for comparison or debugging.

`--prefetch-batches 1` overlaps CPU JPEG decode/I/O for the next batch with GPU
work on the current batch. Use `--prefetch-batches 0` for profiler comparisons.

## Profiling

Nsight Systems traces from the current optimization pass are under `outputs/nsys*`.
The final prefetch-enabled trace is:

```text
outputs/nsys_prefetch/uni_gpu_limit5_prefetch_sync.nsys-rep
outputs/nsys_prefetch/uni_gpu_limit5_prefetch_sync.sqlite
```

Reproduce the focused profile:

```bash
nsys profile \
  --force-overwrite=true \
  --stats=false \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --python-sampling=true \
  --python-sampling-frequency=500 \
  --pytorch=functions-trace \
  --output=outputs/nsys_prefetch/uni_gpu_limit5_prefetch_sync \
  python3 -u fastpath_uni_infer.py \
    --input-root /mnt/fastpath_d/thyroid \
    --output-dir outputs/nsys_prefetch_run_limit5_patch256 \
    --limit 5 \
    --batch-size 128 \
    --decode-workers 8 \
    --patch-size 256 \
    --output-dtype float16 \
    --preprocess gpu \
    --prefetch-batches 1 \
    --profile-sync \
    --overwrite
```

`--profile-sync` is for profiler attribution only; omit it for production runs.

## Biopsy MIL Classification

`scripts/train_biopsy_mil.py` trains slide-level binary carcinoma classifiers
from the existing UNI `.npz` embedding bags and `biopsy_embedding_labels.csv`.
The binary target is:

- positive: label major `V` or `VI`
- negative: label major `I`, `II`, `III`, or `IV`

Rows with non-binary labels and empty embedding bags are excluded. Cross
validation is grouped by `case_id`, so slides from the same case stay in one
split.

The current environment is externally managed, so use a local venv if `pip`
refuses to install into system Python:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install scikit-learn
```

Run the ABMIL baseline and a mean-pool MIL comparison:

```bash
.venv/bin/python scripts/train_biopsy_mil.py \
  --labels-csv /mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv \
  --output-dir outputs/biopsy_mil_abmil \
  --models abmil meanpool \
  --folds 5 \
  --epochs 12 \
  --patience 4 \
  --max-patches 1024 \
  --eval-max-patches 1024 \
  --normalizer-max-patches 2048 \
  --attention-topk 20 \
  --cache-in-memory
```

Key outputs:

- `metrics_summary.csv` and `metrics_by_fold.csv`
- `test_predictions_all.csv`
- `roc.png` and `precision_recall.png`
- `{model}/fold_{n}/best_model.pt`
- `abmil/fold_{n}/test_attention_top_patches.csv`

For the six-class Bethesda major-label task (`I` through `VI`), use the same
settings with `--target-mode major`, for example:

```bash
.venv/bin/python scripts/train_biopsy_mil.py \
  --labels-csv /mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv \
  --output-dir outputs/biopsy_mil_major \
  --target-mode major \
  --models abmil meanpool \
  --folds 5 \
  --epochs 12 \
  --patience 4 \
  --max-patches 1024 \
  --eval-max-patches 1024 \
  --normalizer-max-patches 2048 \
  --attention-topk 20 \
  --cache-in-memory
```

Multiclass metrics use macro one-vs-rest ROC AUC, macro average precision,
macro balanced accuracy/recall, macro F1, and macro specificity. ROC/PR plots
are exported only for the binary task.

To train on a subset of multiclass labels, pass `--target-classes` in the
desired class order, for example `--target-mode major --target-classes II III
IV V VI`.

For imbalanced multiclass runs, validation-only class-bias calibration can
improve macro metrics without using test labels:

```bash
.venv/bin/python scripts/train_biopsy_mil.py \
  --labels-csv /mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv \
  --output-dir outputs/biopsy_mil_major_II_to_VI_abmil_calibrated_f1 \
  --target-mode major \
  --target-classes II III IV V VI \
  --models abmil \
  --folds 5 \
  --epochs 12 \
  --patience 4 \
  --max-patches 1024 \
  --eval-max-patches 1024 \
  --normalizer-max-patches 2048 \
  --attention-topk 20 \
  --cache-in-memory \
  --selection-metric f1 \
  --calibrate-class-bias \
  --calibration-metric f1
```
