#!/usr/bin/env python3
"""Export raw ABMIL out-of-fold attention as FastPATH heatmap overlays.

The existing FastPATH viewer can render plugin `tileScores` as a heatmap. This
script reconstructs full-bag attention from saved ABMIL fold checkpoints, writes
the 256-pixel UNI patch attention as an explicit 256-cell heatmap grid, and
persists a result under each `.fastpath` directory's `abmil_attention_output/`
folder so the viewer auto-loads the heatmap.

No retraining is performed. Each slide is scored by the fold where it was in
the held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_biopsy_mil import (  # noqa: E402
    EmbeddingStore,
    FeatureNormalizer,
    SlideRecord,
    make_model,
    record_to_tensor,
)


DEFAULT_RUN_DIR = Path("outputs/biopsy_mil_major_II_to_VI_profiled")
DEFAULT_OUTPUT_DIR = Path("outputs/biopsy_mil_major_II_to_VI_attention_heatmaps")
DEFAULT_FASTPATH_ROOT = Path("/mnt/fastpath_d/thyroid/biopsy")
DEFAULT_RESULT_PREFIX = "abmil_attention"
RESULT_FILENAME = "result.json"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.txt"
ATTENTION_OUTPUT_DIRNAME = "abmil_attention_output"
DEFAULT_ATTENTION_SOURCE = "raw_abmil"
LEGACY_PSMA_OUTPUT_DIRNAME = "psma_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fastpath-root", type=Path, default=DEFAULT_FASTPATH_ROOT)
    parser.add_argument("--model-name", default="abmil")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--normalize-percentile",
        type=float,
        default=99.5,
        help="Positive tile-score percentile mapped to heatmap score 1.0.",
    )
    parser.add_argument(
        "--min-visible-score",
        type=float,
        default=0.01,
        help="Scores at or below this are hidden by the current FastPATH heatmap layer.",
    )
    parser.add_argument(
        "--result-prefix",
        default=DEFAULT_RESULT_PREFIX,
        help="Prefix used in local exported artifact names and summaries.",
    )
    parser.add_argument(
        "--fastpath-output-dirname",
        default=ATTENTION_OUTPUT_DIRNAME,
        help="Directory name inside each .fastpath folder for installed viewer results.",
    )
    parser.add_argument(
        "--attention-source",
        default=DEFAULT_ATTENTION_SOURCE,
        help="attentionMeta.source value written to installed result.json files.",
    )
    parser.add_argument(
        "--no-install-fastpath-results",
        action="store_true",
        help="Only write local artifacts; do not write viewer-visible FastPATH result.json files.",
    )
    parser.add_argument(
        "--clear-old-attention-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove older ABMIL attention result runs for the same slide before writing a new one.",
    )
    parser.add_argument(
        "--clear-legacy-psma-attention-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove legacy ABMIL attention runs that were previously written under psma_output. "
            "Real PSMA runs are left untouched."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional slide limit for smoke runs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.rename(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def as_int(value: Any) -> int:
    if isinstance(value, str) and value.strip() == "":
        return 0
    return int(value)


def as_float(value: Any) -> float:
    if isinstance(value, str) and value.strip() == "":
        return 0.0
    return float(value)


def split_row_to_record(row: dict[str, str]) -> SlideRecord:
    binary_label = as_int(row["binary_label"])
    target = as_int(row.get("target", binary_label))
    target_name = row.get("target_name") or ("positive" if target == 1 else "negative")
    return SlideRecord(
        index=as_int(row["index"]),
        embedding_path=row["embedding_path"],
        embedding_file=row["embedding_file"],
        case_id=row["case_id"],
        rel_path=row["rel_path"],
        label=row["label"],
        label_major=row["label_major"],
        binary_label=binary_label,
        target=target,
        target_name=target_name,
        n_patches=as_int(row["n_patches"]),
    )


def load_split_records(path: Path, split_name: str) -> list[SlideRecord]:
    rows: list[SlideRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split_name:
                rows.append(split_row_to_record(row))
    return rows


def load_normalizer(path: Path) -> FeatureNormalizer | None:
    if not path.exists():
        return None
    state = read_json(path)
    return FeatureNormalizer(
        mean=np.asarray(state["mean"], dtype=np.float32),
        std=np.asarray(state["std"], dtype=np.float32),
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def load_fold_model(
    fold_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, argparse.Namespace, list[str], int]:
    checkpoint = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(args, "target_mode"):
        args.target_mode = "binary" if "class_names" not in checkpoint else "major"
    class_names = [str(name) for name in checkpoint.get("class_names") or ["negative", "positive"]]
    input_dim = int(checkpoint["input_dim"])
    output_dim = 1 if args.target_mode == "binary" else len(class_names)
    model = make_model(args, checkpoint["model_name"], input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args, class_names, input_dim


@torch.inference_mode()
def predict_attention(
    model: torch.nn.Module,
    record: SlideRecord,
    store: EmbeddingStore,
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    class_names: list[str],
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str, float]:
    rng = np.random.default_rng(0)
    x = record_to_tensor(
        record,
        store,
        normalizer,
        device,
        max_patches=None,
        training=False,
        rng=rng,
    )
    logits, attention = model(x, return_attention=True)
    logits_np = logits.detach().cpu().numpy()
    if target_mode == "binary":
        prob_pos = float(torch.sigmoid(logits.view(())).detach().cpu().item())
        probs = np.asarray([1.0 - prob_pos, prob_pos], dtype=np.float32)
    else:
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    attention_np = attention.detach().cpu().numpy().astype(np.float32)
    pred = int(np.argmax(probs))
    return logits_np, probs, attention_np, pred, class_names[pred], float(probs[pred])


def positive_percentile(values: np.ndarray, percentile: float) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    if len(positive) == 0:
        return 1.0
    denom = float(np.percentile(positive, percentile))
    if not math.isfinite(denom) or denom <= 0:
        denom = float(positive.max())
    return denom if denom > 0 else 1.0


def normalize_scores(values: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    denom = positive_percentile(values, percentile)
    return np.clip(values / denom, 0.0, 1.0).astype(np.float32), denom


def attention_to_patch_grid(
    tile_rc: np.ndarray,
    attention: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=np.float32)
    for patch_index, (row, col) in enumerate(tile_rc.astype(np.int64)):
        if row < 0 or col < 0 or row >= rows or col >= cols:
            continue
        grid[row, col] = float(attention[patch_index])
    return grid


def heatmap_grid_shape(metadata: dict[str, Any], cell_size: int) -> tuple[int, int]:
    width, height = metadata["dimensions"]
    return int(math.ceil(height / cell_size)), int(math.ceil(width / cell_size))


def fastpath_dir_for_record(record: SlideRecord, fastpath_root: Path) -> Path:
    return fastpath_root / record.rel_path


def load_fastpath_metadata(fastpath_dir: Path) -> dict[str, Any]:
    metadata_path = fastpath_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"FastPATH metadata not found: {metadata_path}")
    return read_json(metadata_path)


def level_info(metadata: dict[str, Any], level: int) -> dict[str, Any]:
    for item in metadata.get("levels", []):
        if int(item.get("level", -1)) == level:
            return item
    raise ValueError(f"metadata is missing level {level}")


def slide_roi(metadata: dict[str, Any]) -> dict[str, float]:
    dimensions = metadata["dimensions"]
    return {"x": 0.0, "y": 0.0, "w": float(dimensions[0]), "h": float(dimensions[1])}


def slide_hash(record: SlideRecord) -> str:
    value = f"{record.rel_path}|{record.embedding_file}|{record.index}".encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:8]


def make_run_id(saved_at: datetime, record: SlideRecord) -> str:
    return f"{saved_at:%Y%m%dT%H%M%SZ}_{slide_hash(record)}"


def remove_old_attention_runs(root: Path, keep_run_id: str, result_prefix: str) -> None:
    if not root.exists():
        return
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name == keep_run_id or run_dir.name.startswith("."):
            continue
        manifest_path = run_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if manifest.get("plugin") == result_prefix:
            shutil.rmtree(run_dir, ignore_errors=True)


def remove_legacy_psma_attention_runs(fastpath_dir: Path, result_prefix: str) -> None:
    root = fastpath_dir / LEGACY_PSMA_OUTPUT_DIRNAME
    if not root.exists():
        return
    remove_old_attention_runs(root, keep_run_id="", result_prefix=result_prefix)
    try:
        next(root.iterdir())
    except StopIteration:
        root.rmdir()
    except OSError:
        pass


def result_payload(
    *,
    record: SlideRecord,
    class_names: list[str],
    probs: np.ndarray,
    pred_name: str,
    pred_prob: float,
    tile_scores: np.ndarray,
    metadata: dict[str, Any],
    normalizer_denom: float,
    normalize_percentile: float,
    fold: int,
    cell_size: int,
    attention_source: str,
) -> dict[str, Any]:
    roi = slide_roi(metadata)
    classification = {class_name: float(probs[idx]) for idx, class_name in enumerate(class_names)}
    classification["predicted_index"] = float(int(np.argmax(probs)))
    true_display_label = record.label if attention_source == "raw_abmil_binary" else record.target_name
    return {
        "success": True,
        "message": (
            "Raw ABMIL out-of-fold attention heatmap "
            f"(fold {fold}, true={true_display_label}, pred={pred_name}, prob={pred_prob:.4f})"
        ),
        "processingTime": 0.0,
        "outputType": "tile_scores",
        "classification": classification,
        "tileScores": tile_scores.tolist(),
        "tileLevel": 0,
        "tileScoreCellSize": cell_size,
        "hasMask": False,
        "hasHeatmap": False,
        "hasImage": False,
        "hasTileScores": True,
        "attentionMeta": {
            "source": attention_source,
            "fold": fold,
            "recordIndex": record.index,
            "embeddingFile": record.embedding_file,
            "caseId": record.case_id,
            "label": record.label,
            "labelMajor": record.label_major,
            "targetName": record.target_name,
            "predName": pred_name,
            "predProb": pred_prob,
            "grid": "uni_patch_attention",
            "normalizePercentile": normalize_percentile,
            "normalizerDenominator": normalizer_denom,
            "fastpathTileSize": int(metadata["tile_size"]),
            "heatmapCellSize": cell_size,
            "scoreRows": int(tile_scores.shape[0]),
            "scoreCols": int(tile_scores.shape[1]),
        },
    }


def install_fastpath_result(
    fastpath_dir: Path,
    run_id: str,
    saved_at: datetime,
    result: dict[str, Any],
    record: SlideRecord,
    result_prefix: str,
    output_dirname: str,
    clear_old: bool,
    clear_legacy: bool,
) -> Path:
    root = fastpath_dir / output_dirname
    root.mkdir(parents=True, exist_ok=True)
    if clear_old:
        remove_old_attention_runs(root, run_id, result_prefix)
    if clear_legacy:
        remove_legacy_psma_attention_runs(fastpath_dir, result_prefix)
    run_dir = root / run_id
    tmp_dir = root / f".{run_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    tmp_dir.mkdir(parents=True)
    saved_at_text = saved_at.isoformat().replace("+00:00", "Z")
    manifest = {
        "schemaVersion": 1,
        "runId": run_id,
        "slideId": record.rel_path,
        "jobId": run_id,
        "plugin": result_prefix,
        "savedAt": saved_at_text,
        "params": {
            "model": "raw_abmil",
            "fold": result.get("attentionMeta", {}).get("fold"),
            "grid": result.get("attentionMeta", {}).get("grid"),
            "heatmapCellSize": result.get("attentionMeta", {}).get("heatmapCellSize"),
            "normalizePercentile": result.get("attentionMeta", {}).get("normalizePercentile"),
        },
        "resultFile": RESULT_FILENAME,
        "summaryFile": SUMMARY_FILENAME,
    }
    write_json(tmp_dir / MANIFEST_FILENAME, manifest)
    write_json(tmp_dir / RESULT_FILENAME, result)
    (tmp_dir / SUMMARY_FILENAME).write_text(
        "\n".join(
            [
        "ABMIL Attention Heatmap",
                f"Run ID: {run_id}",
                f"Saved At: {saved_at_text}",
                f"Slide: {record.rel_path}",
                f"Case: {record.case_id}",
                f"Label: {record.label}",
                f"Major Label: {record.label_major}",
                f"Fold: {result.get('attentionMeta', {}).get('fold')}",
                f"Message: {result.get('message')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tmp_dir.rename(run_dir)
    return run_dir


def save_local_attention_npz(
    output_path: Path,
    *,
    record: SlideRecord,
    fold: int,
    logits: np.ndarray,
    probs: np.ndarray,
    pred: int,
    pred_name: str,
    pred_prob: float,
    attention: np.ndarray,
    attention_norm: np.ndarray,
    patch_grid_raw: np.ndarray,
    patch_grid_norm: np.ndarray,
    metadata: dict[str, Any],
    npz_data: np.lib.npyio.NpzFile,
    normalizer_denom: float,
    cell_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        record_index=np.asarray(record.index, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int16),
        embedding_file=np.asarray(record.embedding_file),
        rel_path=np.asarray(record.rel_path),
        case_id=np.asarray(record.case_id),
        label=np.asarray(record.label),
        label_major=np.asarray(record.label_major),
        target=np.asarray(record.target, dtype=np.int16),
        target_name=np.asarray(record.target_name),
        pred=np.asarray(pred, dtype=np.int16),
        pred_name=np.asarray(pred_name),
        pred_prob=np.asarray(pred_prob, dtype=np.float32),
        logits=logits.astype(np.float32),
        probs=probs.astype(np.float32),
        attention=attention.astype(np.float32),
        attention_norm=attention_norm.astype(np.float32),
        tile_rc=np.asarray(npz_data["tile_rc"], dtype=np.int32),
        tile_xy=np.asarray(npz_data["tile_xy"], dtype=np.int32),
        source_tile_rc=np.asarray(npz_data["source_tile_rc"], dtype=np.int32),
        subtile_rc=np.asarray(npz_data["subtile_rc"], dtype=np.int32),
        tissue_fraction=np.asarray(npz_data["tissue_fraction"], dtype=np.float32),
        fastpath_patch_grid_raw=patch_grid_raw.astype(np.float32),
        fastpath_patch_grid_norm=patch_grid_norm.astype(np.float32),
        fastpath_dimensions=np.asarray(metadata["dimensions"], dtype=np.int32),
        fastpath_tile_size=np.asarray(metadata["tile_size"], dtype=np.int32),
        heatmap_cell_size=np.asarray(cell_size, dtype=np.int32),
        normalizer_denom=np.asarray(normalizer_denom, dtype=np.float32),
    )


def class_prob_columns(class_names: list[str], probs: np.ndarray) -> dict[str, float]:
    return {f"prob_{name}": float(probs[idx]) for idx, name in enumerate(class_names)}


def process_record(
    *,
    record: SlideRecord,
    fold: int,
    model: torch.nn.Module,
    class_names: list[str],
    target_mode: str,
    store: EmbeddingStore,
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    fastpath_root: Path,
    output_dir: Path,
    normalize_percentile: float,
    install_results: bool,
    result_prefix: str,
    fastpath_output_dirname: str,
    attention_source: str,
    clear_old: bool,
    clear_legacy: bool,
    saved_at: datetime,
    overwrite: bool,
) -> dict[str, Any]:
    fastpath_dir = fastpath_dir_for_record(record, fastpath_root)
    metadata = load_fastpath_metadata(fastpath_dir)
    level_info(metadata, 0)
    logits, probs, attention, pred, pred_name, pred_prob = predict_attention(
        model,
        record,
        store,
        normalizer,
        device,
        class_names,
        target_mode,
    )
    with np.load(record.embedding_path) as npz_data:
        cell_size = int(np.asarray(npz_data["patch_size"]).item()) if "patch_size" in npz_data else 256
        rows, cols = heatmap_grid_shape(metadata, cell_size)
        patch_grid_raw = attention_to_patch_grid(
            np.asarray(npz_data["tile_rc"], dtype=np.int32),
            attention,
            rows=rows,
            cols=cols,
        )
        patch_grid_norm, denom = normalize_scores(patch_grid_raw, normalize_percentile)
        attention_norm, patch_denom = normalize_scores(attention, normalize_percentile)
        local_path = output_dir / "slides" / f"{Path(record.embedding_file).stem}_attention.npz"
        if overwrite or not local_path.exists():
            save_local_attention_npz(
                local_path,
                record=record,
                fold=fold,
                logits=logits,
                probs=probs,
                pred=pred,
                pred_name=pred_name,
                pred_prob=pred_prob,
                attention=attention,
                attention_norm=attention_norm,
                patch_grid_raw=patch_grid_raw,
                patch_grid_norm=patch_grid_norm,
                metadata=metadata,
                npz_data=npz_data,
                normalizer_denom=denom,
                cell_size=cell_size,
            )

    run_id = make_run_id(saved_at, record)
    result = result_payload(
        record=record,
        class_names=class_names,
        probs=probs,
        pred_name=pred_name,
        pred_prob=pred_prob,
        tile_scores=patch_grid_norm,
        metadata=metadata,
        normalizer_denom=denom,
        normalize_percentile=normalize_percentile,
        fold=fold,
        cell_size=cell_size,
        attention_source=attention_source,
    )
    installed_dir = ""
    if install_results:
        installed_dir = str(
            install_fastpath_result(
                fastpath_dir,
                run_id,
                saved_at,
                result,
                record,
                result_prefix,
                fastpath_output_dirname,
                clear_old,
                clear_legacy,
            )
        )
    visible_cells = int((patch_grid_norm > 0.01).sum())
    positive_cells = int((patch_grid_raw > 0).sum())
    return {
        "record_index": record.index,
        "fold": fold,
        "embedding_file": record.embedding_file,
        "case_id": record.case_id,
        "rel_path": record.rel_path,
        "fastpath_dir": str(fastpath_dir),
        "local_attention_npz": str(local_path),
        "installed_result_dir": installed_dir,
        "run_id": run_id,
        "label": record.label,
        "label_major": record.label_major,
        "target": record.target,
        "target_name": record.target_name,
        "pred": pred,
        "pred_name": pred_name,
        "pred_prob": pred_prob,
        "correct": int(pred == record.target),
        "n_patches": record.n_patches,
        "heatmap_cell_size": cell_size,
        "heatmap_rows": rows,
        "heatmap_cols": cols,
        "heatmap_positive_cells": positive_cells,
        "heatmap_visible_cells": visible_cells,
        "raw_attention_max": float(attention.max()) if len(attention) else 0.0,
        "raw_cell_attention_max": float(patch_grid_raw.max()) if patch_grid_raw.size else 0.0,
        "cell_normalizer_denom": float(denom),
        "patch_normalizer_denom": float(patch_denom),
        **class_prob_columns(class_names, probs),
    }


def main() -> int:
    args = parse_args()
    if args.folds < 1:
        raise ValueError("--folds must be positive")
    if not 0 < args.normalize_percentile <= 100:
        raise ValueError("--normalize-percentile must be in (0, 100]")

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_results = not args.no_install_fastpath_results
    saved_at = datetime.now(UTC)
    store = EmbeddingStore(cache_in_memory=False)
    summary_rows: list[dict[str, Any]] = []
    class_names_seen: list[str] | None = None
    started = time.time()
    processed = 0

    for fold in range(1, args.folds + 1):
        fold_dir = args.run_dir / args.model_name / f"fold_{fold}"
        model, model_args, class_names, _input_dim = load_fold_model(fold_dir, device)
        if class_names_seen is None:
            class_names_seen = class_names
        elif class_names_seen != class_names:
            raise ValueError("class_names differ across folds")
        normalizer = load_normalizer(fold_dir / "normalizer.json")
        records = load_split_records(fold_dir / "split.csv", "test")
        if args.limit is not None:
            remaining = max(0, args.limit - processed)
            records = records[:remaining]
        if not records:
            continue
        print(
            f"fold {fold}: exporting {len(records)} held-out slides "
            f"on device={device}, target_mode={model_args.target_mode}",
            flush=True,
        )
        for record in tqdm(records, desc=f"fold {fold}", unit="slide"):
            row = process_record(
                record=record,
                fold=fold,
                model=model,
                class_names=class_names,
                target_mode=model_args.target_mode,
                store=store,
                normalizer=normalizer,
                device=device,
                fastpath_root=args.fastpath_root,
                output_dir=args.output_dir,
                normalize_percentile=args.normalize_percentile,
                install_results=install_results,
                result_prefix=args.result_prefix,
                fastpath_output_dirname=args.fastpath_output_dirname,
                attention_source=args.attention_source,
                clear_old=args.clear_old_attention_results,
                clear_legacy=args.clear_legacy_psma_attention_results,
                saved_at=saved_at,
                overwrite=args.overwrite,
            )
            summary_rows.append(row)
            processed += 1
            if args.limit is not None and processed >= args.limit:
                break
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.limit is not None and processed >= args.limit:
            break

    if class_names_seen is None:
        raise RuntimeError("no folds were processed")

    write_csv(args.output_dir / "attention_heatmap_manifest.csv", summary_rows)
    write_json(
        args.output_dir / "run_summary.json",
        {
            "runDir": str(args.run_dir),
            "modelName": args.model_name,
            "folds": args.folds,
            "classNames": class_names_seen,
            "fastpathRoot": str(args.fastpath_root),
            "outputDir": str(args.output_dir),
            "installedFastpathResults": install_results,
            "fastpathResultDirName": args.fastpath_output_dirname,
            "attentionSource": args.attention_source,
            "grid": "uni_patch_attention",
            "normalizePercentile": args.normalize_percentile,
            "processedSlides": len(summary_rows),
            "elapsedSeconds": time.time() - started,
        },
    )
    print(
        f"wrote {len(summary_rows)} attention heatmaps to {args.output_dir} "
        f"in {(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    if install_results:
        print(
            "installed viewer-visible tileScores under each slide's "
            f"{args.fastpath_output_dirname} directory",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
