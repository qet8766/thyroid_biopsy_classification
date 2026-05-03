#!/usr/bin/env python3
"""Train slide-level MIL classifiers on UNI biopsy embedding bags.

Binary target definition:
  - carcinoma positive: Bethesda major label V or VI
  - carcinoma negative: Bethesda major label I, II, III, or IV

Custom binary targets can be defined with --binary-positive-labels and/or
--binary-positive-majors. Matching rows are positive and all other known
Bethesda labels are negative.

The script reads only the existing `.npz` embedding files referenced by
`biopsy_embedding_labels.csv`. Splits are grouped by `case_id` so slides from
the same case do not cross train/validation/test boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from tqdm import tqdm


DEFAULT_LABELS_CSV = Path(
    "/mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv"
)
POSITIVE_MAJORS = {"V", "VI"}
NEGATIVE_MAJORS = {"I", "II", "III", "IV"}
MAJOR_LABELS = ["I", "II", "III", "IV", "V", "VI"]
EXACT_LABEL_ORDER = [
    "I",
    "II",
    "III",
    "IIIa",
    "IIIb",
    "IIIc",
    "IIId",
    "IIIe",
    "IV",
    "IVa",
    "IVb",
    "IVc",
    "IVd",
    "V",
    "VI",
]
KNOWN_LABELS = set(EXACT_LABEL_ORDER)
KNOWN_MAJORS = set(MAJOR_LABELS)


@dataclass(frozen=True)
class SlideRecord:
    index: int
    embedding_path: str
    embedding_file: str
    case_id: str
    rel_path: str
    label: str
    label_major: str
    binary_label: int
    target: int
    target_name: str
    n_patches: int


class EmbeddingStore:
    def __init__(self, cache_in_memory: bool = False) -> None:
        self.cache_in_memory = cache_in_memory
        self._cache: dict[str, np.ndarray] = {}

    def load_embeddings(self, path: str) -> np.ndarray:
        if self.cache_in_memory and path in self._cache:
            return self._cache[path]
        with np.load(path) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"{path} embeddings must have shape [patches, dim]")
        if self.cache_in_memory:
            self._cache[path] = embeddings
        return embeddings


class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(
        cls,
        records: list[SlideRecord],
        store: EmbeddingStore,
        max_patches_per_slide: int | None,
        seed: int,
    ) -> "FeatureNormalizer":
        rng = np.random.default_rng(seed)
        total_count = 0
        total_sum: np.ndarray | None = None
        total_sumsq: np.ndarray | None = None

        for record in tqdm(records, desc="fit normalizer", leave=False):
            x = store.load_embeddings(record.embedding_path)
            if max_patches_per_slide is not None and len(x) > max_patches_per_slide:
                keep = rng.choice(len(x), size=max_patches_per_slide, replace=False)
                x = x[keep]
            if len(x) == 0:
                continue
            x64 = x.astype(np.float64, copy=False)
            if total_sum is None:
                total_sum = np.zeros(x64.shape[1], dtype=np.float64)
                total_sumsq = np.zeros(x64.shape[1], dtype=np.float64)
            total_sum += x64.sum(axis=0)
            total_sumsq += np.square(x64).sum(axis=0)
            total_count += x64.shape[0]

        if total_count == 0 or total_sum is None or total_sumsq is None:
            raise ValueError("cannot fit normalizer with zero training patches")
        mean = total_sum / total_count
        var = np.maximum(total_sumsq / total_count - np.square(mean), 1e-6)
        return cls(mean=mean, std=np.sqrt(var))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class ABMIL(nn.Module):
    """Gated attention MIL model from Ilse et al. style ABMIL."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h = self.encoder(x)
        a = self.attention_w(torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h)))
        a = torch.softmax(a.squeeze(-1), dim=0)
        z = torch.sum(h * a.unsqueeze(-1), dim=0)
        logit = self.classifier(z).squeeze(0)
        if return_attention:
            return logit, a
        return logit


class MeanPoolMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h = self.encoder(x)
        z = h.mean(dim=0)
        logit = self.classifier(z).squeeze(0)
        if return_attention:
            attention = torch.full((len(x),), 1.0 / max(len(x), 1), device=x.device)
            return logit, attention
        return logit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_class_names(
    target_mode: str,
    target_classes: list[str] | None = None,
) -> list[str]:
    if target_mode == "binary":
        class_names = ["negative", "positive"]
    elif target_mode == "major":
        class_names = MAJOR_LABELS
    elif target_mode == "label":
        class_names = EXACT_LABEL_ORDER
    else:
        raise ValueError(f"unsupported target mode: {target_mode}")

    if target_classes is None:
        return list(class_names)
    if target_mode == "binary":
        raise ValueError("--target-classes is only supported for multiclass target modes")
    if len(set(target_classes)) != len(target_classes):
        raise ValueError("--target-classes contains duplicate class names")
    unknown = sorted(set(target_classes) - set(class_names))
    if unknown:
        raise ValueError(f"--target-classes contains unknown labels for {target_mode}: {unknown}")
    if len(target_classes) < 2:
        raise ValueError("--target-classes must include at least two classes")
    return list(target_classes)


def normalize_target_names(
    values: list[str] | None,
    valid_names: list[str],
    option_name: str,
) -> list[str]:
    if values is None:
        return []
    by_casefold = {name.casefold(): name for name in valid_names}
    normalized: list[str] = []
    for value in values:
        key = value.strip().casefold()
        if not key:
            raise ValueError(f"--{option_name} contains an empty value")
        if key not in by_casefold:
            raise ValueError(
                f"--{option_name} contains unknown label {value!r}; "
                f"valid values are: {', '.join(valid_names)}"
            )
        normalized.append(by_casefold[key])
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"--{option_name} contains duplicate values")
    return normalized


def row_target(
    row: dict[str, str],
    target_mode: str,
    class_names: list[str],
    binary_positive_labels: set[str] | None = None,
    binary_positive_majors: set[str] | None = None,
) -> tuple[int, str, int] | None:
    major = row.get("label_major", "").strip()
    label = row.get("label", "").strip()
    binary_label = 1 if major in POSITIVE_MAJORS else 0

    if target_mode == "binary":
        if binary_positive_labels or binary_positive_majors:
            if label not in KNOWN_LABELS or major not in KNOWN_MAJORS:
                return None
            positive = label in (binary_positive_labels or set()) or major in (
                binary_positive_majors or set()
            )
            target = int(positive)
            return target, class_names[target], target
        if major in POSITIVE_MAJORS:
            return 1, "positive", binary_label
        if major in NEGATIVE_MAJORS:
            return 0, "negative", binary_label
        return None
    if target_mode == "major":
        if major not in class_names:
            return None
        return class_names.index(major), major, binary_label
    if target_mode == "label":
        if label not in class_names:
            return None
        return class_names.index(label), label, binary_label
    raise ValueError(f"unsupported target mode: {target_mode}")


def read_records(
    labels_csv: Path,
    min_patches: int,
    target_mode: str,
    class_names: list[str],
    binary_positive_labels: list[str] | None = None,
    binary_positive_majors: list[str] | None = None,
) -> tuple[list[SlideRecord], dict[str, object]]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {labels_csv}")

    raw_rows = list(csv.DictReader(labels_csv.open("r", encoding="utf-8-sig", newline="")))
    records: list[SlideRecord] = []
    skipped: dict[str, int] = {
        "non_target_label": 0,
        "missing_embedding": 0,
        "below_min_patches": 0,
        "load_error": 0,
    }
    label_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    positive_label_set = set(binary_positive_labels or [])
    positive_major_set = set(binary_positive_majors or [])

    for row_index, row in enumerate(raw_rows):
        target = row_target(
            row,
            target_mode,
            class_names,
            binary_positive_labels=positive_label_set,
            binary_positive_majors=positive_major_set,
        )
        if target is None:
            skipped["non_target_label"] += 1
            continue
        target_id, target_name, binary_label = target
        major = row.get("label_major", "").strip()

        embedding_path = row.get("embedding_path", "").strip()
        if not embedding_path or not Path(embedding_path).exists():
            skipped["missing_embedding"] += 1
            continue

        try:
            with np.load(embedding_path) as data:
                shape = data["embeddings"].shape
        except Exception:
            skipped["load_error"] += 1
            continue
        if len(shape) != 2:
            skipped["load_error"] += 1
            continue
        n_patches = int(shape[0])
        if n_patches < min_patches:
            skipped["below_min_patches"] += 1
            continue

        label = row.get("label", "").strip()
        label_counts[label] = label_counts.get(label, 0) + 1
        target_counts[target_name] = target_counts.get(target_name, 0) + 1
        records.append(
            SlideRecord(
                index=len(records),
                embedding_path=embedding_path,
                embedding_file=row.get("embedding_file", Path(embedding_path).name).strip(),
                case_id=row.get("case_id", "").strip(),
                rel_path=row.get("rel_path", "").strip(),
                label=label,
                label_major=major,
                binary_label=binary_label,
                target=target_id,
                target_name=target_name,
                n_patches=n_patches,
            )
        )

    if not records:
        raise ValueError("no labeled embedding records remain after filtering")

    summary: dict[str, object] = {
        "labels_csv": str(labels_csv),
        "target_mode": target_mode,
        "class_names": class_names,
        "binary_positive_labels": list(binary_positive_labels or []),
        "binary_positive_majors": list(binary_positive_majors or []),
        "raw_rows": len(raw_rows),
        "used_rows": len(records),
        "positive_rows": sum(record.binary_label for record in records),
        "negative_rows": sum(1 - record.binary_label for record in records),
        "unique_cases": len({record.case_id for record in records}),
        "skipped": skipped,
        "label_counts": dict(sorted(label_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "patch_count": {
            "min": int(min(record.n_patches for record in records)),
            "median": float(np.median([record.n_patches for record in records])),
            "mean": float(np.mean([record.n_patches for record in records])),
            "max": int(max(record.n_patches for record in records)),
        },
    }
    return records, summary


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def make_model(
    args: argparse.Namespace,
    model_name: str,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    if model_name == "abmil":
        return ABMIL(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
            output_dim=output_dim,
        )
    if model_name == "meanpool":
        return MeanPoolMIL(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            output_dim=output_dim,
        )
    raise ValueError(f"unsupported model: {model_name}")


def sample_embeddings(
    x: np.ndarray,
    max_patches: int | None,
    training: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_patches is None or len(x) <= max_patches:
        return x
    if training:
        keep = rng.choice(len(x), size=max_patches, replace=False)
        return x[keep]
    keep = np.linspace(0, len(x) - 1, num=max_patches, dtype=np.int64)
    return x[keep]


def record_to_tensor(
    record: SlideRecord,
    store: EmbeddingStore,
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    max_patches: int | None,
    training: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    x = store.load_embeddings(record.embedding_path)
    x = sample_embeddings(x, max_patches=max_patches, training=training, rng=rng)
    if normalizer is not None:
        x = normalizer.apply(x)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device, non_blocking=True)


def iter_shuffled(records: list[SlideRecord], rng: np.random.Generator) -> list[SlideRecord]:
    order = np.arange(len(records))
    rng.shuffle(order)
    return [records[int(index)] for index in order]


def train_one_epoch(
    model: nn.Module,
    records: list[SlideRecord],
    store: EmbeddingStore,
    normalizer: FeatureNormalizer | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_patches: int | None,
    rng: np.random.Generator,
    target_mode: str,
) -> float:
    model.train()
    losses: list[float] = []
    for record in tqdm(iter_shuffled(records, rng), desc="train", leave=False):
        x = record_to_tensor(record, store, normalizer, device, max_patches, True, rng)
        if target_mode == "binary":
            target = torch.tensor(float(record.target), dtype=torch.float32, device=device)
        else:
            target = torch.tensor([record.target], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if target_mode == "binary":
            loss = loss_fn(logits.view(()), target)
        else:
            loss = loss_fn(logits.unsqueeze(0), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


@torch.inference_mode()
def predict_records(
    model: nn.Module,
    records: list[SlideRecord],
    store: EmbeddingStore,
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    max_patches: int | None,
    seed: int,
    target_mode: str,
    class_names: list[str],
) -> list[dict[str, object]]:
    model.eval()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="predict", leave=False):
        x = record_to_tensor(record, store, normalizer, device, max_patches, False, rng)
        logits = model(x)
        row: dict[str, object] = {
            "record_index": record.index,
            "embedding_file": record.embedding_file,
            "embedding_path": record.embedding_path,
            "case_id": record.case_id,
            "rel_path": record.rel_path,
            "label": record.label,
            "label_major": record.label_major,
            "binary_label": record.binary_label,
            "target": record.target,
            "target_name": record.target_name,
            "n_patches": record.n_patches,
        }
        if target_mode == "binary":
            prob = torch.sigmoid(logits).detach().cpu().item()
            pred = int(prob >= 0.5)
            row.update(
                {
                    "pred": pred,
                    "pred_name": class_names[pred],
                    "logit": float(logits.detach().cpu().item()),
                    "prob": float(prob),
                }
            )
        else:
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            pred = int(np.argmax(probs))
            row.update({"pred": pred, "pred_name": class_names[pred], "prob": float(probs[pred])})
            for class_index, class_name in enumerate(class_names):
                safe_name = class_name.replace(" ", "_")
                row[f"logit_{safe_name}"] = float(logits_np[class_index])
                row[f"prob_{safe_name}"] = float(probs[class_index])
        rows.append(row)
    return rows


def apply_binary_threshold(
    rows: list[dict[str, object]],
    threshold: float | None,
    class_names: list[str],
) -> list[dict[str, object]]:
    selected_threshold = 0.5 if threshold is None else float(threshold)
    thresholded_rows: list[dict[str, object]] = []
    for row in rows:
        out = dict(row)
        pred = int(float(out["prob"]) >= selected_threshold)
        out["pred"] = pred
        out["pred_name"] = class_names[pred]
        out["threshold"] = selected_threshold
        thresholded_rows.append(out)
    return thresholded_rows


def safe_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return math.nan
    return float(roc_auc_score(targets, scores))


def safe_ap(targets: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return math.nan
    return float(average_precision_score(targets, scores))


def probability_matrix(rows: list[dict[str, object]], class_names: list[str]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        values.append([float(row[f"prob_{class_name.replace(' ', '_')}"]) for class_name in class_names])
    return np.asarray(values, dtype=float)


def logits_matrix(rows: list[dict[str, object]], class_names: list[str]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        values.append([float(row[f"logit_{class_name.replace(' ', '_')}"]) for class_name in class_names])
    return np.asarray(values, dtype=float)


def softmax_matrix(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_prediction_metric(
    targets: np.ndarray,
    pred: np.ndarray,
    metric: str,
    class_names: list[str],
) -> float:
    labels = np.arange(len(class_names))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(targets, pred))
    if metric == "f1":
        return float(f1_score(targets, pred, labels=labels, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(targets, pred))
    raise ValueError(f"unsupported multiclass prediction metric: {metric}")


def select_class_biases(
    rows: list[dict[str, object]],
    class_names: list[str],
    metric: str,
) -> list[float]:
    """Tune per-class logit biases on validation rows only."""
    targets = np.asarray([int(row["target"]) for row in rows])
    logits = logits_matrix(rows, class_names)
    biases = np.zeros(len(class_names), dtype=float)
    pred = np.argmax(logits + biases, axis=1)
    best_score = multiclass_prediction_metric(targets, pred, metric, class_names)

    for grid in [
        np.linspace(-3.0, 3.0, 25),
        np.linspace(-1.0, 1.0, 21),
        np.linspace(-0.4, 0.4, 17),
        np.linspace(-0.15, 0.15, 13),
    ]:
        improved = True
        rounds = 0
        while improved and rounds < 8:
            improved = False
            rounds += 1
            for class_index in range(len(class_names)):
                current = float(biases[class_index])
                local_best = best_score
                local_biases = biases
                for delta in grid:
                    candidate = biases.copy()
                    candidate[class_index] = current + float(delta)
                    candidate -= candidate.mean()
                    pred = np.argmax(logits + candidate, axis=1)
                    score = multiclass_prediction_metric(targets, pred, metric, class_names)
                    if score > local_best + 1e-12:
                        local_best = score
                        local_biases = candidate
                if local_best > best_score + 1e-12:
                    biases = local_biases
                    best_score = local_best
                    improved = True
    return [float(value) for value in biases]


def apply_class_biases(
    rows: list[dict[str, object]],
    class_names: list[str],
    class_biases: list[float],
) -> list[dict[str, object]]:
    logits = logits_matrix(rows, class_names)
    biases = np.asarray(class_biases, dtype=float)
    adjusted_logits = logits + biases
    probs = softmax_matrix(adjusted_logits)
    pred = np.argmax(adjusted_logits, axis=1)
    calibrated_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        out = dict(row)
        predicted_class = int(pred[row_index])
        out["pred"] = predicted_class
        out["pred_name"] = class_names[predicted_class]
        out["prob"] = float(probs[row_index, predicted_class])
        for class_index, class_name in enumerate(class_names):
            safe_name = class_name.replace(" ", "_")
            out[f"prob_{safe_name}"] = float(probs[row_index, class_index])
        calibrated_rows.append(out)
    return calibrated_rows


def safe_multiclass_auc(targets: np.ndarray, probs: np.ndarray, class_names: list[str]) -> float:
    if len(np.unique(targets)) < 2:
        return math.nan
    for class_index in range(len(class_names)):
        y_true = targets == class_index
        if not y_true.any() or not (~y_true).any():
            return math.nan
    try:
        return float(
            roc_auc_score(
                targets,
                probs,
                labels=np.arange(len(class_names)),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return math.nan


def safe_multiclass_ap(targets: np.ndarray, probs: np.ndarray, class_names: list[str]) -> float:
    values: list[float] = []
    for class_index in range(len(class_names)):
        y_true = targets == class_index
        if y_true.any() and (~y_true).any():
            values.append(float(average_precision_score(y_true.astype(int), probs[:, class_index])))
    return float(np.mean(values)) if values else math.nan


def macro_specificity(targets: np.ndarray, pred: np.ndarray, class_names: list[str]) -> float:
    values: list[float] = []
    for class_index in range(len(class_names)):
        y_true = targets == class_index
        y_pred = pred == class_index
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        if tn + fp:
            values.append(tn / (tn + fp))
    return float(np.mean(values)) if values else math.nan


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.array([0.5])
    candidates = np.unique(scores)
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], candidates)))


def select_threshold(rows: list[dict[str, object]], metric: str) -> float:
    targets = np.asarray([int(row["target"]) for row in rows])
    scores = np.asarray([float(row["prob"]) for row in rows])
    best_threshold = 0.5
    best_score = -math.inf

    for threshold in threshold_candidates(scores):
        pred = (scores >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(targets, pred)
        elif metric == "f1":
            score = f1_score(targets, pred, zero_division=0)
        else:
            raise ValueError(f"unsupported threshold metric: {metric}")
        if score > best_score or (math.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(
    rows: list[dict[str, object]],
    threshold: float | None,
    target_mode: str,
    class_names: list[str],
) -> dict[str, float | int | str]:
    targets = np.asarray([int(row["target"]) for row in rows])
    if target_mode == "binary":
        scores = np.asarray([float(row["prob"]) for row in rows])
        selected_threshold = 0.5 if threshold is None else float(threshold)
        pred = (scores >= selected_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else math.nan
        return {
            "n": int(len(rows)),
            "positive": int(targets.sum()),
            "negative": int((1 - targets).sum()),
            "threshold": float(selected_threshold),
            "roc_auc": safe_auc(targets, scores),
            "average_precision": safe_ap(targets, scores),
            "accuracy": float(accuracy_score(targets, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, pred)),
            "f1": float(f1_score(targets, pred, zero_division=0)),
            "precision": float(precision_score(targets, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(targets, pred, zero_division=0)),
            "specificity": float(specificity),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    probs = probability_matrix(rows, class_names)
    pred = np.asarray([int(row["pred"]) for row in rows])
    matrix = confusion_matrix(targets, pred, labels=np.arange(len(class_names)))
    return {
        "n": int(len(rows)),
        "classes": int(len(class_names)),
        "threshold": math.nan,
        "roc_auc": safe_multiclass_auc(targets, probs, class_names),
        "average_precision": safe_multiclass_ap(targets, probs, class_names),
        "accuracy": float(accuracy_score(targets, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, pred)),
        "f1": float(f1_score(targets, pred, labels=np.arange(len(class_names)), average="macro", zero_division=0)),
        "precision": float(
            precision_score(
                targets,
                pred,
                labels=np.arange(len(class_names)),
                average="macro",
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                targets,
                pred,
                labels=np.arange(len(class_names)),
                average="macro",
                zero_division=0,
            )
        ),
        "specificity": macro_specificity(targets, pred, class_names),
        "f1_weighted": float(
            f1_score(targets, pred, labels=np.arange(len(class_names)), average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                targets,
                pred,
                labels=np.arange(len(class_names)),
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                targets,
                pred,
                labels=np.arange(len(class_names)),
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_json": json.dumps(matrix.tolist()),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_rows(
    train_records: list[SlideRecord],
    val_records: list[SlideRecord],
    test_records: list[SlideRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        for record in records:
            row = asdict(record)
            row["split"] = split_name
            rows.append(row)
    return rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
        handle.write("\n")


def split_train_val(
    train_val_indices: np.ndarray,
    records: list[SlideRecord],
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([records[int(index)].target for index in train_val_indices])
    groups = np.asarray([records[int(index)].case_id for index in train_val_indices])
    inner_splits = max(2, int(round(1.0 / val_fraction)))
    inner_splits = min(inner_splits, len(np.unique(groups)))
    class_counts = np.bincount(labels)
    nonzero_counts = class_counts[class_counts > 0]
    if len(nonzero_counts):
        inner_splits = min(inner_splits, max(2, int(nonzero_counts.min())))
    splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    relative_train, relative_val = next(splitter.split(train_val_indices, labels, groups))
    return train_val_indices[relative_train], train_val_indices[relative_val]


def make_splits(
    records: list[SlideRecord],
    n_splits: int,
    val_fraction: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(len(records))
    labels = np.asarray([record.target for record in records])
    groups = np.asarray([record.case_id for record in records])
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold, (train_val_rel, test_rel) in enumerate(outer.split(indices, labels, groups), start=1):
        train_val_indices = indices[train_val_rel]
        test_indices = indices[test_rel]
        train_indices, val_indices = split_train_val(
            train_val_indices, records, val_fraction, seed=seed + fold * 1000
        )
        splits.append((train_indices, val_indices, test_indices))
    return splits


def infer_input_dim(records: list[SlideRecord]) -> int:
    for record in records:
        with np.load(record.embedding_path) as data:
            if data["embeddings"].shape[0] > 0:
                return int(data["embeddings"].shape[1])
    raise ValueError("could not infer embedding dimension from empty dataset")


def subset(records: list[SlideRecord], indices: Iterable[int]) -> list[SlideRecord]:
    return [records[int(index)] for index in indices]


def export_attention_topk(
    model: nn.Module,
    records: list[SlideRecord],
    prediction_rows: list[dict[str, object]],
    store: EmbeddingStore,
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    topk: int,
    output_path: Path,
    target_mode: str,
    class_names: list[str],
) -> None:
    if topk <= 0:
        return
    pred_by_index = {int(row["record_index"]): row for row in prediction_rows}
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for record in tqdm(records, desc="attention", leave=False):
            x_np = store.load_embeddings(record.embedding_path)
            if len(x_np) == 0:
                continue
            x_eval = x_np
            if normalizer is not None:
                x_eval = normalizer.apply(x_eval)
            x = torch.from_numpy(np.asarray(x_eval, dtype=np.float32)).to(device)
            full_bag_logits, attention = model(x, return_attention=True)
            if target_mode == "binary":
                full_bag_logit_value = float(full_bag_logits.detach().cpu().item())
                full_bag_prob_value = float(torch.sigmoid(full_bag_logits).detach().cpu().item())
                full_bag_pred = int(full_bag_prob_value >= 0.5)
                full_bag_pred_name = class_names[full_bag_pred]
            else:
                full_bag_probs = torch.softmax(full_bag_logits, dim=0).detach().cpu().numpy()
                full_bag_pred = int(np.argmax(full_bag_probs))
                full_bag_logit_value = float(full_bag_logits.detach().cpu().numpy()[full_bag_pred])
                full_bag_prob_value = float(full_bag_probs[full_bag_pred])
                full_bag_pred_name = class_names[full_bag_pred]
            scores = attention.detach().cpu().numpy()
            if len(scores) == 0:
                continue
            keep = np.argsort(scores)[::-1][: min(topk, len(scores))]
            with np.load(record.embedding_path) as data:
                tile_rc = data["tile_rc"] if "tile_rc" in data else None
                tile_xy = data["tile_xy"] if "tile_xy" in data else None
                tissue = data["tissue_fraction"] if "tissue_fraction" in data else None
            pred_row = pred_by_index.get(record.index, {})
            for rank, patch_index in enumerate(keep, start=1):
                row: dict[str, object] = {
                    "record_index": record.index,
                    "embedding_file": record.embedding_file,
                    "case_id": record.case_id,
                    "target": record.target,
                    "target_name": record.target_name,
                    "binary_label": record.binary_label,
                    "pred": pred_row.get("pred", ""),
                    "pred_name": pred_row.get("pred_name", ""),
                    "eval_logit": pred_row.get("logit", ""),
                    "eval_prob": pred_row.get("prob", ""),
                    "full_bag_logit": full_bag_logit_value,
                    "full_bag_prob": full_bag_prob_value,
                    "full_bag_pred": full_bag_pred,
                    "full_bag_pred_name": full_bag_pred_name,
                    "attention_scope": "full_bag",
                    "attention_n_patches": int(len(scores)),
                    "rank": rank,
                    "patch_index": int(patch_index),
                    "attention": float(scores[patch_index]),
                }
                if tile_rc is not None:
                    row["tile_row"] = int(tile_rc[patch_index][0])
                    row["tile_col"] = int(tile_rc[patch_index][1])
                if tile_xy is not None:
                    row["tile_x"] = int(tile_xy[patch_index][0])
                    row["tile_y"] = int(tile_xy[patch_index][1])
                if tissue is not None:
                    row["tissue_fraction"] = float(tissue[patch_index])
                rows.append(row)
    write_csv(output_path, rows)


def save_plot_curves(output_dir: Path, rows_by_model_fold: dict[tuple[str, int], list[dict[str, object]]]) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except Exception as exc:
        print(f"plot export skipped: {exc}", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for curve_name, display_cls in [
        ("roc", RocCurveDisplay),
        ("precision_recall", PrecisionRecallDisplay),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
        for (model_name, fold), rows in sorted(rows_by_model_fold.items()):
            y_true = np.asarray([int(row["target"]) for row in rows])
            y_score = np.asarray([float(row["prob"]) for row in rows])
            if len(np.unique(y_true)) < 2:
                continue
            display_cls.from_predictions(y_true, y_score, name=f"{model_name} fold {fold}", ax=ax)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{curve_name}.png")
        plt.close(fig)


def run_model_fold(
    args: argparse.Namespace,
    model_name: str,
    fold: int,
    train_records: list[SlideRecord],
    val_records: list[SlideRecord],
    test_records: list[SlideRecord],
    input_dim: int,
    device: torch.device,
    store: EmbeddingStore,
    output_dir: Path,
    class_names: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    fold_dir = output_dir / model_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fold_dir / "split.csv", split_rows(train_records, val_records, test_records))

    normalizer = None
    if args.normalize:
        normalizer = FeatureNormalizer.fit(
            train_records,
            store,
            max_patches_per_slide=args.normalizer_max_patches,
            seed=args.seed + fold,
        )
        write_json(fold_dir / "normalizer.json", normalizer.state_dict())

    train_labels = np.asarray([record.target for record in train_records])
    if args.target_mode == "binary":
        n_pos = int(train_labels.sum())
        n_neg = int((1 - train_labels).sum())
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        counts = np.bincount(train_labels, minlength=len(class_names)).astype(np.float32)
        weights = np.zeros(len(class_names), dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = counts.sum() / (float(nonzero.sum()) * counts[nonzero])
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    output_dim = 1 if args.target_mode == "binary" else len(class_names)
    model = make_model(
        args,
        model_name=model_name,
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = -math.inf
    best_epoch = 0
    best_val_rows: list[dict[str, object]] = []
    best_class_biases: list[float] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed + fold * 10_000)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_records,
            store,
            normalizer,
            optimizer,
            loss_fn,
            device,
            args.max_patches,
            rng,
            args.target_mode,
        )
        val_rows = predict_records(
            model,
            val_records,
            store,
            normalizer,
            device,
            args.eval_max_patches,
            seed=args.seed + fold * 100 + epoch,
            target_mode=args.target_mode,
            class_names=class_names,
        )
        threshold = select_threshold(val_rows, args.threshold_metric) if args.target_mode == "binary" else None
        class_biases = None
        if args.calibrate_class_bias and args.target_mode != "binary":
            class_biases = select_class_biases(val_rows, class_names, args.calibration_metric)
            val_rows_for_metrics = apply_class_biases(val_rows, class_names, class_biases)
        else:
            val_rows_for_metrics = val_rows
        val_metrics = compute_metrics(val_rows_for_metrics, threshold, args.target_mode, class_names)
        score = val_metrics[args.selection_metric]
        if isinstance(score, float) and math.isnan(score):
            score = val_metrics["balanced_accuracy"]
        history_row: dict[str, object] = {
            "model": model_name,
            "fold": fold,
            "epoch": epoch,
            "train_loss": loss,
            "class_biases": json.dumps(class_biases) if class_biases is not None else "",
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(history_row)

        if score > best_val_auc:
            best_val_auc = float(score)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_rows = val_rows
            best_class_biases = class_biases
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} fold {fold} did not train")
    model.load_state_dict(best_state)

    threshold = select_threshold(best_val_rows, args.threshold_metric) if args.target_mode == "binary" else None
    test_rows = predict_records(
        model,
        test_records,
        store,
        normalizer,
        device,
        args.eval_max_patches,
        seed=args.seed + fold * 1000,
        target_mode=args.target_mode,
        class_names=class_names,
    )
    val_rows_for_metrics = best_val_rows
    if best_class_biases is not None:
        val_rows_for_metrics = apply_class_biases(best_val_rows, class_names, best_class_biases)
        test_rows = apply_class_biases(test_rows, class_names, best_class_biases)
        write_json(
            fold_dir / "class_bias_calibration.json",
            {
                "metric": args.calibration_metric,
                "class_names": class_names,
                "biases": dict(zip(class_names, best_class_biases)),
            },
        )
    val_metrics = compute_metrics(val_rows_for_metrics, threshold, args.target_mode, class_names)
    test_metrics = compute_metrics(test_rows, threshold, args.target_mode, class_names)
    if args.target_mode == "binary":
        best_val_rows = apply_binary_threshold(best_val_rows, threshold, class_names)
        test_rows = apply_binary_threshold(test_rows, threshold, class_names)
    metrics_row: dict[str, object] = {
        "model": model_name,
        "fold": fold,
        "best_epoch": best_epoch,
        "train_n": len(train_records),
        "val_n": len(val_records),
        "test_n": len(test_records),
        "selection_metric": args.selection_metric,
        "calibration_metric": args.calibration_metric if best_class_biases is not None else "",
        "class_biases": json.dumps(best_class_biases) if best_class_biases is not None else "",
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    torch.save(
        {
            "model": model.state_dict(),
            "model_name": model_name,
            "input_dim": input_dim,
            "class_names": class_names,
            "args": vars(args),
            "metrics": metrics_row,
        },
        fold_dir / "best_model.pt",
    )
    write_csv(fold_dir / "history.csv", history_rows)
    write_csv(fold_dir / "val_predictions.csv", best_val_rows)
    write_csv(fold_dir / "test_predictions.csv", test_rows)
    write_json(fold_dir / "metrics.json", metrics_row)
    if model_name == "abmil" and args.attention_topk > 0:
        export_attention_topk(
            model,
            test_records,
            test_rows,
            store,
            normalizer,
            device,
            args.attention_topk,
            fold_dir / "test_attention_top_patches.csv",
            args.target_mode,
            class_names,
        )

    return best_val_rows, test_rows, metrics_row


def summarize_metrics(metrics_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in metrics_rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    metric_names = [
        "test_roc_auc",
        "test_average_precision",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall_sensitivity",
        "test_specificity",
    ]
    summary_rows: list[dict[str, object]] = []
    for model_name, rows in sorted(grouped.items()):
        summary: dict[str, object] = {"model": model_name, "folds": len(rows)}
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in rows], dtype=float)
            values = values[~np.isnan(values)]
            summary[f"{metric_name}_mean"] = float(values.mean()) if len(values) else math.nan
            summary[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/biopsy_mil_abmil"))
    parser.add_argument(
        "--target-mode",
        choices=["binary", "major", "label"],
        default="binary",
        help=(
            "Prediction target: binary carcinoma target, six Bethesda major labels, "
            "or exact label strings."
        ),
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=None,
        help=(
            "Optional subset/order of multiclass target names. Example: "
            "--target-mode major --target-classes II III IV V VI."
        ),
    )
    parser.add_argument(
        "--binary-positive-labels",
        nargs="+",
        default=None,
        help=(
            "Exact label names to treat as positive in --target-mode binary. "
            "Example: --binary-positive-labels IIIb. Matching is case-insensitive."
        ),
    )
    parser.add_argument(
        "--binary-positive-majors",
        nargs="+",
        default=None,
        help=(
            "Major label names to treat as positive in --target-mode binary. "
            "Example: --binary-positive-majors IV. Matching is case-insensitive."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["abmil"],
        choices=["abmil", "meanpool"],
        help="MIL model(s) to train. Use both for a quick sanity comparison.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Randomly sample this many patches per slide during training. Default uses every patch.",
    )
    parser.add_argument(
        "--eval-max-patches",
        type=int,
        default=None,
        help="Deterministically sample this many patches per slide during validation/test.",
    )
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalizer-max-patches", type=int, default=2048)
    parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1"],
        default="balanced_accuracy",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"],
        default="roc_auc",
        help="Validation metric used for early stopping/checkpoint selection.",
    )
    parser.add_argument(
        "--calibrate-class-bias",
        action="store_true",
        help="Tune per-class logit biases on validation predictions and apply them to test predictions.",
    )
    parser.add_argument(
        "--calibration-metric",
        choices=["balanced_accuracy", "f1", "accuracy"],
        default="balanced_accuracy",
        help="Validation prediction metric optimized by --calibrate-class-bias.",
    )
    parser.add_argument("--attention-topk", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.min_patches < 1:
        raise ValueError("--min-patches must be at least 1")
    for name in ["max_patches", "eval_max_patches", "normalizer_max_patches"]:
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when set")
    if args.attention_topk < 0:
        raise ValueError("--attention-topk must be non-negative")
    target_class_names(args.target_mode, args.target_classes)
    args.binary_positive_labels = normalize_target_names(
        args.binary_positive_labels,
        EXACT_LABEL_ORDER,
        "binary-positive-labels",
    )
    args.binary_positive_majors = normalize_target_names(
        args.binary_positive_majors,
        MAJOR_LABELS,
        "binary-positive-majors",
    )
    if args.target_mode != "binary" and (args.binary_positive_labels or args.binary_positive_majors):
        raise ValueError("--binary-positive-labels/--binary-positive-majors require --target-mode binary")
    if args.calibrate_class_bias and args.target_mode == "binary":
        raise ValueError("--calibrate-class-bias is only supported for multiclass target modes")


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    class_names = target_class_names(args.target_mode, args.target_classes)
    records, dataset_summary = read_records(
        args.labels_csv,
        min_patches=args.min_patches,
        target_mode=args.target_mode,
        class_names=class_names,
        binary_positive_labels=args.binary_positive_labels,
        binary_positive_majors=args.binary_positive_majors,
    )
    input_dim = infer_input_dim(records)
    dataset_summary["input_dim"] = input_dim
    dataset_summary["device"] = str(device)
    write_json(output_dir / "dataset_summary.json", dataset_summary)
    write_json(output_dir / "config.json", vars(args))

    splits = make_splits(records, n_splits=args.folds, val_fraction=args.val_fraction, seed=args.seed)
    store = EmbeddingStore(cache_in_memory=args.cache_in_memory)
    metrics_rows: list[dict[str, object]] = []
    test_rows_for_plot: dict[tuple[str, int], list[dict[str, object]]] = {}
    all_test_rows: list[dict[str, object]] = []

    print(
        "dataset: {used_rows} slides, target_mode={target_mode}, {unique_cases} cases, "
        "input_dim={input_dim}, device={device}".format(**dataset_summary),
        flush=True,
    )
    print(f"class_counts={dataset_summary['target_counts']}", flush=True)

    for model_name in args.models:
        for fold, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
            print(
                f"model={model_name} fold={fold}/{args.folds} "
                f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
                flush=True,
            )
            val_rows, test_rows, metrics_row = run_model_fold(
                args,
                model_name,
                fold,
                subset(records, train_idx),
                subset(records, val_idx),
                subset(records, test_idx),
                input_dim,
                device,
                store,
                output_dir,
                class_names,
            )
            metrics_rows.append(metrics_row)
            test_rows_for_plot[(model_name, fold)] = test_rows
            for row in test_rows:
                all_row = {"model": model_name, "fold": fold, **row}
                all_test_rows.append(all_row)
            print(
                f"  best_epoch={metrics_row['best_epoch']} "
                f"test_auc={metrics_row['test_roc_auc']:.4f} "
                f"test_bal_acc={metrics_row['test_balanced_accuracy']:.4f} "
                f"test_recall={metrics_row['test_recall_sensitivity']:.4f} "
                f"test_specificity={metrics_row['test_specificity']:.4f}",
                flush=True,
            )

    summary_rows = summarize_metrics(metrics_rows)
    write_csv(output_dir / "metrics_by_fold.csv", metrics_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "test_predictions_all.csv", all_test_rows)
    if not args.no_plots and args.target_mode == "binary":
        save_plot_curves(output_dir, test_rows_for_plot)

    elapsed_s = time.time() - start_time
    write_json(output_dir / "run_summary.json", {"elapsed_s": elapsed_s, "summary": summary_rows})
    print(f"wrote results to {output_dir} in {elapsed_s / 60:.1f} min", flush=True)
    for row in summary_rows:
        print(
            "{model}: auc={test_roc_auc_mean:.4f}+/-{test_roc_auc_std:.4f}, "
            "bal_acc={test_balanced_accuracy_mean:.4f}+/-{test_balanced_accuracy_std:.4f}, "
            "ap={test_average_precision_mean:.4f}+/-{test_average_precision_std:.4f}".format(**row),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
