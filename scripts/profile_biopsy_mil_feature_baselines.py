#!/usr/bin/env python3
"""Profile CPU feature baselines on pooled UNI biopsy embeddings.

This is meant as a lightweight companion to ABMIL experiments. It reuses the
existing grouped fold split files, converts each slide bag to pooled embedding
statistics, tunes simple classifiers on each fold's validation split, and
reports test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_BASE_DIR = Path("outputs/biopsy_mil_major_II_to_VI_profiled")
DEFAULT_OUTPUT_DIR = Path("outputs/biopsy_mil_major_II_to_VI_feature_baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--selection-metric", choices=["f1", "balanced_accuracy", "accuracy"], default="f1")
    parser.add_argument("--skip-trees", action="store_true", help="Skip ExtraTrees baselines.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
        handle.write("\n")


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def class_names_from_base(base_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return list(requested)
    summary_path = base_dir / "dataset_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        names = summary.get("class_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    sample = pd.read_csv(base_dir / "abmil" / "fold_1" / "test_predictions.csv", nrows=1)
    return [column.removeprefix("logit_") for column in sample.columns if column.startswith("logit_")]


def load_splits(base_dir: Path, folds: int) -> dict[int, pd.DataFrame]:
    splits: dict[int, pd.DataFrame] = {}
    for fold in range(1, folds + 1):
        path = base_dir / "abmil" / f"fold_{fold}" / "split.csv"
        frame = pd.read_csv(path)
        if "index" not in frame.columns:
            raise ValueError(f"{path} is missing the record index column")
        splits[fold] = frame
    return splits


def collect_unique_records(splits: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(splits.values(), ignore_index=True)
    return frame.drop_duplicates("index").sort_values("index").reset_index(drop=True)


def feature_cache_path(output_dir: Path) -> Path:
    return output_dir / "slide_feature_cache_mean_std.npz"


def build_or_load_features(
    records: pd.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    cache_path = feature_cache_path(output_dir)
    if cache_path.exists():
        data = np.load(cache_path)
        record_indices = data["record_indices"].astype(int)
        mean_features = data["mean_features"].astype(np.float32)
        mean_std_features = data["mean_std_features"].astype(np.float32)
        index_to_row = {int(record_index): row_index for row_index, record_index in enumerate(record_indices)}
        return record_indices, mean_features, mean_std_features, index_to_row

    output_dir.mkdir(parents=True, exist_ok=True)
    record_indices: list[int] = []
    mean_rows: list[np.ndarray] = []
    mean_std_rows: list[np.ndarray] = []
    for count, row in enumerate(records.itertuples(index=False), start=1):
        with np.load(row.embedding_path) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"{row.embedding_path} has invalid embeddings shape {embeddings.shape}")
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        patch_features = np.asarray([np.log1p(float(embeddings.shape[0]))], dtype=np.float32)
        record_indices.append(int(row.index))
        mean_rows.append(np.concatenate([mean, patch_features]).astype(np.float32))
        mean_std_rows.append(np.concatenate([mean, std, patch_features]).astype(np.float32))
        if count % 200 == 0:
            print(f"  pooled {count}/{len(records)} slides", flush=True)

    record_indices_array = np.asarray(record_indices, dtype=np.int64)
    mean_features = np.vstack(mean_rows).astype(np.float32)
    mean_std_features = np.vstack(mean_std_rows).astype(np.float32)
    np.savez_compressed(
        cache_path,
        record_indices=record_indices_array,
        mean_features=mean_features,
        mean_std_features=mean_std_features,
    )
    index_to_row = {int(record_index): row_index for row_index, record_index in enumerate(record_indices_array)}
    return record_indices_array, mean_features, mean_std_features, index_to_row


def matrix_for_split(
    split_frame: pd.DataFrame,
    features: np.ndarray,
    index_to_row: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = [index_to_row[int(record_index)] for record_index in split_frame["index"]]
    return features[rows], split_frame["target"].to_numpy(dtype=int), split_frame.reset_index(drop=True)


def metric_score(targets: np.ndarray, pred: np.ndarray, metric: str, class_count: int) -> float:
    labels = np.arange(class_count)
    if metric == "f1":
        return float(f1_score(targets, pred, labels=labels, average="macro", zero_division=0))
    if metric == "balanced_accuracy":
        return float(recall_score(targets, pred, labels=labels, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(targets, pred))
    raise ValueError(f"unsupported metric: {metric}")


def safe_auc(targets: np.ndarray, probs: np.ndarray, class_count: int) -> float:
    for class_index in range(class_count):
        y_true = targets == class_index
        if not y_true.any() or not (~y_true).any():
            return math.nan
    try:
        return float(
            roc_auc_score(
                targets,
                probs,
                labels=np.arange(class_count),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return math.nan


def safe_ap(targets: np.ndarray, probs: np.ndarray, class_count: int) -> float:
    values: list[float] = []
    for class_index in range(class_count):
        y_true = targets == class_index
        if y_true.any() and (~y_true).any():
            values.append(float(average_precision_score(y_true.astype(int), probs[:, class_index])))
    return float(np.mean(values)) if values else math.nan


def macro_specificity(targets: np.ndarray, pred: np.ndarray, class_count: int) -> float:
    values: list[float] = []
    for class_index in range(class_count):
        y_true = targets == class_index
        y_pred = pred == class_index
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        if tn + fp:
            values.append(tn / (tn + fp))
    return float(np.mean(values)) if values else math.nan


def metrics(targets: np.ndarray, pred: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict[str, object]:
    class_count = len(class_names)
    labels = np.arange(class_count)
    distance = np.abs(pred - targets)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    matrix = confusion_matrix(targets, pred, labels=labels)
    return {
        "n": int(len(targets)),
        "roc_auc": safe_auc(targets, probs, class_count),
        "average_precision": safe_ap(targets, probs, class_count),
        "accuracy": float(accuracy_score(targets, pred)),
        "balanced_accuracy": float(recall_score(targets, pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(targets, pred, labels=labels, average="macro", zero_division=0)),
        "precision": float(precision_score(targets, pred, labels=labels, average="macro", zero_division=0)),
        "recall_sensitivity": float(recall_score(targets, pred, labels=labels, average="macro", zero_division=0)),
        "specificity": macro_specificity(targets, pred, class_count),
        "f1_weighted": float(f1_score(targets, pred, labels=labels, average="weighted", zero_division=0)),
        "within_one_accuracy": float(np.mean(distance <= 1)),
        "top2_accuracy": float(np.mean([target in pair for target, pair in zip(targets, top2)])),
        "mean_abs_error": float(distance.mean()),
        "severe_error_rate": float(np.mean(distance >= 2)),
        "confusion_json": json.dumps(matrix.tolist()),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_weighted",
        "within_one_accuracy",
        "top2_accuracy",
        "mean_abs_error",
        "severe_error_rate",
    ]
    out: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "folds": len(method_rows)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in method_rows], dtype=float)
            values = values[~np.isnan(values)]
            summary[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def align_probabilities(classes: np.ndarray, probs: np.ndarray, class_count: int) -> np.ndarray:
    out = np.zeros((len(probs), class_count), dtype=float)
    for source_index, class_value in enumerate(classes):
        out[:, int(class_value)] = probs[:, source_index]
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return out / row_sum


def decision_to_probs(decision: np.ndarray, classes: np.ndarray, class_count: int) -> np.ndarray:
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    probs = softmax(decision)
    return align_probabilities(classes, probs, class_count)


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
    metric: str,
) -> tuple[object, dict[str, object]]:
    best: tuple[float, float, object] | None = None
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=alpha, class_weight="balanced"),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        score = metric_score(y_val, pred, metric, class_count)
        if best is None or score > best[0] + 1e-12:
            best = (score, alpha, clf)
    if best is None:
        raise RuntimeError("ridge grid did not run")
    return best[2], {"alpha": best[1], "val_score": best[0]}


def fit_sgd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
    metric: str,
    seed: int,
) -> tuple[object, dict[str, object]]:
    best: tuple[float, float, object] | None = None
    for alpha in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                class_weight="balanced",
                max_iter=1500,
                tol=1e-3,
                random_state=seed,
                average=True,
            ),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        score = metric_score(y_val, pred, metric, class_count)
        if best is None or score > best[0] + 1e-12:
            best = (score, alpha, clf)
    if best is None:
        raise RuntimeError("sgd grid did not run")
    return best[2], {"alpha": best[1], "val_score": best[0]}


def fit_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
    metric: str,
) -> tuple[object, dict[str, object]]:
    best: tuple[float, int, object] | None = None
    for neighbors in [3, 5, 9, 15, 25, 35]:
        clf = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=neighbors, weights="distance", metric="cosine"),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        score = metric_score(y_val, pred, metric, class_count)
        if best is None or score > best[0] + 1e-12:
            best = (score, neighbors, clf)
    if best is None:
        raise RuntimeError("knn grid did not run")
    return best[2], {"neighbors": best[1], "val_score": best[0]}


def fit_extra_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
    metric: str,
    seed: int,
) -> tuple[object, dict[str, object]]:
    best: tuple[float, str | float, int, object] | None = None
    for max_features in ["sqrt", 0.2]:
        for min_samples_leaf in [1, 3]:
            clf = ExtraTreesClassifier(
                n_estimators=240,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(x_train, y_train)
            pred = clf.predict(x_val)
            score = metric_score(y_val, pred, metric, class_count)
            if best is None or score > best[0] + 1e-12:
                best = (score, max_features, min_samples_leaf, clf)
    if best is None:
        raise RuntimeError("extra trees grid did not run")
    return best[3], {"max_features": best[1], "min_samples_leaf": best[2], "val_score": best[0]}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return x / norm


def select_centroid_bias(targets: np.ndarray, scores: np.ndarray, metric: str) -> tuple[np.ndarray, float]:
    class_count = scores.shape[1]
    biases = np.zeros(class_count, dtype=float)
    best_score = metric_score(targets, np.argmax(scores + biases, axis=1), metric, class_count)
    for grid in [np.linspace(-0.5, 0.5, 21), np.linspace(-0.15, 0.15, 13), np.linspace(-0.05, 0.05, 11)]:
        improved = True
        while improved:
            improved = False
            for class_index in range(class_count):
                current = float(biases[class_index])
                local_best = best_score
                local_biases = biases
                for delta in grid:
                    candidate = biases.copy()
                    candidate[class_index] = current + float(delta)
                    candidate -= candidate.mean()
                    score = metric_score(targets, np.argmax(scores + candidate, axis=1), metric, class_count)
                    if score > local_best + 1e-12:
                        local_best = score
                        local_biases = candidate
                if local_best > best_score + 1e-12:
                    biases = local_biases
                    best_score = local_best
                    improved = True
    return biases, float(best_score)


def centroid_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    class_count: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_norm = normalize_rows(x_train)
    centroids = []
    for class_index in range(class_count):
        class_rows = train_norm[y_train == class_index]
        if len(class_rows) == 0:
            centroids.append(np.zeros(train_norm.shape[1], dtype=float))
        else:
            centroids.append(class_rows.mean(axis=0))
    centroids_array = normalize_rows(np.vstack(centroids))
    val_scores = normalize_rows(x_val) @ centroids_array.T
    biases, val_score = select_centroid_bias(y_val, val_scores, metric)
    test_scores = normalize_rows(x_test) @ centroids_array.T + biases
    probs = softmax(test_scores * 8.0)
    pred = np.argmax(test_scores, axis=1)
    return pred, probs, {"biases": biases.tolist(), "val_score": val_score}


def predict_model(clf: object, x_test: np.ndarray, class_count: int) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(clf.predict(x_test), dtype=int)
    final_estimator = clf[-1] if hasattr(clf, "__getitem__") else clf
    classes = np.asarray(final_estimator.classes_, dtype=int)
    if hasattr(clf, "predict_proba"):
        probs = align_probabilities(classes, clf.predict_proba(x_test), class_count)
    elif hasattr(clf, "decision_function"):
        probs = decision_to_probs(clf.decision_function(x_test), classes, class_count)
    else:
        probs = np.zeros((len(pred), class_count), dtype=float)
        probs[np.arange(len(pred)), pred] = 1.0
    empty = probs.sum(axis=1) == 0.0
    if empty.any():
        probs[empty, :] = 0.0
        probs[np.where(empty)[0], pred[empty]] = 1.0
    return pred, probs


def prediction_rows(
    method: str,
    fold: int,
    split_frame: pd.DataFrame,
    pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, source in split_frame.iterrows():
        out: dict[str, object] = {
            "method": method,
            "fold": fold,
            "record_index": int(source["index"]),
            "embedding_file": source["embedding_file"],
            "case_id": source["case_id"],
            "target": int(source["target"]),
            "target_name": source["target_name"],
            "pred": int(pred[row_index]),
            "pred_name": class_names[int(pred[row_index])],
            "confidence": float(probs[row_index].max()),
            "correct": int(pred[row_index] == int(source["target"])),
            "abs_error": int(abs(pred[row_index] - int(source["target"]))),
        }
        for class_index, class_name in enumerate(class_names):
            out[f"prob_{class_name}"] = float(probs[row_index, class_index])
        rows.append(out)
    return rows


def render_report(path: Path, summary_rows: list[dict[str, object]], class_names: list[str]) -> None:
    def fmt(row: dict[str, object], metric: str) -> str:
        return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"

    ranked = sorted(summary_rows, key=lambda row: float(row["f1_mean"]), reverse=True)
    lines = [
        "# Feature Baseline Profile",
        "",
        f"Target classes: `{', '.join(class_names)}`",
        "",
        "All models use pooled UNI embedding statistics and the same grouped folds as the ABMIL profile.",
        "Hyperparameters are selected on the validation split for each fold.",
        "",
        "| method | accuracy | balanced accuracy | macro F1 | within-one acc | top-2 acc | severe error rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            "| {method} | {accuracy} | {bal} | {f1} | {within_one} | {top2} | {severe} |".format(
                method=row["method"],
                accuracy=fmt(row, "accuracy"),
                bal=fmt(row, "balanced_accuracy"),
                f1=fmt(row, "f1"),
                within_one=fmt(row, "within_one_accuracy"),
                top2=fmt(row, "top2_accuracy"),
                severe=fmt(row, "severe_error_rate"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    class_names = class_names_from_base(args.base_dir, args.class_names)
    class_count = len(class_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.base_dir, args.folds)
    records = collect_unique_records(splits)
    print(f"building/loading pooled features for {len(records)} slides", flush=True)
    _, mean_features, mean_std_features, index_to_row = build_or_load_features(records, args.output_dir)

    metric_rows: list[dict[str, object]] = []
    prediction_output_rows: list[dict[str, object]] = []
    configs: dict[str, dict[str, object]] = {}

    model_specs = [
        ("centroid_mean_cosine", "mean", "centroid"),
        ("ridge_mean_std", "mean_std", "ridge"),
        ("sgd_log_mean_std", "mean_std", "sgd"),
        ("knn_mean_cosine", "mean", "knn"),
    ]
    if not args.skip_trees:
        model_specs.append(("extra_trees_mean_std", "mean_std", "extra_trees"))

    for fold in range(1, args.folds + 1):
        split = splits[fold]
        train_frame = split[split["split"] == "train"]
        val_frame = split[split["split"] == "val"]
        test_frame = split[split["split"] == "test"]
        for method, feature_name, model_name in model_specs:
            features = mean_features if feature_name == "mean" else mean_std_features
            x_train, y_train, _ = matrix_for_split(train_frame, features, index_to_row)
            x_val, y_val, _ = matrix_for_split(val_frame, features, index_to_row)
            x_test, y_test, test_rows = matrix_for_split(test_frame, features, index_to_row)

            if model_name == "centroid":
                pred, probs, config = centroid_fit_predict(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    class_count,
                    args.selection_metric,
                )
            else:
                if model_name == "ridge":
                    clf, config = fit_ridge(x_train, y_train, x_val, y_val, class_count, args.selection_metric)
                elif model_name == "sgd":
                    clf, config = fit_sgd(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        class_count,
                        args.selection_metric,
                        seed=2026 + fold,
                    )
                elif model_name == "knn":
                    clf, config = fit_knn(x_train, y_train, x_val, y_val, class_count, args.selection_metric)
                elif model_name == "extra_trees":
                    clf, config = fit_extra_trees(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        class_count,
                        args.selection_metric,
                        seed=2026 + fold,
                    )
                else:
                    raise ValueError(f"unsupported model: {model_name}")
                pred, probs = predict_model(clf, x_test, class_count)

            row = {"method": method, "fold": fold, **metrics(y_test, pred, probs, class_names)}
            metric_rows.append(row)
            prediction_output_rows.extend(prediction_rows(method, fold, test_rows, pred, probs, class_names))
            configs.setdefault(method, {})[f"fold_{fold}"] = config
            print(
                f"{method} fold={fold}: val={config.get('val_score', math.nan):.4f} "
                f"test_f1={row['f1']:.4f} test_bal={row['balanced_accuracy']:.4f} "
                f"test_acc={row['accuracy']:.4f}",
                flush=True,
            )

    summary_rows = summarize(metric_rows)
    aggregate_confusions: dict[str, object] = {}
    for method in sorted({row["method"] for row in prediction_output_rows}):
        rows = [row for row in prediction_output_rows if row["method"] == method]
        targets = np.asarray([int(row["target"]) for row in rows])
        pred = np.asarray([int(row["pred"]) for row in rows])
        aggregate_confusions[method] = {
            "class_names": class_names,
            "confusion": confusion_matrix(targets, pred, labels=np.arange(class_count)).tolist(),
        }

    write_csv(args.output_dir / "metrics_by_fold.csv", metric_rows)
    write_csv(args.output_dir / "metrics_summary.csv", summary_rows)
    write_csv(args.output_dir / "test_predictions_all.csv", prediction_output_rows)
    write_json(args.output_dir / "method_configs.json", configs)
    write_json(args.output_dir / "aggregate_confusion.json", aggregate_confusions)
    render_report(args.output_dir / "feature_report.md", summary_rows, class_names)

    print(f"wrote feature baseline results to {args.output_dir}")
    for row in sorted(summary_rows, key=lambda item: float(item["f1_mean"]), reverse=True):
        print(
            "{method}: f1={f1_mean:.4f}+/-{f1_std:.4f}, "
            "bal_acc={balanced_accuracy_mean:.4f}+/-{balanced_accuracy_std:.4f}, "
            "acc={accuracy_mean:.4f}+/-{accuracy_std:.4f}, "
            "within1={within_one_accuracy_mean:.4f}+/-{within_one_accuracy_std:.4f}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
