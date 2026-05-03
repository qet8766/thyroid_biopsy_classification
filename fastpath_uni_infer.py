#!/usr/bin/env python3
"""Run UNI feature extraction on tissue-positive FastPATH level-0 patches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import struct
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


IDX_MAGIC = b"FPLIDX1\0"
DEFAULT_MODEL_ID = "hf-hub:MahmoodLab/uni"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@contextmanager
def nvtx_range(torch_module, message: str):
    if (
        torch_module is not None
        and hasattr(torch_module, "cuda")
        and torch_module.cuda.is_available()
        and hasattr(torch_module.cuda, "nvtx")
    ):
        torch_module.cuda.nvtx.range_push(message)
        try:
            yield
        finally:
            torch_module.cuda.nvtx.range_pop()
    else:
        yield


@dataclass(frozen=True)
class LevelInfo:
    level: int
    downsample: float
    cols: int
    rows: int


@dataclass(frozen=True)
class PatchRecord:
    row: int
    col: int
    source_row: int
    source_col: int
    subtile_row: int
    subtile_col: int
    tissue_fraction: float


@dataclass
class SlidePlan:
    slide_path: Path
    rel_path: str
    source_file: str
    tile_size: int
    patch_size: int
    subtiles_per_tile: int
    target_mpp: float | None
    source_mpp: float | None
    level0: LevelInfo
    selected_patches: list[PatchRecord]
    total_tiles: int
    total_patches: int


class FastPathPackReader:
    """Minimal reader for FastPATH pack_v2 level files."""

    def __init__(self, slide_path: Path, level: int) -> None:
        self.slide_path = slide_path
        tiles_dir = slide_path / "tiles"
        self.idx_path = tiles_dir / f"level_{level}.idx"
        self.pack_path = tiles_dir / f"level_{level}.pack"
        self.cols, self.rows, self.entries = self._parse_idx(self.idx_path)
        self._fd: int | None = None

    def __enter__(self) -> "FastPathPackReader":
        self._fd = os.open(self.pack_path, os.O_RDONLY)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None

    @staticmethod
    def _parse_idx(idx_path: Path) -> tuple[int, int, list[tuple[int, int]]]:
        data = idx_path.read_bytes()
        if len(data) < 16:
            raise ValueError(f"{idx_path}: index is too short")
        if data[:8] != IDX_MAGIC:
            raise ValueError(f"{idx_path}: invalid index magic")
        version = struct.unpack_from("<I", data, 8)[0]
        if version != 1:
            raise ValueError(f"{idx_path}: unsupported index version {version}")
        cols, rows = struct.unpack_from("<HH", data, 12)
        expected = 16 + cols * rows * 12
        if len(data) < expected:
            raise ValueError(f"{idx_path}: truncated index")

        entries: list[tuple[int, int]] = []
        for offset in range(16, expected, 12):
            pack_offset, length = struct.unpack_from("<QI", data, offset)
            entries.append((pack_offset, length))
        return cols, rows, entries

    def read_raw(self, row: int, col: int) -> bytes | None:
        if self._fd is None:
            raise RuntimeError("FastPathPackReader must be used as a context manager")
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return None
        pack_offset, length = self.entries[row * self.cols + col]
        if length <= 0:
            return None
        return os.pread(self._fd, length, pack_offset)


def load_case_list(case_list_path: Path | None) -> set[str] | None:
    if case_list_path is None:
        return None

    case_ids: set[str] = set()
    with case_list_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            case_ids.add(item)
    return case_ids


def slide_matches_case_list(path: Path, input_root: Path, case_ids: set[str] | None) -> bool:
    if case_ids is None:
        return True
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = path

    parts = set(rel.parts)
    if any(case_id in parts for case_id in case_ids):
        return True

    rel_text = str(rel)
    return any(rel_text == case_id or rel_text.startswith(f"{case_id}/") for case_id in case_ids)


def discover_slides(input_root: Path, include_ignore: bool, case_ids: set[str] | None = None) -> list[Path]:
    def eligible(path: Path) -> bool:
        if "IHC" in path.name.upper():
            return False
        if not include_ignore and "ignore" in path.parts:
            return False
        if not slide_matches_case_list(path, input_root, case_ids):
            return False
        return True

    slides: list[Path] = []
    if input_root.name.endswith(".fastpath") and input_root.is_dir():
        return [input_root] if eligible(input_root) else []

    for dirpath, dirnames, _filenames in os.walk(input_root):
        current = Path(dirpath)
        kept_dirnames: list[str] = []
        for dirname in dirnames:
            child = current / dirname
            if dirname.endswith(".fastpath"):
                if eligible(child):
                    slides.append(child)
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames
    return sorted(slides, key=lambda p: str(p))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_level(metadata: dict, level: int) -> LevelInfo:
    for item in metadata.get("levels", []):
        if int(item.get("level", -1)) == level:
            return LevelInfo(
                level=level,
                downsample=float(item["downsample"]),
                cols=int(item["cols"]),
                rows=int(item["rows"]),
            )
    raise ValueError(f"metadata has no level {level}")


def load_tissue_polygons(slide_path: Path) -> list[list[tuple[float, float]]]:
    result_path = slide_path / "plugin_results" / "tis_seg.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing tissue mask: {result_path}")

    data = read_json(result_path)
    result = data.get("result") or {}
    if not result.get("success", False):
        raise ValueError(f"tissue segmentation did not succeed: {result_path}")

    polygons: list[list[tuple[float, float]]] = []
    for annotation in result.get("annotations", []):
        if annotation.get("type") != "polygon":
            continue
        coords = annotation.get("coordinates") or []
        polygon: list[tuple[float, float]] = []
        for point in coords:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                polygon.append((float(point[0]), float(point[1])))
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def rasterize_polygons(
    polygons: Iterable[list[tuple[float, float]]],
    full_width: float,
    full_height: float,
    max_edge: int,
) -> np.ndarray:
    if full_width <= 0 or full_height <= 0:
        raise ValueError("slide grid dimensions must be positive")

    scale = min(1.0, float(max_edge) / max(full_width, full_height))
    mask_w = max(1, int(round(full_width * scale)))
    mask_h = max(1, int(round(full_height * scale)))
    image = Image.new("L", (mask_w, mask_h), 0)
    draw = ImageDraw.Draw(image)
    scale_x = mask_w / full_width
    scale_y = mask_h / full_height

    for polygon in polygons:
        scaled = [(x * scale_x, y * scale_y) for x, y in polygon]
        draw.polygon(scaled, fill=1)

    return np.asarray(image, dtype=np.uint8)


def tile_tissue_fractions(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    padded = np.pad(mask.astype(np.int64), ((1, 0), (1, 0)), mode="constant")
    integral = padded.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    fractions = np.zeros((rows, cols), dtype=np.float32)
    mask_h, mask_w = mask.shape

    for row in range(rows):
        y0 = row * mask_h // rows
        y1 = (row + 1) * mask_h // rows
        for col in range(cols):
            x0 = col * mask_w // cols
            x1 = (col + 1) * mask_w // cols
            area = (y1 - y0) * (x1 - x0)
            if area <= 0:
                continue
            tissue = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )
            fractions[row, col] = tissue / area
    return fractions


def build_slide_plan(
    slide_path: Path,
    input_root: Path,
    tissue_threshold: float,
    mask_max_edge: int,
    patch_size: int,
) -> SlidePlan:
    metadata = read_json(slide_path / "metadata.json")
    if metadata.get("tile_format") != "pack_v2":
        raise ValueError(f"unsupported tile_format: {metadata.get('tile_format')}")
    level0 = get_level(metadata, 0)
    tile_size = int(metadata["tile_size"])
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_size > tile_size:
        raise ValueError(f"patch_size {patch_size} is larger than source tile_size {tile_size}")
    if tile_size % patch_size != 0:
        raise ValueError(f"patch_size {patch_size} must evenly divide source tile_size {tile_size}")
    subtiles_per_tile = tile_size // patch_size
    polygons = load_tissue_polygons(slide_path)

    full_width = level0.cols * tile_size * level0.downsample
    full_height = level0.rows * tile_size * level0.downsample
    patch_rows = level0.rows * subtiles_per_tile
    patch_cols = level0.cols * subtiles_per_tile
    if polygons:
        mask = rasterize_polygons(polygons, full_width, full_height, mask_max_edge)
        fractions = tile_tissue_fractions(mask, patch_rows, patch_cols)
    else:
        fractions = np.zeros((patch_rows, patch_cols), dtype=np.float32)

    selected: list[PatchRecord] = []
    for row, col in np.argwhere(fractions >= tissue_threshold):
        source_row = int(row) // subtiles_per_tile
        source_col = int(col) // subtiles_per_tile
        subtile_row = int(row) % subtiles_per_tile
        subtile_col = int(col) % subtiles_per_tile
        selected.append(
            PatchRecord(
                row=int(row),
                col=int(col),
                source_row=source_row,
                source_col=source_col,
                subtile_row=subtile_row,
                subtile_col=subtile_col,
                tissue_fraction=float(fractions[row, col]),
            )
        )

    return SlidePlan(
        slide_path=slide_path,
        rel_path=str(slide_path.relative_to(input_root)),
        source_file=str(metadata.get("source_file", "")),
        tile_size=tile_size,
        patch_size=patch_size,
        subtiles_per_tile=subtiles_per_tile,
        target_mpp=float(metadata["target_mpp"]) if "target_mpp" in metadata else None,
        source_mpp=float(metadata["source_mpp"]) if "source_mpp" in metadata else None,
        level0=level0,
        selected_patches=selected,
        total_tiles=level0.rows * level0.cols,
        total_patches=patch_rows * patch_cols,
    )


def safe_stem(rel_path: str) -> str:
    stem = rel_path.removesuffix(".fastpath")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "slide"
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}_{digest}"


def output_path_for(output_dir: Path, rel_path: str) -> Path:
    return output_dir / f"{safe_stem(rel_path)}.npz"


def load_uni_model(model_id: str, device: str, hf_token_env: str):
    import torch
    import timm
    from huggingface_hub import login
    from timm.data import create_transform, resolve_data_config

    token = None
    for env_name in dict.fromkeys([hf_token_env, "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"]):
        token = os.environ.get(env_name)
        if token:
            break
    if token:
        login(token=token, add_to_git_credential=False)

    resolved_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    with nvtx_range(torch, "load_model:timm_create_model"):
        model = timm.create_model(
            model_id,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
    with nvtx_range(torch, "load_model:transform"):
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    with nvtx_range(torch, "load_model:to_device"):
        model.eval().to(resolved_device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, transform, resolved_device, torch


def decode_source_tile_pil(reader: FastPathPackReader, source_row: int, source_col: int):
    raw = reader.read_raw(source_row, source_col)
    if raw is None:
        return None
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def decode_source_tile_cv2(reader: FastPathPackReader, source_row: int, source_col: int):
    raw = reader.read_raw(source_row, source_col)
    if raw is None:
        return None

    import cv2

    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_patch_np(image: np.ndarray, record: PatchRecord, patch_size: int) -> np.ndarray:
    y0 = record.subtile_row * patch_size
    x0 = record.subtile_col * patch_size
    patch = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    if y0 >= h or x0 >= w:
        return patch

    crop = image[y0 : min(y0 + patch_size, h), x0 : min(x0 + patch_size, w)]
    patch[: crop.shape[0], : crop.shape[1]] = crop
    return patch


def crop_patch_pil(image: Image.Image, record: PatchRecord, patch_size: int) -> Image.Image:
    y0 = record.subtile_row * patch_size
    x0 = record.subtile_col * patch_size
    patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
    if y0 >= image.height or x0 >= image.width:
        return patch

    crop = image.crop((x0, y0, min(x0 + patch_size, image.width), min(y0 + patch_size, image.height)))
    patch.paste(crop, (0, 0))
    return patch


def prepare_gpu_batch(
    decoded,
    tile_size: int,
    image_size: int,
    device,
    torch,
    mean,
    std,
):
    batch_np = np.empty((len(decoded), tile_size, tile_size, 3), dtype=np.uint8)
    all_full_size = True
    for idx, (_record, image) in enumerate(decoded):
        h, w = image.shape[:2]
        if h == tile_size and w == tile_size:
            batch_np[idx] = image
        else:
            all_full_size = False
            batch_np[idx].fill(255)
            batch_np[idx, : min(h, tile_size), : min(w, tile_size)] = image[:tile_size, :tile_size]

    batch_cpu = torch.from_numpy(batch_np)
    if device.type == "cuda":
        batch_cpu = batch_cpu.pin_memory()

    batch = batch_cpu.to(device, non_blocking=True)
    batch = batch.permute(0, 3, 1, 2).to(dtype=torch.float32)
    batch = batch.div_(255.0)

    if (not all_full_size) or tile_size != image_size:
        batch = torch.nn.functional.interpolate(
            batch,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    return (batch - mean) / std


def iter_batches(items: list[PatchRecord], batch_size: int) -> Iterable[list[PatchRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def order_patches_for_batches(patches: list[PatchRecord], batch_order: str) -> list[PatchRecord]:
    if batch_order == "row-major":
        return patches
    if batch_order == "source-tile":
        return sorted(
            patches,
            key=lambda record: (
                record.source_row,
                record.source_col,
                record.subtile_row,
                record.subtile_col,
            ),
        )
    raise ValueError(f"unsupported batch_order: {batch_order}")


def run_slide_inference(
    plan: SlidePlan,
    output_path: Path,
    model,
    transform,
    device,
    torch,
    batch_size: int,
    decode_workers: int,
    compress: bool,
    output_dtype: str,
    preprocess: str,
    image_size: int,
    profile_sync: bool,
    prefetch_batches: int,
    batch_order: str,
) -> tuple[int, int]:
    embeddings: list[np.ndarray] = []
    tile_rc: list[tuple[int, int]] = []
    tile_xy: list[tuple[int, int]] = []
    source_tile_rc: list[tuple[int, int]] = []
    subtile_rc: list[tuple[int, int]] = []
    tissue_fraction: list[float] = []
    skipped_missing = 0
    mean = std = None
    if preprocess == "gpu":
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    with nvtx_range(torch, f"slide:{plan.rel_path}"):
        with FastPathPackReader(plan.slide_path, level=0) as reader:
            if reader.cols != plan.level0.cols or reader.rows != plan.level0.rows:
                raise ValueError(
                    f"level-0 index dimensions {reader.cols}x{reader.rows} do not match metadata "
                    f"{plan.level0.cols}x{plan.level0.rows}"
                )

            workers = max(1, decode_workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                def decode_one(source_row: int, source_col: int):
                    if preprocess == "gpu":
                        return decode_source_tile_cv2(reader, source_row, source_col)
                    return decode_source_tile_pil(reader, source_row, source_col)

                def submit_batch(records: list[PatchRecord]):
                    source_keys = list(dict.fromkeys((record.source_row, record.source_col) for record in records))
                    return records, [(key, pool.submit(decode_one, *key)) for key in source_keys]

                ordered_patches = order_patches_for_batches(plan.selected_patches, batch_order)
                batch_iter = iter(iter_batches(ordered_patches, batch_size))
                pending: list[tuple[list[PatchRecord], list]] = []

                def append_next_batch() -> bool:
                    try:
                        records = next(batch_iter)
                    except StopIteration:
                        return False
                    with nvtx_range(torch, "batch:decode_submit"):
                        pending.append(submit_batch(records))
                    return True

                append_next_batch()

                while pending:
                    records, futures = pending.pop(0)
                    while prefetch_batches > 0 and len(pending) < prefetch_batches:
                        if not append_next_batch():
                            break

                    with nvtx_range(torch, "batch:decode_wait"):
                        source_images = {}
                        missing_sources = set()
                        for key, future in futures:
                            image = future.result()
                            if image is None:
                                missing_sources.add(key)
                            else:
                                source_images[key] = image

                        decoded = []
                        for record in records:
                            key = (record.source_row, record.source_col)
                            image = source_images.get(key)
                            if image is None:
                                continue
                            if preprocess == "gpu":
                                decoded.append((record, crop_patch_np(image, record, plan.patch_size)))
                            else:
                                patch = crop_patch_pil(image, record, plan.patch_size)
                                decoded.append((record, transform(patch)))

                    skipped_missing += sum(
                        1 for record in records if (record.source_row, record.source_col) in missing_sources
                    )
                    if not decoded:
                        continue

                    batch_records = [item[0] for item in decoded]
                    with nvtx_range(torch, "batch:prepare"):
                        if preprocess == "gpu":
                            assert mean is not None
                            assert std is not None
                            batch = prepare_gpu_batch(
                                decoded,
                                plan.patch_size,
                                image_size,
                                device,
                                torch,
                                mean,
                                std,
                            )
                        else:
                            batch = torch.stack([item[1] for item in decoded]).to(device, non_blocking=True)
                        if profile_sync and device.type == "cuda":
                            torch.cuda.synchronize()

                    with nvtx_range(torch, "batch:forward"):
                        with torch.inference_mode():
                            if device.type == "cuda":
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    output = model(batch)
                            else:
                                output = model(batch)
                            if profile_sync and device.type == "cuda":
                                torch.cuda.synchronize()

                    with nvtx_range(torch, "batch:to_cpu"):
                        detached = output.detach()
                        if output_dtype == "float16":
                            detached = detached.to(dtype=torch.float16)
                        else:
                            detached = detached.float()
                        features = detached.cpu().numpy()
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        embeddings.append(features.astype(output_dtype, copy=False))

                    with nvtx_range(torch, "batch:metadata"):
                        for record in batch_records:
                            tile_rc.append((record.row, record.col))
                            tile_xy.append((record.col * plan.patch_size, record.row * plan.patch_size))
                            source_tile_rc.append((record.source_row, record.source_col))
                            subtile_rc.append((record.subtile_row, record.subtile_col))
                            tissue_fraction.append(record.tissue_fraction)

                    if prefetch_batches == 0 and not pending:
                        append_next_batch()

    if embeddings:
        embeddings_array = np.concatenate(embeddings, axis=0).astype(output_dtype, copy=False)
    else:
        embeddings_array = np.empty((0, 0), dtype=np.dtype(output_dtype))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".npz",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    save = np.savez_compressed if compress else np.savez
    try:
        with nvtx_range(torch, "slide:save_npz"):
            save(
                temp_path,
                embeddings=embeddings_array,
                tile_rc=np.asarray(tile_rc, dtype=np.int32),
                tile_xy=np.asarray(tile_xy, dtype=np.int32),
                source_tile_rc=np.asarray(source_tile_rc, dtype=np.int32),
                subtile_rc=np.asarray(subtile_rc, dtype=np.int32),
                tissue_fraction=np.asarray(tissue_fraction, dtype=np.float32),
                rel_path=np.asarray(plan.rel_path),
                slide_path=np.asarray(str(plan.slide_path)),
                source_file=np.asarray(plan.source_file),
                target_mpp=np.asarray(
                    plan.target_mpp if plan.target_mpp is not None else np.nan,
                    dtype=np.float32,
                ),
                source_mpp=np.asarray(
                    plan.source_mpp if plan.source_mpp is not None else np.nan,
                    dtype=np.float32,
                ),
                tile_size=np.asarray(plan.patch_size, dtype=np.int32),
                patch_size=np.asarray(plan.patch_size, dtype=np.int32),
                source_tile_size=np.asarray(plan.tile_size, dtype=np.int32),
                subtiles_per_tile=np.asarray(plan.subtiles_per_tile, dtype=np.int32),
                model_input_size=np.asarray(image_size, dtype=np.int32),
            )
            temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return embeddings_array.shape[0], skipped_missing


def write_manifest_row(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "rel_path",
        "source_file",
        "output_path",
        "selected_tiles",
        "embedded_tiles",
        "skipped_missing_tiles",
        "total_tiles",
        "tile_size",
        "source_tile_size",
        "subtiles_per_tile",
        "model_input_size",
        "level0_cols",
        "level0_rows",
        "tissue_threshold",
        "target_mpp",
        "source_mpp",
        "elapsed_s",
        "output_dtype",
        "preprocess",
        "prefetch_batches",
        "error",
    ]
    exists = manifest_path.exists()
    if exists:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            first_line = handle.readline()
        if first_line:
            current_header = next(csv.reader([first_line]))
            if current_header != fieldnames:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = manifest_path.with_name(f"{manifest_path.stem}.{stamp}.bak.csv")
                manifest_path.replace(backup_path)
                exists = False

    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/fastpath_d/thyroid"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/uni_embeddings"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Level-0 patch size for tissue filtering and UNI inference. Must divide the FastPATH tile size.",
    )
    parser.add_argument("--tissue-threshold", type=float, default=0.10)
    parser.add_argument("--mask-max-edge", type=int, default=4096)
    parser.add_argument(
        "--case-list",
        type=Path,
        default=None,
        help="Optional text file of case/patient IDs or relative path prefixes to process first.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-ignore", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--output-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--preprocess",
        choices=["gpu", "pil"],
        default="gpu",
        help="Use batched OpenCV decode plus tensor resize/normalize, or exact timm/PIL transforms.",
    )
    parser.add_argument(
        "--profile-sync",
        action="store_true",
        help="Synchronize after preprocessing and forward NVTX ranges for cleaner profiler attribution.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=1,
        help="Number of decoded batches to keep queued ahead of GPU work. Use 0 to disable.",
    )
    parser.add_argument(
        "--batch-order",
        choices=["row-major", "source-tile"],
        default="row-major",
        help="Patch ordering within inference batches. source-tile groups quadrants from the same FastPATH tile.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.decode_workers <= 0:
        raise SystemExit("--decode-workers must be positive")
    if args.patch_size <= 0:
        raise SystemExit("--patch-size must be positive")
    if args.tissue_threshold < 0 or args.tissue_threshold > 1:
        raise SystemExit("--tissue-threshold must be between 0 and 1")
    if args.mask_max_edge <= 0:
        raise SystemExit("--mask-max-edge must be positive")
    if args.prefetch_batches < 0:
        raise SystemExit("--prefetch-batches must be non-negative")

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.csv"
    case_ids = load_case_list(args.case_list)
    slides = discover_slides(input_root, include_ignore=args.include_ignore, case_ids=case_ids)
    if args.limit is not None:
        slides = slides[: args.limit]

    if case_ids is None:
        print(f"Found {len(slides)} H&E FastPATH slides under {input_root}")
    else:
        print(
            f"Found {len(slides)} H&E FastPATH slides under {input_root} "
            f"matching {len(case_ids)} case-list entries"
        )

    model = transform = device = torch = None
    if not args.dry_run:
        model, transform, device, torch = load_uni_model(args.model_id, args.device, args.hf_token_env)
        print(f"Loaded {args.model_id} on {device}")
        image_size = int(model.pretrained_cfg.get("input_size", [3, 224, 224])[-1])
    else:
        image_size = 224

    for slide_path in tqdm(slides, desc="Slides", unit="slide"):
        started = time.perf_counter()
        rel_path = str(slide_path.relative_to(input_root))
        output_path = output_path_for(output_dir, rel_path)

        try:
            with nvtx_range(torch, "slide:plan"):
                plan = build_slide_plan(
                    slide_path,
                    input_root,
                    tissue_threshold=args.tissue_threshold,
                    mask_max_edge=args.mask_max_edge,
                    patch_size=args.patch_size,
                )

            if output_path.exists() and not args.overwrite and not args.dry_run:
                status = "skipped_existing"
                embedded_tiles = ""
                skipped_missing = ""
            elif args.dry_run:
                status = "dry_run"
                embedded_tiles = ""
                skipped_missing = ""
            elif not plan.selected_patches:
                with nvtx_range(torch, "slide:save_empty_npz"):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        output_path,
                        embeddings=np.empty((0, 0), dtype=np.dtype(args.output_dtype)),
                        tile_rc=np.empty((0, 2), dtype=np.int32),
                        tile_xy=np.empty((0, 2), dtype=np.int32),
                        source_tile_rc=np.empty((0, 2), dtype=np.int32),
                        subtile_rc=np.empty((0, 2), dtype=np.int32),
                        tissue_fraction=np.empty((0,), dtype=np.float32),
                        rel_path=np.asarray(plan.rel_path),
                        slide_path=np.asarray(str(plan.slide_path)),
                        source_file=np.asarray(plan.source_file),
                        target_mpp=np.asarray(
                            plan.target_mpp if plan.target_mpp is not None else np.nan,
                            dtype=np.float32,
                        ),
                        source_mpp=np.asarray(
                            plan.source_mpp if plan.source_mpp is not None else np.nan,
                            dtype=np.float32,
                        ),
                        tile_size=np.asarray(plan.patch_size, dtype=np.int32),
                        patch_size=np.asarray(plan.patch_size, dtype=np.int32),
                        source_tile_size=np.asarray(plan.tile_size, dtype=np.int32),
                        subtiles_per_tile=np.asarray(plan.subtiles_per_tile, dtype=np.int32),
                        model_input_size=np.asarray(image_size, dtype=np.int32),
                    )
                status = "ok"
                embedded_tiles = 0
                skipped_missing = 0
            else:
                assert model is not None
                assert transform is not None
                assert device is not None
                assert torch is not None
                embedded_tiles, skipped_missing = run_slide_inference(
                    plan,
                    output_path,
                    model,
                    transform,
                    device,
                    torch,
                    batch_size=args.batch_size,
                    decode_workers=args.decode_workers,
                    compress=args.compress,
                    output_dtype=args.output_dtype,
                    preprocess=args.preprocess,
                    image_size=image_size,
                    profile_sync=args.profile_sync,
                    prefetch_batches=args.prefetch_batches,
                    batch_order=args.batch_order,
                )
                status = "ok"

            elapsed = time.perf_counter() - started
            with nvtx_range(torch, "slide:write_manifest"):
                write_manifest_row(
                    manifest_path,
                    {
                        "status": status,
                        "rel_path": plan.rel_path,
                        "source_file": plan.source_file,
                        "output_path": str(output_path),
                        "selected_tiles": len(plan.selected_patches),
                        "embedded_tiles": embedded_tiles,
                        "skipped_missing_tiles": skipped_missing,
                        "total_tiles": plan.total_patches,
                        "tile_size": plan.patch_size,
                        "source_tile_size": plan.tile_size,
                        "subtiles_per_tile": plan.subtiles_per_tile,
                        "model_input_size": image_size,
                        "level0_cols": plan.level0.cols,
                        "level0_rows": plan.level0.rows,
                        "tissue_threshold": args.tissue_threshold,
                        "target_mpp": plan.target_mpp,
                        "source_mpp": plan.source_mpp,
                        "elapsed_s": f"{elapsed:.3f}",
                        "output_dtype": args.output_dtype,
                        "preprocess": args.preprocess,
                        "prefetch_batches": args.prefetch_batches,
                        "error": "",
                    },
                )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            write_manifest_row(
                manifest_path,
                {
                    "status": "error",
                    "rel_path": rel_path,
                    "source_file": "",
                    "output_path": str(output_path),
                    "selected_tiles": "",
                    "embedded_tiles": "",
                    "skipped_missing_tiles": "",
                    "total_tiles": "",
                    "tile_size": "",
                    "source_tile_size": "",
                    "subtiles_per_tile": "",
                    "model_input_size": image_size,
                    "level0_cols": "",
                    "level0_rows": "",
                    "tissue_threshold": args.tissue_threshold,
                    "target_mpp": "",
                    "source_mpp": "",
                    "elapsed_s": f"{elapsed:.3f}",
                    "output_dtype": args.output_dtype,
                    "preprocess": args.preprocess,
                    "prefetch_batches": args.prefetch_batches,
                    "error": repr(exc),
                },
            )

    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
