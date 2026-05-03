# UNI Inference Optimal Settings

This note records the current best settings for thyroid FastPATH UNI embedding
extraction on this workstation.

## Recommended Production Command

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

Do not use `--profile-sync` for production. It is only for Nsight Systems
attribution and adds synchronization points.

Omit `--case-list` when ready to continue through the entire H&E cohort.

The current inference unit is a `256 x 256` patch. Each `512 x 512` FastPATH
source tile is split into four quadrants; tissue filtering is performed on each
quadrant independently at the same `0.10` threshold, then each selected patch is
resized to UNI's `224 x 224` input.

## Chosen Settings

- `--batch-size 128`: best tested throughput on the RTX 5090. Larger batches
  (`256`, `512`) did not improve speed.
- `--decode-workers 8`: best practical decode parallelism. `16` was not faster;
  `4` was slower.
- `--patch-size 256`: splits each source `512 x 512` FastPATH tile into four
  inference patches and records the parent tile/quadrant metadata.
- `--output-dtype float16`: halves embedding storage with acceptable output
  format for downstream slide-level work.
- `--preprocess gpu`: uses OpenCV JPEG decode, then batched GPU resize and
  normalization. This was faster than the PIL/timm transform path.
- `--prefetch-batches 1`: overlaps CPU JPEG decode/I/O for the next batch with
  current-batch GPU work.

## Profiling Evidence

The Nsight/throughput numbers below were collected before the patch-size
correction, when the inference unit was still a full `512 x 512` FastPATH tile.
The command settings remain the baseline for the corrected `256 x 256` run, but
exact throughput should be re-profiled on patch-level inference.

Benchmark on the first 10 H&E slides:

| Setting | Tiles | Slide Time | Throughput |
| --- | ---: | ---: | ---: |
| `gpu`, prefetch off | 17,069 | 24.359 s | 700.7 tiles/s |
| `gpu`, `--prefetch-batches 1` | 17,069 | 22.675 s | 752.8 tiles/s |

Focused Nsight Systems run on the first 5 H&E slides:

| Setting | Tiles | Slide Time | Throughput |
| --- | ---: | ---: | ---: |
| sync profile, prefetch off | 9,366 | 13.808 s | 678.3 tiles/s |
| sync profile, prefetch on | 9,366 | 12.640 s | 741.0 tiles/s |

Final Nsight artifacts:

```text
outputs/nsys_prefetch/uni_gpu_limit5_prefetch_sync.nsys-rep
outputs/nsys_prefetch/uni_gpu_limit5_prefetch_sync.sqlite
```

## Notes

Nsight CPU IP/context-switch sampling was blocked by the current kernel perf
configuration (`perf_event_open` unavailable, paranoid level `4`). CUDA, NVTX,
OS runtime calls, Python sampling, and PyTorch module ranges were captured.

The optimized GPU preprocessing path was compared against the exact PIL/timm
path on the pilot slide. Mean embedding cosine similarity was `0.99956`, with
1st percentile cosine similarity `0.99837`.
