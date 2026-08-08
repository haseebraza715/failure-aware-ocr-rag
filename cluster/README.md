# Cluster handoff

The first command on the shared server is a read-only hardware preflight. It
does not download the benchmark, load FAAR, load a model, or use an API key.

```bash
python3 cluster/preflight.py --out cluster/preflight.json
```

Send `cluster/preflight.json` back with the GPU model and available VRAM, CPU
RAM or scheduler limit, free disk, CUDA version, and scheduler name. Do not
send `.env` files, API keys, or model caches.

Until the preflight is known, use conservative environment values for any
visual smoke run:

```bash
export CUDA_VISIBLE_DEVICES=0
export FAAR_VISUAL_BATCH_SIZE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Fill these in after preflight; they are fail-fast guards, not replacements
# for the scheduler's hard memory limit.
# export FAAR_MAX_RSS_GB=<agreed CPU-RAM budget in GiB>
# export FAAR_MIN_GPU_FREE_GB=<GPU reserve in GiB>
```

Run one process on one GPU. Do not launch B0--B4 in parallel. The visual
retrievers process images in bounded chunks and keep the default batch size at
one so the first job is safe on an unknown shared machine.

`FAAR_MAX_RSS_GB` stops the process before another inference step when recorded
peak CPU memory is already above the configured budget. `FAAR_MIN_GPU_FREE_GB`
stops before a batch when the requested GPU reserve is unavailable. Replace the
example values with limits agreed with the lab; do not guess them for a paper
run.

## Staged execution

1. Run the preflight and a 100-page GPU calibration. This produces throughput
   and memory evidence, not paper results.
2. Run an end-to-end pilot on complete documents covering roughly 50--100
   questions. This is still an engineering check.
3. Prepare the full OHR validation assets: 1,274 questions, 549 documents, and
   7,033 pages. Use this split for real B0 and gate tuning.
4. Run B0, B1, B2, B3, and B4 sequentially, saving each result before the next.
5. Run the full OHR test split: 1,276 questions, 567 documents, and 6,842
   pages.
6. Run ArXivQA only after all 500 page mappings are human-confirmed. MP-DocVQA
   is a separate job after its official validation data is acquired.

The dataset is staged once and reused. Do not make a separate copy of the
page images for each baseline. The repository's locked split and model
revisions remain the source of truth.
