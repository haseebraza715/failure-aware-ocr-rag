# Supervisor handoff

Short operational handoff for the FAAR AAAI experiments. Copy-paste commands,
recovery, and the full checklist live in [RUNBOOK.md](RUNBOOK.md).

## What is ready

The `faar-aaai-experiments` branch is ready for a bounded CUDA calibration on a
shared cluster. Local tests pass. GPU and process-memory budgets, single-GPU
Slurm templates, resumable checkpoints, pinned model revisions, and fail-closed
split checks are in place.

| Stage | Status |
| --- | --- |
| Local implementation and regression tests | Ready |
| Supervisor handoff for the 108-page calibration | Ready |
| Full OHR validation preparation | Waiting for calibration approval |
| Full B0-B4 validation runs | Not run |
| OHR test, ArXivQA, and MP-DocVQA paper runs | Not run |

Prototype 40-example mock-backend results in the repository are not paper
baselines.

## What remains

1. Run allocated-GPU preflight and return the report.
2. Run the 108-page calibration and return the summary plus projection.
3. Wait for approval before any validation preparation or baseline job.
4. After approval: sharded validation prep, merge, manifest registration, then
   a 50-100 question pilot, then ordered B0-B4.

## Repository and branch

Repository: <https://github.com/haseebraza715/FailSafeRAG>

```bash
git clone https://github.com/haseebraza715/FailSafeRAG.git faar
cd faar
git checkout faar-aaai-experiments
git rev-parse HEAD
```

Record that SHA in every returned calibration artifact.

The supplied bounded templates are Slurm. The code can read PBS allocation
metadata, but there is no bounded PBS calibration template yet. Confirm the
scheduler before adapting anything.

## Dataset paths and environment

Minimum inputs for calibration:

- `split.json` and `config/split_checksums.json` (tracked, immutable)
- `OHR-Bench/data/qas_v2.json`
- `OHR-Bench/data/retrieval_base/gt/`
- the OHR PDF archive at `data/ohr_bench_raw/pdfs.zip`, or `FAAR_PDF_ROOT` /
  `FAAR_PDF_ZIP`

Do not copy a ~200 GB expanded corpus before measuring the pipeline. The
development archive is about 1.4 GB and is ignored by Git, so it must be
transferred separately.

Copy `.env.example` to `.env`. Calibration needs paths and resource names, not
paid API keys.

| Variable | Role |
| --- | --- |
| `FAAR_PROJECT_ROOT` | Checkout directory |
| `FAAR_SCRATCH` | Fast scratch for outputs and caches |
| `FAAR_OUT_ROOT` | Asset output root, defaults to scratch |
| `HF_HOME` | Hugging Face cache on scratch |
| `FAAR_PDF_ROOT` / `FAAR_PDF_ZIP` | PDF source if not at the default zip |
| `FAAR_DOCUMENT_INVENTORY` | Per-document page inventory |
| `FAAR_GPU_BUDGET_GB` | Required positive GiB VRAM budget |
| `HF_TOKEN` | Only if preflight reports a gated-model failure |

`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are not required for calibration. They
are first needed by paid VLM stages B1, B2, and B4.

## Shared-server memory and thread limits

The launcher fails closed without a positive `FAAR_GPU_BUDGET_GB` chosen from
the allocated-GPU preflight. It also requires the budget plus a co-tenant
reserve (`FAAR_MIN_GPU_FREE_GB`, 20% of visible VRAM by default) to fit in free
VRAM.

When unset, the launcher derives:

- `FAAR_MAX_RSS_GB`: 90% of the scheduler or cgroup RAM allocation
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: half the allocated CPUs

The calibration template requests 1 GPU, 8 CPUs, and 32G RAM. Exceeding the GPU
or RSS budget aborts the job with a checkpoint. Do not raise those limits by
editing code.

## Preflight

Create the pinned environment once:

```bash
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
cp .env.example .env
```

Login node, no CUDA:

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Exit 0 is ready, 1 is blocking, 2 is warnings only. Then edit the Slurm
placeholders and run the allocated-GPU preflight:

```bash
sbatch cluster/templates/slurm_preflight.sbatch
```

Choose `FAAR_GPU_BUDGET_GB` from that report before calibration.

## Bounded 108-page calibration

One complete document (`manual/User_Manual_1500S_Classic_EN`, 108 pages) through
PDF extraction, Docling, page rendering, and pinned GOT-OCR. Calibration
evidence only. Never a paper result.

```bash
sbatch cluster/templates/slurm_calibration_108.sbatch
```

Do not submit validation preparation or a baseline job at the same time.

## Resume and preemption

Slurm sends SIGTERM 120 seconds before walltime. The job stops at the next
page or document boundary, writes the checkpoint, and exits 143. Re-submit the
same script. Completed stages are skipped.

A SIGKILL or node failure leaves a running attempt. Resume marks it unclean and
sets `timing_complete=false`. Supply the scheduler wall time from `sacct` or
`qstat` as `--scheduler-elapsed-sec` on `prepare_benchmark_assets.py` before
building the summary. Details are in RUNBOOK section 8.

## Outputs to return

| File | What it records |
| --- | --- |
| `cluster/preflight_<jobid>.json` | Allocated GPU, VRAM, CPU, RAM, disk |
| `results/environment/preflight_calibration.json` | Launcher preflight for the calibration job |
| `results/calibration/faar-ohr-108/prepare_checkpoint.json` | Per-stage runtime, peak RSS, OCR throughput, storage |
| `results/calibration/calibration_summary.json` | Commit SHA, GPU, RAM, runtime, storage, hashes |
| `results/calibration/preparation_projection.json` | Sized validation and test estimates |

Build the last two with the commands in RUNBOOK section 9. Do not send `.env`,
keys, raw PDFs, or model caches.

## How to report measurements

The calibration summary is the measurement record. It includes commit SHA, GPU
model, peak RSS, per-stage and total runtime, per-page OCR throughput, generated
storage, cache growth, resume flag, `timing_complete`, and split or manifest
hashes. `total_wall_sec` is present only when every attempt has measured or
scheduler-supplied elapsed time. If `timing_complete=false`, correct the
checkpoint first. Do not guess wall time from the start timestamp.

## Merging preparation shards

This step is after calibration is approved, not part of the first job. Each
validation shard writes `shard_manifest_shardNofM.json`. When all shards finish:

```bash
.venv-aaai/bin/python cluster/merge_prep_shards.py \
  --project-root "$PWD" \
  --split val \
  --out "$FAAR_SCRATCH/faar-ohr-val/merged_assets.json" \
  "$FAAR_SCRATCH/faar-ohr-val"/shard_manifest_shard*.json
```

Then register the locked manifest with `register_benchmark_assets.py` as in
RUNBOOK section 10. The merge refuses overlapping shards, missing indexes, or a
document set that does not match `split.json`.

## Stop conditions

Stop and return the logs if any of these occurs:

- Preflight exits 1.
- The repository commit or split checksum differs from the recorded run.
- The GPU budget plus co-tenant reserve exceeds free VRAM.
- The process raises CUDA out-of-memory or exceeds its RSS budget.
- A checkpoint reports `timing_complete=false` and scheduler wall time is not
  available.
- Scratch usage or projected validation storage exceeds the lab quota.

Do not start full validation until the calibration summary and projection have
been approved.
