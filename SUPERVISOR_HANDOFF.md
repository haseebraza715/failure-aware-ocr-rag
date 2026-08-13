# Supervisor handoff

This page is the short operational handoff for the FAAR AAAI experiments. The
full command sequence and recovery instructions are in [RUNBOOK.md](RUNBOOK.md).

## What is ready

The `faar-aaai-experiments` branch is ready for a bounded CUDA calibration on
a shared cluster. The code has explicit GPU and process-memory budgets,
single-GPU scheduler templates, resumable checkpoints, atomic outputs, pinned
model revisions, dataset integrity checks, and fail-closed shard merging.

The current gate is intentionally narrow:

| Stage | Status |
| --- | --- |
| Local implementation and regression tests | Ready |
| Supervisor handoff for the 108-page calibration | Ready |
| Full OHR validation preparation | Waiting for calibration approval |
| Full B0-B4 validation runs | Not run |
| OHR test, ArXivQA, and MP-DocVQA paper runs | Not run |

The repository contains older 40-example mock-backend results. Treat them as
prototype evidence only. They are not full-scale AAAI baselines.

## Repository and branch

Repository: <https://github.com/haseebraza715/failure-aware-ocr-rag>

```bash
git clone https://github.com/haseebraza715/failure-aware-ocr-rag.git faar
cd faar
git checkout faar-aaai-experiments
git rev-parse HEAD
git status --short --branch
```

Record the commit SHA in every returned calibration artifact. Do not run the
cluster workflow from `main` unless the research branch has first been merged
and revalidated.

## What the supervisor needs to provide

- The scheduler type, partition, account, QOS, and maximum wall time.
- One allocated NVIDIA GPU for preflight and calibration.
- The allowed CPU count, RAM, scratch path, and scratch quota.
- A process VRAM budget chosen after the allocated-GPU preflight.
- Hugging Face access if the preflight reports a gated-model failure.

The calibration does not need an OpenAI or Anthropic key. Those keys are first
needed by paid VLM baseline stages.

The supplied bounded preflight and calibration templates use Slurm. The code
can read PBS allocation metadata, and the repository has a generic PBS
one-GPU baseline template, but it does not yet have a bounded PBS calibration
template. Confirm the cluster scheduler before handoff. Do not adapt the full
baseline template into a calibration job without review.

## Minimum data for calibration

Do not copy a 200 GB expanded corpus before measuring the pipeline. The first
job needs the tracked split, QA metadata and page inventory, plus the OHR PDF
archive. The archive used during development is about 1.4 GB and is ignored by
Git, so it must be transferred separately to
`data/ohr_bench_raw/pdfs.zip` or supplied through `FAAR_PDF_ROOT` or
`FAAR_PDF_ZIP`.

The calibration processes one complete 108-page document. Its report measures
wall time, per-stage time, OCR throughput, peak RSS, GPU details, generated
storage, and model-cache growth. Those measurements determine whether and how
the 7,037-page validation split should be staged.

## First run only

```bash
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
cp .env.example .env
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Fill only paths and resource settings in `.env`. Never commit it. Then edit
the scheduler placeholders in `cluster/templates/slurm_preflight.sbatch` and
`cluster/templates/slurm_calibration_108.sbatch` for the lab.

Inside an allocated GPU job, run the preflight first:

```bash
sbatch cluster/templates/slurm_preflight.sbatch
```

After the preflight is reviewed and `FAAR_GPU_BUDGET_GB` is chosen, submit only
the bounded calibration:

```bash
sbatch cluster/templates/slurm_calibration_108.sbatch
```

Do not submit validation preparation or a baseline job at the same time.

## Files to return

Return these files without `.env`, keys, raw PDFs, or model caches:

1. `cluster/preflight_<jobid>.json`
2. `results/environment/preflight_calibration.json`
3. `results/calibration/faar-ohr-108/prepare_checkpoint.json`
4. `results/calibration/calibration_summary.json`
5. `results/calibration/preparation_projection.json`

The last two are generated with the commands in RUNBOOK sections 8 and 9.
Wait for the research team to approve the projection before preparing the full
validation split.

## Stop conditions

Stop and return the logs if any of these occurs:

- Preflight exits 1.
- The repository commit or split checksum differs from the recorded run.
- The GPU budget plus co-tenant reserve exceeds free VRAM.
- The process raises CUDA out-of-memory or exceeds its RSS budget.
- A checkpoint reports `timing_complete=false` and scheduler wall time is not
  available.
- Scratch usage or projected validation storage exceeds the lab quota.

Resume and hard-kill correction are documented in [RUNBOOK.md](RUNBOOK.md).
