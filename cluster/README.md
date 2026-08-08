# Cluster handoff

The first command on the shared server is a read-only hardware preflight. It
does not download the benchmark, load FAAR, load a model, or use an API key.

```bash
python3 cluster/preflight.py --out cluster/preflight.json
```

Send `cluster/preflight.json` back with the GPU model and available VRAM, CPU
RAM or scheduler limit, free disk, CUDA version, and scheduler name. Do not
send `.env` files, API keys, or model caches. The preflight reads only an
explicit whitelist of environment variables and redacts secret-shaped strings
before printing or writing the report; the `--out` write is atomic.

Beyond the basics, the report now carries:

- parsed Slurm metadata (job id/name, partition, QOS, CPUs, memory, walltime,
  and requested/on-node GPU counts) and PBS metadata (job id/name, queue, CPU
  count, memory, GPU count, walltime);
- cgroup v1/v2 current, peak, and limit memory;
- `ulimit` soft/hard limits;
- scratch (`TMPDIR`, `TMP`, `SCRATCH`, ...) and cache (`HF_HOME`,
  `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`) paths with
  existence, writability, and free-space checks.

## Launcher

`cluster/launcher.py` is the generic entry point for one bounded job. It is
pure-stdlib, never installs packages, never copies datasets, and never
overrides `CUDA_VISIBLE_DEVICES`.

```bash
./.venv/bin/python cluster/launcher.py \
  --project-root "$PWD" \
  --gate on --recovery off --split test \
  --out results/b1.json
```

It runs preflight, then:

1. Requires exactly one logical GPU unless `--cpu-only` is passed; a
   multi-GPU or GPU-less allocation is rejected with exit code 2.
2. Derives conservative guards only when absent:
   - `FAAR_MAX_RSS_GB` (90% of the scheduler/cgroup CPU-RAM budget, else 50%
     of physical memory),
   - `FAAR_MIN_GPU_FREE_GB` (a 20% co-tenant reserve),
   - `FAAR_MAX_GPU_MEMORY_FRACTION` (at most 50%, reduced by the reserve when the GPU is partly occupied),
   - `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (half the allocated CPUs).
3. Refuses a GPU with less than 30% free VRAM. It validates the Hugging Face
   cache and, only for a run that can invoke a VLM, a non-empty key required by
   `VLM_BACKEND` (`OPENAI_API_KEY` for OpenAI or `ANTHROPIC_API_KEY` for
   Claude). B0 does not require either key. Key values are never printed.
4. Spawns `run.py` with a plain argv list (no shell interpolation) and
   forwards `SIGTERM`/`SIGINT` to the child, propagating its exit code.

Every launch writes a redacted, job-namespaced report under
`results/environment/`; override this with `--preflight-out`.
For deliberate reuse of an already-recorded preflight, set
`FAAR_PREFLIGHT_JSON` to the report path; the launcher will not re-probe the
host. `FAAR_PYTHON` overrides the interpreter used for `run.py`.

## Scheduler templates

Editable one-GPU templates call the launcher directly:

- `cluster/templates/slurm_one_gpu.sbatch` — `--gpus=1`, bounded `--mem` and
  `--time`, and `--signal=TERM@120` so Slurm delivers SIGTERM 120 s before
  walltime.
- `cluster/templates/pbs_one_gpu.pbs` — `ngpus=1` and bounded `ncpus`/`mem`/
  `walltime`; its optional TORQUE `softwalltime` directive is disabled until
  the target cluster confirms support.

Both templates allocate exactly one GPU and never set `CUDA_VISIBLE_DEVICES`
(Slurm and PBS manage the allocation themselves). Edit the budget knobs and
the `run.py` arguments; do not guess paper values.

Credentials are read from the submit environment or a `.env` at the repository
root. Never write keys into a template file.

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
