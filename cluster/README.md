# Cluster handoff

## Environment

Create the pinned CPython 3.12 environment and verify it:

```bash
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
mkdir -p results/environment
.venv-aaai/bin/python -m pip freeze > results/environment/pip-freeze.txt
```

A paper run is invalid if `pip check` fails or the freeze file is missing; see
`docs/aaai-reproducibility.md`. The first command on the shared server is a
read-only hardware preflight. It does not download the benchmark, load FAAR,
load a model, or use an API key:

```bash
.venv-aaai/bin/python cluster/preflight.py --out cluster/preflight.json
```

Send `cluster/preflight.json` back with the GPU model and available VRAM, CPU
RAM or scheduler limit, free disk, CUDA version, and scheduler name. Do not
send `.env` files, API keys, or model caches. The preflight reads only an
explicit whitelist of environment variables and redacts secret-shaped strings
before printing or writing the report; the `--out` write is atomic.

The report includes:

- parsed Slurm metadata (job id/name, partition, QOS, CPUs, memory, walltime,
  and requested/on-node GPU counts) and PBS metadata (job id/name, queue, CPU
  count, memory, GPU count, walltime);
- cgroup v1/v2 current, peak, and limit memory;
- `ulimit` soft/hard limits;
- scratch (`TMPDIR`, `TMP`, `SCRATCH`, ...) and cache (`HF_HOME`,
  `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`) paths with
  existence, writability, and free-space checks.

## Dry run

Before any billed or GPU time, print the exact ordered B0-B4 commands with a
one-example dry run:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench --split test \
  --max-examples 1 --dry-run
```

`--dry-run` only prints each stage's launcher command; it writes nothing, needs
no API key, and does not require the B2 gate lock.

## Ordered baseline runner

`cluster/run_baselines.py` runs B0, B1, B2, B3, B4 in order, launching each
stage as a fresh subprocess through `cluster/launcher.py` and validating every
written result before the next stage:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench --split test \
  --resume
```

Stages and boundaries:

- B0 `--gate off --recovery off`: text-only RAG; runs without an OpenAI key.
- B1 `--recovery always_vlm`: first `OPENAI_API_KEY` boundary; the launcher
  refuses to start without the key required by `VLM_BACKEND`.
- B2 `--gate on --recovery random_type`: requires
  `config/gate_threshold.json` locked from validation-only results
  (`source_split` `val`, precision >= 0.75, recall >= 0.70); run
  `tune_gate.py` on validation results first. The gate is never tuned on test
  data.
- B3 `--mode colpali`: visual retrieval baseline.
- B4 `--mode visrag`: visual retrieval baseline.

The runner prints `[run-baselines] launch <stage>` before each stage and
`[run-baselines] done <stage>` after validation; `--dry-run` shows the same
commands without running them.

Resume behavior: without `--resume` the runner refuses to start if any
`results/baselines/<dataset>/<split>/{b0,b1,b2,b3,b4}.json` exists. With
`--resume`, completed stages are re-validated and skipped, and each missing
stage runs. `--resume` is forwarded to `run.py` for B0..B4, so all stages
reuse completed per-example checkpoints.

Output validation: each stage result must carry that stage's profile and
label, a summary with `EM`, `F1`, `vlm_rate`, `harm_rate`, API requests,
tokens, `cost_usd`, and `runtime_sec`, full `run_spec` provenance (dataset,
split, seed,
`max_examples`, embedding, reranker, OCR, model revisions, unsharded), rows
that exactly match B0's example ids, and summary values recomputed from the
rows. Any mismatch stops the run. Because each stage runs in a fresh
subprocess, per-stage memory is released before the next stage starts.

## Launcher

`cluster/launcher.py` is the generic entry point for one bounded job, spawned
per stage by `run_baselines.py`. It is pure-stdlib, never installs packages,
never copies datasets, and never overrides `CUDA_VISIBLE_DEVICES`.

```bash
.venv-aaai/bin/python cluster/launcher.py \
  --project-root "$PWD" \
  --gate off --recovery off --split test \
  --out results/b0.json
```

It runs preflight, then:

1. Requires exactly one logical GPU unless `--cpu-only` is passed; a
   multi-GPU or GPU-less allocation is rejected with exit code 2.
2. Requires an explicit positive `FAAR_GPU_BUDGET_GB` chosen after the
   preflight shows the GPU model and free VRAM. The budget is never inferred
   from VRAM headroom or model size. For backward compatibility an explicitly
   supplied `FAAR_MAX_GPU_MEMORY_FRACTION` in `(0, 1]` is accepted instead and
   translated to an equivalent budget; with neither variable set the launch
   fails closed. Setting both variables is rejected as ambiguous.
3. Rejects a budget above the GPU's total VRAM and any non-finite or
   non-positive value; the requested budget is never silently clamped.
4. Requires the initial free VRAM to cover the process budget plus the
   co-tenant reserve (`FAAR_MIN_GPU_FREE_GB`, a 20% reserve by default).
5. Derives guards only when absent:
   - `FAAR_MAX_RSS_GB` (90% of the scheduler/cgroup CPU-RAM budget, else 50%
     of physical memory),
   - `FAAR_MIN_GPU_FREE_GB` (a 20% co-tenant reserve),
   - `FAAR_MAX_GPU_MEMORY_FRACTION` (`FAAR_GPU_BUDGET_GB` / total VRAM),
   - `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (half the allocated CPUs).
6. Validates the Hugging Face cache and, only for a run that can invoke a VLM,
   a non-empty key required by `VLM_BACKEND` (`OPENAI_API_KEY` for OpenAI or
   `ANTHROPIC_API_KEY` for Claude). B0 does not require either key. Key values
   are never printed.
7. Spawns `run.py` with a plain argv list (no shell interpolation) and
   forwards `SIGTERM`/`SIGINT` to the child, propagating its exit code.

Every launch writes a redacted, job-namespaced report under
`results/environment/`; override this with `--preflight-out`.
For deliberate reuse of an already-recorded preflight, set
`FAAR_PREFLIGHT_JSON` to the report path; the launcher will not re-probe the
host. `FAAR_PYTHON` overrides the interpreter used for `run.py`.

## Scheduler templates

Editable one-GPU templates run the full ordered baseline runner:

- `cluster/templates/slurm_one_gpu.sbatch`: `--gpus=1`, bounded `--mem` and
  `--time`, and `--signal=TERM@120` so Slurm delivers SIGTERM 120 s before
  walltime.
- `cluster/templates/pbs_one_gpu.pbs`: `ngpus=1` and bounded `ncpus`/`mem`/
  `walltime`; its optional TORQUE `softwalltime` directive is disabled until
  the target cluster confirms support.

Both templates allocate exactly one GPU and invoke `cluster/run_baselines.py`
with `--resume` and `--project-root` set to the submission directory. They use
the configured `FAAR_PYTHON`; `FAAR_DATASET` and `FAAR_SPLIT` default to
`ohrbench` and `test`. They never set `CUDA_VISIBLE_DEVICES` because Slurm and
PBS manage the allocation. Keep the resource boundaries and change scheduler
directives only after the lab allocation is known. Set `FAAR_GPU_BUDGET_GB` to
a positive budget chosen from the preflight report; the launcher fails closed
without it.

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

## Multi-GPU sharding (text baselines B0-B2)

The recommended scaling strategy is one process per GPU with deterministic
shards; DDP/DataParallel add no value for this embarrassingly parallel
retrieval workload. Each shard writes per-example checkpoints and its own
shard result file, so shards can be resumed independently and merged later.

```bash
# one job per GPU, e.g. 8 GPUs:
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i .venv-aaai/bin/python run.py \
    --gate off --recovery off --dataset ohrbench --split test \
    --shard-index $i --num-shards 8 \
    --out results/b0.json > results/b0_shard$i.log 2>&1 &
done
wait

# merge the shards into the single result the analysis pipeline expects:
.venv-aaai/bin/python cluster/merge_shards.py \
  --out results/b0.json \
  --baseline results/b0.json \
  results/b0_shard*of8.json
```

Rules:

- Every shard must be launched with the same `--seed`, `--num-shards`,
  dataset, split, and model flags; `merge_shards.py` rejects shards whose
  run_specs differ or whose example ids overlap.
- The merged file resets `shard_index`/`num_shards` to null and recomputes the
  summary (EM, F1, vlm_rate, API totals) and `harm_rate` from the merged rows,
  so it passes the normal stage validation on later `--resume` runs.
- B2 shards need the matching B0 shard as `--baseline` (run_spec comparison
  includes the shard index); merge the B0 shards first, then pass the merged
  B0 to the B2 merge.
- B3/B4 (ColPali/VisRAG) are single-GPU per job: `run.py` rejects shard flags
  for visual modes because corpus embeddings are built once per process. The
  corpus-embedding cache (below) keeps a resumed visual run from re-encoding
  the corpus.

## Persistent corpus-embedding cache

Both the text retriever and the visual retrievers persist corpus embeddings
under `<cache_dir>/` so that resumes and shards skip the encode step:

- text: `<cache_dir>/text_embeddings/<key>.npz` + `<key>.meta.json`
- visual: `<cache_dir>/visual_embeddings/<key>.npz` + `<key>.meta.json`

The key covers the model, revision, dtype, and the exact corpus identity
(chunk ids + texts, or the image path list). Any mismatch recomputes; a hit
reuses bitwise-identical embeddings. `cache_dir` defaults to
`<project_root>/cache` and can be moved with `FAAR_CACHE_DIR`. Concurrent
writers produce identical content, so atomic last-writer-wins is safe.

## Runtime configuration

All of the following are environment variables with validated values; the
launcher derives the memory/GPU ones when unset:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FAAR_GPU_BUDGET_GB` | required for GPU runs | explicit per-process VRAM budget in GiB |
| `FAAR_MAX_GPU_MEMORY_FRACTION` | derived from budget | per-process CUDA memory fraction in `(0,1]` |
| `FAAR_MIN_GPU_FREE_GB` | 20% of GPU VRAM | co-tenant free-VRAM reserve |
| `FAAR_MAX_RSS_GB` | 90% of scheduler/cgroup RAM | in-process RSS fail-fast guard |
| `FAAR_EMBED_BATCH_SIZE` | 2 | corpus/query embedding batch size |
| `FAAR_VISUAL_BATCH_SIZE` | 1 | ColPali/VisRAG corpus-encode batch size |
| `FAAR_VISUAL_SCORE_BATCH_SIZE` | 8 | ColPali score batch size |
| `FAAR_VLM_MAX_RETRIES` | 3 | total attempts per VLM call; retries only on 429/5xx/timeouts |
| `FAAR_VLM_RETRY_BACKOFF_SECONDS` | 2.0 | exponential backoff base with jitter |
| `FAAR_VLM_TIMEOUT_SECONDS` | 60 | per-request VLM timeout |
| `FAAR_CACHE_DIR` | `<project_root>/cache` | corpus-embedding cache root |
| `FAAR_LOG_INTERVAL_SECONDS` | 60 | progress-line interval |
| `FAAR_LOG_INTERVAL_ITEMS` | 25 | progress-line item interval |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | half the allocated CPUs | thread caps |

CUDA out-of-memory during corpus encoding or scoring halves the batch size
automatically (floor of 1) and retries the same slice after releasing the
CUDA cache; the event is printed and the reduced size is kept for the rest of
the process. A batch of 1 that still OOMs fails the job rather than hiding the
error.

A job interrupted by SIGTERM/SIGINT stops at the next example boundary,
leaves all completed per-example checkpoints intact, and exits 130/143 without
publishing a partial result file. Per-example failures (bad data, exhausted
VLM retries) are recorded in a failed checkpoint with the error message and
the run continues; the run then fails at the end with the exact list of failed
example ids, so nothing is silently skipped and a single bad example cannot
waste a multi-hour job.
