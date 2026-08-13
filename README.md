# Failure-aware OCR-RAG

Failure-Aware Agentic Recovery (FAAR) for OCR-RAG document question answering.

## Current research status

The AAAI experiment workflow lives on the `faar-aaai-experiments` branch. It
is ready for a supervisor to run the bounded 108-page CUDA calibration on a
shared cluster. It is not yet ready for full validation or paper-result claims.
Those require calibration measurements, approval of the projected resource
plan, a representative pilot, and then the ordered B0-B4 runs.

- Supervisor starting point: [SUPERVISOR_HANDOFF.md](SUPERVISOR_HANDOFF.md)
- Exact shared-cluster procedure: [RUNBOOK.md](RUNBOOK.md)
- Experiment protocol and gates: [docs/faar-aaai-plan.md](docs/faar-aaai-plan.md)
- Reproducibility rules: [docs/aaai-reproducibility.md](docs/aaai-reproducibility.md)

The older 40-example prototype reports under `docs/reports/` are development
history. Their mock-backend numbers are not the AAAI B0-B4 baseline results.

## Overview

FAAR is a failure-aware OCR-RAG pipeline for document QA. It uses text-first retrieval by default, then applies targeted recovery only when quality signals indicate likely OCR-related failure.

Key recovery types:
- `semantic`: retry retrieval when evidence is likely missing or mismatched
- `word_level`: correct OCR noise before answer generation
- `structural`: fall back to selective visual reasoning for layout-heavy failures

Research objective:
- improve answer quality on OCR-heavy inputs
- reduce unnecessary multimodal cost compared to always-on visual pipelines

## Local prototype setup

Prerequisites:

- Python 3.12+
- phase assets under `data/phase0/` and `artifacts/phase0/`

Install:

```bash
python -m pip install -e .
```

Basic verification:

```bash
python -m pytest
```

## Local prototype usage

Run one end-to-end example:

```bash
faar-demo run-example --example-id 446d159e-b5c2-45dc-91cc-faaa931f3649 --project-root . --vlm-backend mock --seed 42 --output logs/phase1/phase1_e2e_latest.json
```

Notes:

- `vlm-backend=mock` is the default for offline reproducibility
- outputs are written to `logs/` and phase artifacts under `artifacts/`

Do not use this demo command for paper baselines. Cluster and paper runs use
the pinned environment and ordered runner documented in [RUNBOOK.md](RUNBOOK.md).

## Architecture

- Top-level architecture summary: [ARCHITECTURE.md](ARCHITECTURE.md)
- Detailed handbook view: [docs/repo_handbook/architecture_overview.md](docs/repo_handbook/architecture_overview.md)

## Repository Structure

- `src/faar/`: controller, quality, retrieval, recovery, answering, CLI
- `tests/`: unit and integration coverage
- `data/phase0/`: sampled benchmark metadata and manual labels
- `artifacts/`: phase artifacts and summary files
- `logs/`: per-run structured outputs by phase
- `docs/`: modular phase and repository documentation
- `OHR-Bench/`: benchmark/evaluation subproject

## Documentation

- Docs home: [docs/index.md](docs/index.md)
- Phase docs: [docs/phases/index.md](docs/phases/index.md)
- Repo handbook: [docs/repo_handbook/index.md](docs/repo_handbook/index.md)
- Reports: [docs/reports/index.md](docs/reports/index.md)
- Archives: [docs/archives/index.md](docs/archives/index.md)
