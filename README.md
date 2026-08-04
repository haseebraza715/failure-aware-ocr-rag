# FailSafeRAG

Failure-Aware Agentic Recovery (FAAR) for OCR-RAG document question answering.

## Overview

FAAR is a failure-aware OCR-RAG pipeline for document QA. It uses text-first retrieval by default, then applies targeted recovery only when quality signals indicate likely OCR-related failure.

Key recovery types:
- `semantic`: retry retrieval when evidence is likely missing or mismatched
- `word_level`: correct OCR noise before answer generation
- `structural`: fall back to selective visual reasoning for layout-heavy failures

Research objective:
- improve answer quality on OCR-heavy inputs
- reduce unnecessary multimodal cost compared to always-on visual pipelines

## Setup

Prerequisites:
- Python 3.12+
- phase assets under `data/phase0/` and `artifacts/phase0/`

Install:

```bash
python -m pip install -e .
```

The base install covers the deterministic offline path (local-hash embeddings,
mock VLM). The neural components — sentence-transformers embeddings and ByT5
word-level correction — live in the `ml` extra:

```bash
python -m pip install -e ".[ml]"
```

Basic verification:

```bash
python -m pytest
```

## Usage

Run one end-to-end example:

```bash
faar-demo run-example --example-id 446d159e-b5c2-45dc-91cc-faaa931f3649 --project-root . --vlm-backend mock --seed 42 --output logs/phase1/phase1_e2e_latest.json
```

Notes:
- `vlm-backend=mock` is the default for offline reproducibility
- retrieval defaults to the deterministic `local-hash-v1` embedder, so mock
  mode performs no Hugging Face/model lookup; set
  `retrieval.embedding_backend="sentence-transformers"` in code for research
  runs that intentionally use the pinned external model revision
- outputs are written to `logs/` and phase artifacts under `artifacts/`

## Architecture

- Top-level architecture summary: [ARCHITECTURE.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/ARCHITECTURE.md)
- Detailed handbook view: [docs/repo_handbook/architecture_overview.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/docs/repo_handbook/architecture_overview.md)

## Repository Structure

- `src/faar/`: controller, quality, retrieval, recovery, answering, CLI
- `tests/`: unit and integration coverage
- `data/phase0/`: sampled benchmark metadata and manual labels
- `artifacts/`: phase artifacts and summary files
- `logs/`: per-run structured outputs by phase
- `docs/`: modular phase and repository documentation
- `OHR-Bench/`: benchmark/evaluation subproject

## Documentation

- Docs home: [docs/index.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/docs/index.md)
- Phase docs: [docs/phases/index.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/docs/phases/index.md)
- Repo handbook: [docs/repo_handbook/index.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/docs/repo_handbook/index.md)
- Reports: [docs/reports/index.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/docs/reports/index.md)
- Archives: [docs/archives/index.md](https://github.com/haseebraza715/FailSafeRAG/blob/main/docs/archives/index.md)

## Reproducible environment and evidence

Use `constraints-py312.txt` with Python 3.12 for the verified dependency set.
Model identity and benchmark asset provenance are recorded in
[`docs/reproducibility.md`](docs/reproducibility.md), and every committed log or
artifact is classified by [`evidence-manifest.tsv`](evidence-manifest.tsv).
The checked-in Phase 3 tables describe an offline 40-example fixture run; they
are not evidence of live multimodal model quality.
