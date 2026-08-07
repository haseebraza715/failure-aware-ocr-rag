# FailSafeRAG

RAG that detects its own OCR failures and recovers, paying for the expensive visual path only when needed.

[![License: none](https://img.shields.io/badge/license-none-yellow)](https://github.com/haseebraza715/FailSafeRAG)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

<video controls autoplay muted loop playsinline width="100%" src="https://github.com/haseebraza715/FailSafeRAG/raw/main/docs/demo.mp4"></video>

Prefer a GIF? [docs/demo.gif](docs/demo.gif)

## Try it in 60 seconds

```bash
git clone https://github.com/haseebraza715/FailSafeRAG.git
cd FailSafeRAG
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
./scripts/demo.sh
```

Fully offline and deterministic: local-hash embeddings, a mock VLM, and ByT5 from the local cache. Four real questions run through the `faar-demo` CLI; the walkthrough narrates every quality signal and routing decision.

**What you'll see:**

- The quality gate fires on **low lexical evidence** → semantic recovery re-queries → the correct answer at **0 multimodal tokens**.
- A **clean example passes the gate** and answers directly, with no recovery and no multimodal spend.
- A **routing table** shows word-level and structural routes; only the structural case would invoke a VLM (mock here, zero spend).

## What it does

- **Text-first retrieval**: hybrid BM25 + dense retrieval over OCR text; images are never touched on the default path.
- **Quality gate**: scores evidence for OCR word noise, layout damage, and weak lexical evidence before trusting an answer.
- **Semantic recovery**: re-queries with context-anchored evidence when retrieval is empty or mismatched.
- **Word-level recovery**: corrects OCR noise (ByT5); degrades to a guarded skip when the model is unavailable offline.
- **Structural recovery**: a *selective* visual fallback for layout-heavy failures, routed to a VLM only when text cannot answer.
- **Cost control**: multimodal tokens are spent only on gated failures; clean questions cost nothing.

## How it works

Text-first retrieval → the quality gate scores the evidence (lexical, OCR corruption, layout) → a typed recovery is routed and applied, or the answer is produced directly. Recovery is *failure-aware*: it spends compute only when a signal says the evidence cannot be trusted.

| Fact | Detail |
| --- | --- |
| Language | Python 3.12+ |
| Dependencies | faiss-cpu, langgraph, pydantic, rank-bm25, typer; `[ml]` extra adds sentence-transformers, transformers |
| Offline? | Yes. Demo runs with `HF_HUB_OFFLINE=1`, mock VLM, no API keys |
| Status | Research prototype, phased |
| License | None specified |

## Honest research stance

This is an in-progress research prototype. The checked-in tables describe an offline fixture run and are **not** evidence of live multimodal model quality.

## Links

- [Architecture](ARCHITECTURE.md)
- [Benchmark](OHR-Bench/)
- [Evidence manifest](evidence-manifest.tsv)
