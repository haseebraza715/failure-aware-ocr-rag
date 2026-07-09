# FAAR AAAI Run Log

| Timestamp UTC | Phase | Command | Result |
| --- | --- | --- | --- |
| 2026-07-06T23:02:47Z | preflight | `cat docs/faar-aaai-plan.md` | Missing at requested root path; user supplied attached spec instead. |
| 2026-07-06T23:02:47Z | preflight | `git switch -c faar-aaai-experiments` | Created and switched to requested experiment branch from clean `main`. |
| 2026-07-06T23:02:47Z | phase0 | explorer subagent survey | Completed read-only survey of current mock VLM, run/evaluate equivalents, and Phase 0 gaps. |
| 2026-07-06T23:02:47Z | phase0 | `.venv/bin/python split_dataset.py --seed 42 --out split.json` | Created immutable OHR-Bench split: train 5948, val 1274, test 1276. |
| 2026-07-06T23:02:47Z | phase0 | `.venv/bin/python split_dataset.py --seed 42 --out split.json` | Verified existing `split.json` unchanged; script refused to rewrite different content by design. |
| 2026-07-06T23:02:47Z | phase0 | `.venv/bin/python evaluate.py logs/phase3/naive_rag/76550cb6-e96e-4c38-8ad6-f55d9cbc4414.json` | Emitted required metrics: `EM`, `F1`, `vlm_rate`, `harm_rate`. |
| 2026-07-06T23:02:47Z | phase0 | `.venv/bin/python -m pytest -q` | Passed: 28 tests, 3 warnings. |
| 2026-07-06T23:02:47Z | phase1 | `.venv/bin/python run.py --gate off --recovery off --out results/b0.json` | Blocked: `NV-Embed-v2` was not accessible through Hugging Face; `results/b0.json` was not created. |
| 2026-07-06T23:02:47Z | phase1 | `VLM_BACKEND=claude-sonnet-4-5 ... .venv/bin/python run.py --gate off --recovery always_vlm --out results/b1.json` | Blocked: missing `ANTHROPIC_API_KEY`; `results/b1.json` was not created. |
| 2026-07-06T23:02:47Z | phase1 | `.venv/bin/python run.py --gate on --recovery random_type --out results/b2.json` | Blocked: `NV-Embed-v2` was not accessible through Hugging Face; `results/b2.json` was not created. |
| 2026-07-06T23:02:47Z | phase1 | `VLM_BACKEND=claude-sonnet-4-5 ... .venv/bin/python run.py --mode colpali --out results/b3.json` | Blocked: missing `ANTHROPIC_API_KEY`; `results/b3.json` was not created. |
| 2026-07-06T23:02:47Z | phase1 | `VLM_BACKEND=claude-sonnet-4-5 ... .venv/bin/python run.py --mode visrag --out results/b4.json` | Blocked: missing `ANTHROPIC_API_KEY`; `results/b4.json` was not created. |
| 2026-07-09T00:00:00Z | phase2 | `tune_gate.py` implementation and `.venv/bin/python -m pytest -q` | Implemented validation-only BGE top-1 reranker threshold search over 0.3/0.4/0.5/0.6/0.7, lock-file writer, and provenance guard. Tests passed: 31. Execution is pending a real B0 validation result; no threshold or metrics were fabricated. |
| 2026-07-09T00:00:00Z | phase3 | annotation tooling and `.venv/bin/python -m pytest -q` | Implemented deterministic B0 failure sampling, GOT-OCR 2.0 extraction, Label Studio task/config generation, and two-export Cohen's kappa verification. Tests passed: 34. Execution is pending B0 failures, GOT-OCR model runtime, and real independent human labels. |
| 2026-07-09T00:00:00Z | phase4 | benchmark runner implementation and `.venv/bin/python -m pytest -q` | Implemented complete benchmark asset manifests, immutable OHR split registration, dataset/split provenance, per-run VLM cost deltas, and complete-asset guards. Tests passed: 36. No main-run metrics were fabricated; execution is pending assets and credentials. |
| 2026-07-09T00:00:00Z | phase5 | A1-A4 worker implementation review and `.venv/bin/python -m pytest -q` | Implemented no-gate forced recovery, no-diagnosis seeded random recovery, local SymSpell word-level fallback without ByT5 construction, and no-semantic-retry routing. Four worker subagents supplied independent contracts; tests passed: 44. Execution is pending real assets, the locked gate, and credentials. |
