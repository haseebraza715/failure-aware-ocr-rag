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
