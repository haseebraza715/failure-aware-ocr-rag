# FAAR AAAI Experimental Execution Plan

This plan mirrors the fixed Phase 0-7 experimental plan. Each phase is a checkpoint: implement, run, verify against the phase success criteria, commit, then continue.

## Phase 0: Setup

- Fix the OHR-Bench train/val/test split with seed 42 and save it to `split.json`.
- Treat `split.json` as immutable after creation.
- Add dataset registry entries for OHR-Bench, MP-DocVQA val, and ArXivQA val.
- Wire the recommended stack through the existing interfaces:
  - VLM: `claude-sonnet-4-5`
  - Embeddings: `NV-Embed-v2`
  - Reranker: `bge-reranker-v2-m3`
  - OCR: `GOT-OCR 2.0`
  - PDF pre-processing: `Docling`
  - Visual RAG base: `ColPali`
  - Eval framework: `RAGAS`
  - Experiment tracking: `Weights & Biases`
  - Annotation tool: `Label Studio`
- Add API call counting and cost logging before any paid VLM call.
- Add `evaluate.py`, which always emits `EM`, `F1`, `vlm_rate`, and `harm_rate`.
- Verify the setup commands and required outputs.
- Commit as `phase0: setup`.

## Phase 1: Baselines

- Run B0 text-only RAG, gate off, recovery off.
- Run B1 always-VLM.
- Run B2 random recovery with diagnosis off and random recovery type.
- Run B3 ColPali.
- Run B4 VisRAG.
- Use one worker subagent per independent baseline.
- Wait for all baseline outputs before moving on.
- Verify outputs are saved and FAAR can later be compared against B2.
- Commit as `phase1: baselines`.

## Phase 2: Quality Gate Specification

- Use the BGE-reranker-v2-m3 top-1 score as the gate signal.
- Run the theta grid on the validation split only: `0.3`, `0.4`, `0.5`, `0.6`, `0.7`.
- Log precision, recall, and F1 for each theta.
- Lock the winning theta in config and write it verbatim to `RUNLOG.md`.
- Verify gate precision is at least `0.75` and gate recall is at least `0.70`.
- Commit as `phase2: gate threshold`.

Implementation status: complete; execution is pending a real validation B0 result file produced with the pinned stack.

## Phase 3: Annotation Study

- Sample 100 examples where B0 got the wrong answer.
- Extract raw OCR text with GOT-OCR 2.0.
- Prepare Label Studio import assets with exactly the four label choices: `semantic`, `word_level`, `structural`, `other`.
- Stop for real double annotation by humans rather than fabricating labels.
- After labels exist, compute Cohen's kappa and require `kappa >= 0.65`.
- Commit prepared annotation assets as `phase3: annotation setup`.

Implementation status: complete; execution is pending B0 failures, GOT-OCR runtime/model access, and two independent human annotations.

## Phase 4: Main FAAR Runs

- Run FAAR on OHR-Bench test split.
- Run FAAR on MP-DocVQA val split.
- Run FAAR on ArXivQA val split.
- Log EM, F1, `vlm_rate`, `cost_usd`, and `runtime_sec` for each run.
- Verify all result JSON files exist and contain the required metrics.
- Commit as `phase4: main faar runs`.

Implementation status: complete; execution is pending full registered OHR-Bench/MP-DocVQA/ArXivQA OCR and image assets plus the pinned-model credentials.

## Phase 5: Ablations

- Run A1 no gate, always recover.
- Run A2 no diagnosis, random recovery type.
- Run A3 no word-level LLM, SymSpell fallback.
- Run A4 no semantic retry.
- Use one worker subagent per independent ablation.
- Verify all ablation rows are saved.
- Commit as `phase5: ablations`.

Implementation status: complete; execution is pending the locked gate, complete benchmark assets, and pinned-model credentials.

## Phase 6: Analysis

- Compute bootstrap 95% confidence intervals.
- Compute per-failure-type breakdowns for `semantic`, `word_level`, and `structural`.
- Compute harm rate.
- Generate the Plotly cost-accuracy Pareto figure and export it as PDF.
- Verify numbers are loaded from result JSON files.
- Commit as `phase6: analysis`.

Implementation status: complete; execution is pending real saved result JSON files and the Plotly/Kaleido PDF export dependency.

## Phase 7: Paper Tables and Pre-Submission Checklist

- Generate Table 1 main results from result JSON files.
- Generate Table 2 ablations from result JSON files.
- Generate Table 3 per-failure-type results from result JSON files.
- Run the before-submitting checklist verbatim:
  - grep for `haseebraza`
  - grep for `your-real-name`
  - check PDF metadata author field
  - confirm anonymization checklist items
- Commit as `phase7: paper tables`.
