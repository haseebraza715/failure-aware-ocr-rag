# FAAR AAAI Experimental Execution Plan

This plan mirrors the fixed Phase 0-7 experimental plan. Each phase is a checkpoint: implement, run, verify against the phase success criteria, commit, then continue.

## Phase 0: Setup

- Fix the OHR-Bench train/val/test split with seed 42 and save it to `split.json`.
- Treat `split.json` as immutable after creation.
- Add dataset registry entries for OHR-Bench, MP-DocVQA val, and ArXivQA val.
- Wire the recommended stack through the existing interfaces:
  - VLM: `gpt-4o-2024-11-20` (user-approved Alternate A)
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
- Commit as `phase0: complete protocol setup`.

Status (2026-07-23):
- **Implementation complete:** all three dataset registration paths, shared-corpus retrieval, pinned model loaders, Docling adapter, visual baseline dispatch, OpenAI request accounting, four-metric evaluation, and model provenance are implemented.
- **Locally verified:** focused protocol tests and the full suite pass (`72 passed`); Python compilation, editable package build/import, and `git diff --check` pass.
- **Split protected:** `split.json` remains unchanged, its deterministic rewrite guard accepts the existing split, and SHA-256 is `64583a532c5db5aa31e4cbb5cd9c7d894c7a2d5e8aa49f1a7f6041f54e714f53`.
- **Model preflight:** all six locked Hugging Face revisions resolve and are accessible; the selected OpenAI snapshot is fixed to `gpt-4o-2024-11-20`. No paid experiment was run.
- **Not execution-ready:** complete OHR-Bench/MP-DocVQA/ArXivQA OCR and image manifests are absent. The current `.venv` is also not the locked AAAI environment: final `pip check` reports missing packages and version mismatches, including absent Docling, RAGAS, W&B, and Label Studio. Phase 0 must not be reported as a completed paper run until a fresh environment installs `constraints-aaai.txt` successfully and the external assets are available.

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

Implementation status: complete; tables and Figure 1 require completed Phase 4-6 artifacts. The source-tree anonymization scan is clean except for intentional checklist text and local `.git` metadata; PDF metadata cannot be checked until `paper.pdf` exists and `pdfinfo` is available.
