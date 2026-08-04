# Phase 3 Results

## Scope

Results below are aggregated from the current 40-example committed
`logs/phase3/` offline fixture artifacts using the Phase 3 metrics protocol.
The earlier five-row pilot table is historical and has been superseded by this
reproducible aggregation.

## Main Metrics Table

| Profile | Count | NDCG@5 | Recall@5 | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `faar_full` | 40 | 0.4789 | 0.5250 | 0.1250 | 0.2036 |
| `faar_no_backtrack` | 40 | 0.4789 | 0.5250 | 0.1250 | 0.2036 |
| `faar_no_vlm` | 40 | 0.4789 | 0.5250 | 0.1250 | 0.2036 |
| `faar_no_diagnosis` | 40 | 0.4789 | 0.5250 | 0.1250 | 0.2036 |
| `naive_rag` | 40 | 0.4789 | 0.5250 | 0.1250 | 0.2036 |

## Retrieval and QA Artifact References

- `artifacts/phase3/metrics_summary.json`
- `artifacts/phase3/retrieval_metrics.csv`
- `artifacts/phase3/qa_metrics.csv`

## Notes

- Current run slice uses mock visual fallback and is intended to validate full pipeline wiring and reporting.
- Larger-scale benchmark claims should rely on extended Phase 3 runs with broader manifest coverage.
