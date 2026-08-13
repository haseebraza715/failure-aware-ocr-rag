# Documentation index

Current AAAI cluster work starts at the repository root:

- [Supervisor handoff](../SUPERVISOR_HANDOFF.md)
- [Shared-cluster runbook](operations/runbook.md)

Those two files are the operational source of truth. Everything below is
supporting or historical.

## Current

| Path | Contents |
| --- | --- |
| [architecture/overview.md](architecture/overview.md) | Current system design |
| [experiments/aaai-plan.md](experiments/aaai-plan.md) | Fixed experimental protocol and B0-B4 order |
| [experiments/aaai-reproducibility.md](experiments/aaai-reproducibility.md) | Environment, model pins, and identity checks |
| [../cluster/README.md](../cluster/README.md) | Launcher, templates, and scheduler mechanics |
| [../annotation/README.md](../annotation/README.md) | Label study commands, after a real B0 exists |

## Historical

Do not use these as cluster instructions. They record the prototype workstream
and dated execution notes.

| Path | Contents |
| --- | --- |
| [history/](history/) | Superseded plans, run log, portfolio write-up |
| [phases/](phases/index.md) | Phase folders from the prototype |
| [reports/](reports/index.md) | Prototype phase reports, including 40-example mock numbers |
| [archives/](archives/index.md) | Indexes of committed logs and artifacts |
| [repo_handbook/](repo_handbook/index.md) | Older modular handbook |
| [repo_architecture/](repo_architecture/index.md) | Older Diataxis layout |

## Evidence and paper

| Path | Contents |
| --- | --- |
| [evidence/evidence-manifest.tsv](evidence/evidence-manifest.tsv) | Provenance of committed prototype assets |
| [paper/faar_aaai_findings.tex](paper/faar_aaai_findings.tex) | Draft findings note, not a submitted paper |

## Command-line tools

User-facing commands are grouped under `scripts/` by purpose: experiments,
data preparation, annotation, smoke checks, and release checks. Cluster launchers
and scheduler templates remain under `cluster/`.
