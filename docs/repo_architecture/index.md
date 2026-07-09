# Repository Architecture Docs (Legacy)

This folder documents the FAAR repository using a Diataxis-oriented layout:

- Tutorial: learn by doing
- How-to: complete specific tasks
- Reference: exact components and interfaces
- Explanation: design rationale and architecture

## Navigation

Canonical modular docs entrypoint:

- [Docs Home](/docs/index.md)

- [Repository Handbook Index](/docs/repo_handbook/index.md)
- [Architecture Overview](/docs/repo_handbook/architecture_overview.md)
- [Runtime Components](/docs/repo_handbook/runtime_components.md)
- [Phase 2 Methodology Formalization](/docs/phases/phase2/methodology_formalization.md)
- [Phase 2 Code-to-Math Mapping](/docs/phases/phase2/code_math_mapping.md)
- [Phase 2 Acceptance Checklist](/docs/phases/phase2/acceptance_checklist.md)
- [Documentation Style Standard](/docs/repo_handbook/documentation_style_standard.md)

## Documentation Map

```mermaid
flowchart TD
    index[Index]
    tutorial[Tutorial]
    howto[HowTo]
    reference[Reference]
    explanation[Explanation]
    index --> tutorial
    index --> howto
    index --> reference
    index --> explanation
```


