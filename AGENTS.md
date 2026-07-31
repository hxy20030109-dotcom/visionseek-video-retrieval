# VisionSeek Working Instructions

Read `PROJECT_PLAN.md`, `docs/ARCHITECTURE.md`, and
`.codex/PROJECT_CONTEXT.md` before making project-level decisions.

## Non-negotiable rules

- Never invent experiment results, speedups, dataset sizes, or hardware claims.
- Keep the public repository reproducible and recruiter-readable.
- Do not commit datasets, checkpoints, secrets, generated caches, or private career context.
- Attribute every adapted idea and every copied or modified source file.
- Do not copy code from a repository unless its license permits the intended use.
- Prefer a small, tested implementation over adding another framework or feature.
- Every model change must have a baseline, a metric, and an ablation.
- Update `.codex/PROJECT_CONTEXT.md` after a meaningful milestone.

## Current project boundary

The core contribution is a budget-aware router for a two-stage bilingual
text-to-video retrieval system. Foundation-model pretraining, custom CUDA
kernels, a large web frontend, and SOTA claims are out of scope.
