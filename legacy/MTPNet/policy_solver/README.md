# Policy Solver

`policy_solver` is the clean infrastructure wrapper around the current
chunk-ranker + DP selector. It exists so the deployable path policy has one
auditable place to live while the older research modules remain available.

The pipeline boundary is:

```text
candidates -> features -> models -> DP -> report
```

## What It Does

1. Builds a schema-safe candidate bank: base paths, shifts, drifts, slope
   continuations, residual-stack paths, and optional diagnostic candidates.
2. Builds per `well x hidden-run x chunk x candidate` features without hidden
   target leakage.
3. Trains a CatBoost ranker/cost policy grouped by chunk.
4. Runs Viterbi/DP over chunks with switch and boundary penalties.
5. Writes row predictions, metrics, and a Markdown report.

## Schema-Safe Contract

Deployable features must not include train-only columns such as `TVT`,
`Geology`, or raw formation columns. Candidate losses, oracle labels, and tail
classes are diagnostic-only: they may be used for evaluation and reports, but
they must not enter inference features.

Optional SoftSegment and location-aware correlation artifacts are currently
treated as weak diagnostic features/candidates. Smoke runs showed they do not
yet improve the chunk policy, so they should be kept behind explicit inputs
until a fold-safe gain is proven.

## Commands

Run through the dedicated facade:

```bash
uv run --extra dev python -m policy_solver.train \
  --data-dir data/train \
  --output-dir artifacts/policy_solver_v0
```

Or via Make:

```bash
make policy-solver
```

The old command remains available:

```bash
make chunk-policy
```

## Current Status

This directory is intentionally a thin facade over `mtpnet.chunk_policy` and
`mtpnet.candidate_bank`. The next cleanup step is to move implementation pieces
behind this package one at a time after tests lock down behavior.
