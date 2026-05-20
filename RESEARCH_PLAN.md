# Research Plan, 2026-05-20

Public LB snapshot at time of writing:

```text
1. Jacoby Jaeger       8.239   9 submits   7h ago
2. Ehimen Nathaniel    8.801  17 submits   8h ago
3. Virtute             8.947  57 submits   2h ago
4. kitsune             9.057  53 submits   7h ago
5. Takahiro Saito      9.102  54 submits   8h ago
...
us (schema10 isolated): 10.084
```

This is not a "tune the model" gap. It is a different-method gap. Our
9 → 5 jump needs roughly 4.3 ft of RMS systematics removed, the 9 → 1 jump
roughly 5.8 ft. Public LB is computed on only 3 test wells and 14,151 rows,
so the bar is "solve three specific trajectories" much more than
"average-rank a GBM over 773 train wells".

## 1. Honest take on the solver pack we just shipped

We now have 24 direct-test-time variants (affine grid, GR/geo safe and bold,
tie-point, CEM, Stage1/Stage2, blends, consensuses) plus a Rust correction
kernel, matched-triple pseudo-public harness, and a bold-mode prediction guard.

I think the realistic ceiling of this pack, as-is, is roughly `9.6 - 9.8`
on public LB. It will not get us to `8.x`. Reasons:

- **Energy function is anchor-pinned.** Every family (`gr_safe`,
  `gr_bold`, `geo_*`, `cem_*`, `stage12_*`, all blends) carries an
  `anchor_weight` component or is constructed as
  `anchor + offset + slope*centered + curvature*shape`. Our anchor itself
  scores `10.084`. We are optimizing around a known-suboptimal point. Bold
  variants in the smoke run already drifted ~`85 ft` from the synthetic
  anchor, which means the energy disagrees with the anchor — but we then
  blend right back toward it.
- **Energy is built from the same priors GBM already saw.** GR/typewell
  match, geo-tail RMSE, slope penalty — that is more or less the schema14
  feature pack. Reranking with these priors will not introduce a signal
  the GBM did not already use; it will reduce variance but not bias.
- **CEM is the riskiest piece.** Five iterations × 320 candidates with no
  ground-truth-coupled energy is exactly the HMM-style failure mode the
  audit warned about: a feature/path family that the local diagnostic
  accepts but public LB punishes. The bold guard helps, but the bold guard
  is necessary, not sufficient.
- **Tie-point solver is largely cosmetic.** It needs strong GR landmarks
  with reliable monotonic typewell counterparts. In practice horizontal
  and typewell GR rarely line up cleanly across 6 formation surfaces; the
  smoke run picked too few landmarks to move the path.
- **Stage1/Stage2 is the most defensible family** (linear global + bounded
  knot refinement, max ±12 ft, smoothness penalty). It is the one family I
  would still submit. The rest I would keep as diagnostics only.

Plan item 4 (MTP CNN) is correctly deferred.

## 2. Why the top of the LB likely got there

Two distinguishable patterns:

- **Few submits, high rank.** Jaeger (8.239 in 9 submits) and Nathaniel
  (8.801 in 17). This is "they found the trick", not "they ground the
  board". Most likely culprits: a working public/external geological
  prior, a clean direct geometry-driven path, or a clean reproduction of a
  public notebook with stronger features than schema14.
- **Many submits, high rank.** Virtute (8.947 in 57) and kitsune (9.057
  in 53). Three test wells × 50+ submits is plenty of bandwidth to
  effectively binary-search per-well offsets. This is public-LB grinding
  and is probably not stable on private LB, but it sets the public bar we
  are being measured against.

We cannot replicate the grinding strategy (limited submits, no LB
oracle). We have to chase the "found the trick" pattern.

## 3. What I would actually do next

### Tier 0 — Cheap, must-do regardless

1. **Full train-eval with matched triples.**
   - Command:
     `make direct-solver-train-eval
     DIRECT_SOLVER_OUTPUT=artifacts/direct_solver_full
     DIRECT_SOLVER_PSEUDO_PUBLIC_TRIALS=200
     DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED=true
     DIRECT_SOLVER_PROGRESS_INTERVAL=10
     DIRECT_SOLVER_BACKEND=rust`.
   - Expected runtime: `2 - 3 h`.
   - Goal: get the honest per-family median/P90/win-rate vs `consensus_safe`
     anchor on signature-matched triples. Kill any family with
     `win_rate_vs_anchor < 0.55` or `p90_triple_rmse` worse than anchor by
     more than `0.3 ft`.

2. **One submit of `consensus_safe`** under strict-guard, but only as a
   sanity baseline — confirms the direct solver does not regress us
   below `10.10`. Burn one submit, get a stable anchor.

3. **Compare schema14 vs schema15 vs direct-solver `consensus_safe`** on
   the same matched-triple harness. If schema15 wins, the next move is
   GBM/feature work, not more solvers.

### Tier 1 — Most likely to actually unlock 8.x

4. **Cross-well typewell prior.** **STATUS: IMPLEMENTED.**
   Currently each test well is paired with one typewell. We have 773 train
   wells with full TVT/MD/Z. For each test well, find the `k` nearest
   train wells in `(X, Y, formation_surface_depths, GR statistics)` and
   build a synthetic typewell whose `TVT - last_known_tvt` is the median
   of those `k` train wells. This is exactly the kind of "found the
   trick" Jaeger could have.

   - Lives in `rogii/cross_well_prior.py`. Enabled with
     `--cross-well-prior --cross-well-k 8` (or
     `DIRECT_SOLVER_CROSS_WELL=true` which is now the Makefile default).
   - Produces three raw variants `crosswell_md_raw`, `crosswell_z_raw`,
     `crosswell_median` and two CEM-on-top variants
     `cem_over_crosswell_raw`, `cem_over_crosswell_top_median`.
   - Train-eval result is intentionally NOT a validation. On a 5-well
     slice these variants land at `15 - 53 ft` median triple RMSE vs
     `0.24 - 3.4 ft` for `geo_consensus`/`cem_raw`. Each train well has a
     near-perfect official typewell so train-eval rewards the wrong
     thing. The hypothesis is that on the 3 public test wells the
     official typewell is the weak signal and cross-well is the right
     one. This can only be tested by submitting.

5. **Direct geological-prior path solver.** **STATUS: IMPLEMENTED.**
   The plan's section "Azimuth / dip / formation-relative geological pack"
   was never built into a *path* solver; we built it into GBM features.
   The actual move was:
   ```text
   for each formation S in [ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA]:
       a_S, b_S = fit_on_train_tail(z_minus_S, TVT_input)
       cand_path_S = a_S * (z - S) + b_S    # extended to hidden rows
   pred = weighted_median(cand_path_S for S in formations,
                          weights = inverse_train_tail_rmse_S)
   ```
   This is now `fit_geo_candidate` returning both `best_path` and
   `consensus_path` (inverse-RMSE-weighted median of all formations).
   Surfaces with bad tail fit are down-weighted; surfaces that agree
   reinforce each other. Exposed as the `geo_consensus` variant.

6. **Per-well solver selection by signature.**
   Right now `consensus_safe` blends gr_safe + geo_safe + anchor for every
   well. Different wells are easy for different families. Build a small
   per-well router:
   ```text
   for each test well:
       compute signature (hidden_len, GR stats, MD/Z drift, formation
                          surface coverage, tail length)
       look up the nearest k train wells by signature
       pick the family that wins on those k train wells
       use that family's path for this test well
   ```
   This is essentially "model selection by example". Concrete and small.

   - Expected impact: `+0.1 - +0.4`.
   - Risk: low.

### Tier 2 — Worth doing, but not the main bet

7. **Strip anchor_weight from CEM/Stage2 energy in a "free" variant.**
   Run CEM with `anchor_weight = 0` (or `0.05`), score with GR/geo/slope
   only, then re-check with the bold prediction guard. This frees the
   solver from the suboptimal anchor without removing the safety net.

   - Expected impact: `+0.1 - +0.5` for the bold pack, but also wider
     variance.
   - Risk: high. Strict guard becomes mandatory.

8. **Public-discussion clean-room of a stage2-style notebook.**
   The earlier brief flagged a public discussion on
   `stage.1 global search using linear prior tvt = linear(md, z)` →
   `stage.2 iterative local search`. We implemented that. But the
   discussion may have additional context (knot choice, smoothness, energy
   formulation) that our reimplementation does not capture. Read the
   discussion(s) carefully and align the implementation, do not re-execute
   the notebook artifact.

   - Expected impact: hard to estimate, `+0.1 - +0.7`.
   - Risk: low (we already have the skeleton).

9. **Reweight per-well within a submission by signature confidence.**
   For each row, the prediction is a blend of variants. Today the blend is
   well-uniform. Make the blend depend on per-well signature (long-hidden
   wells favor stage12; short-hidden wells favor anchor; sparse-GR wells
   favor geo).

   - Expected impact: `+0.05 - +0.2`.
   - Risk: low.

### Tier 3 — Things I would explicitly NOT do next

10. **Do not add a sixth solver family.** The marginal return on a
    seventh, eighth tie-point variant is approximately zero given the
    anchor-pinned energy. The bottleneck is energy quality, not the
    number of parametric families.

11. **Do not start MTP CNN.** Six to twelve hours of training time, 773
    wells, and an energy oracle that may not transfer to private LB. The
    risk profile is worse than items 4 and 5.

12. **Do not Optuna-tune solver weights.** This is the exact trap the
    plan keeps flagging. If a tuned setting beats anchor on train-eval but
    we never had public confirmation, it is overfitting.

13. **Do not chase Virtute/kitsune by submit-grinding.** We do not have
    the bandwidth, and their result is probably not stable on private LB.

## 4. Submit budget reasoning

Assuming 4 - 6 remaining public submits in the relevant window:

```text
Submit 1: consensus_safe under strict guard       (sanity baseline)
Submit 2: best Tier-1 winner from matched triples (probably cross-well
                                                   typewell prior or
                                                   geological-prior solver)
Submit 3: stage12_raw OR consensus_bold_family under bold guard
          ONLY IF a Tier-1 method ranks above safe in matched triples
Submit 4: 0.4-blend(best Tier-1, consensus_safe)  (variance reduction)
```

I would not spend submits 5+ on solver-only changes. After submit 4 we
will either be in the top-10 region or we need to pivot to a fundamentally
new method.

## 5. Decision criteria for stopping

Stop pushing the solver direction and pivot if any of these become true:

- Submit 1 (`consensus_safe`) scores worse than `10.10` on public LB. That
  means the direct solver framework is broken end-to-end.
- The matched-triple harness ranks the safe anchor above every bold
  family. That means our energy is not informative.
- Submit 2 of a Tier-1 method does not score `<= 9.80`. That means even
  the most principled extension does not close the gap and a different
  research direction is needed.

In any of those cases the next move is:

- Re-examine the public discussion / external dataset landscape.
- Look hard at whether the GBM is even ranking `consensus_safe`-style
  candidates above schema10/15 internally — if not, the GBM is the
  bottleneck.
- Consider whether private LB looks structurally different from public
  (3 wells vs. many): protect against the case where Virtute/kitsune
  collapse on private.

## 6. What I am most worried about

- **Anchor lock-in.** Every safety mechanism we built (guard, gated blend,
  bold guard, consensus median) pulls the answer toward an anchor that is
  itself `10.084`. The further the true answer is from our anchor, the
  more harm our safety mechanisms do.
- **Energy/RMSE decorrelation on private.** Public is 3 wells, private is
  more. A solver that wins on energy could fail on private if the
  energy-vs-RMSE correlation is well-specific.
- **Path drift on cross-well typewell.** When the nearest train wells are
  spatially close but stratigraphically rotated, the synthetic typewell
  will bias the path in a structured way. Need to validate on
  train-eval before submit.
- **Sunk-cost on solver direction.** We just shipped 1,600 lines of
  solver code. The temptation to use them is strong. The honest answer is
  that Tier-1 work (cross-well typewell + geological-prior solver) is a
  bigger lever than reweighting what we have.

## 7. Concrete next 48 hours

```text
Day 1 morning:
  - run full matched-triple train-eval (Tier 0 item 1)
  - submit consensus_safe under strict guard (Tier 0 item 2)
  - read the public discussion thread carefully (Tier 2 item 8)

Day 1 afternoon:
  - implement cross-well typewell prior (Tier 1 item 4)
  - integrate into direct_solver as a new candidate family
  - run train-eval + matched triples for the new family

Day 1 evening:
  - if cross-well typewell beats anchor on matched triples and bold
    guard passes, submit it
  - else, implement direct geological-prior path (Tier 1 item 5)

Day 2 morning:
  - look at submit-1/submit-2 results
  - if we are at or below 9.6, refine the same direction
  - if we are still at 9.9+, pivot: check if there is a public
    notebook/dataset we are missing, look at the schema14/15 GBM
    output diff vs solver output

Day 2 afternoon:
  - prepare the final pair: one safe submit and one bold-blend submit
  - never re-tune weights based on the day-1 public LB
```

## 8. One-line summary

The solver work we shipped is correct as a defensive layer and as a
diagnostic toolbox. It is not the path to `8.x`. The path to `8.x` is
likely (a) a cross-well typewell prior or (b) a true geological-prior
path solver. Spend the next two days there, not on a seventh affine
variant.
