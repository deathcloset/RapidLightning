# Rapid Lightning — the record

> Purpose: after a chaotic, multi-context development process, this file is the
> single canonical account of **what each notebook actually is**, **which methods
> are the "best,"** and **what the licensing picture looks like** — written down so
> we stop relitigating it. Everything below is sourced from the notebooks
> themselves (their narrative, their measured outputs, and their code).

## TL;DR

- **Best original method (clean IP):** **Rapid Lightning v3 — head-aware.**
- **Best accuracy / most rigorously validated / most publication-ready:** **Northside (v4)** — but **AGPL-3.0** because it bundles XGBoost.
- v3 and Northside are **not competitors — they're a stack.** Northside *contains* v3's head-aware GP as one of its two experts.
- The other four notebooks are supporting cast: v1/v2 (foundational, superseded), XGForge (a reusable tool), symbolic_regression_v9 (reference research / negative result).

---

## What each notebook is

| Notebook | One-line identity | Engine / deps | Self-contained? | Status |
|---|---|---|---|---|
| `methods/rapid_lightning_v3.ipynb` | **Head-aware** GP — members evolved as *signal generators* for an LR head | Pure NumPy + scikit-learn (XGBoost optional, baseline only) | ✅ Yes (Colab top-to-bottom) | **Best original method** |
| `methods/northside.ipynb` | **v4** — sparsely-gated MoE: head-aware GP **+** XGBoost, joined by a competence gate | NumPy + XGBoost + LightGBM + treelite/tl2cgen | ✅ Yes (needs the deps) | **Best accuracy; AGPL-3.0** |
| `foundations/rapid_lightning_v1.ipynb` | Base production training pipeline: GP ensemble + LR head, accuracy-mode fitness | torch + numba | ✅ Embeds its 4 source files | Foundational, superseded on method |
| `foundations/rapid_lightning_v2.ipynb` | Adds `predict_proba` output + the "Gorilla sweep" capacity finding | Imports external `src/rapid_lightning/` package | ❌ Needs external package | Foundational, superseded; fuzzing-project context |
| `tooling/XGForge.ipynb` | XGBoost→C compiler (`xgforge`) + `forge_logistic` / `forge_lasso` | XGBoost | ✅ Yes | Reusable tool (powers Northside's "RLNF" mode) |
| `research/symbolic_regression_v9.ipynb` | Closes a v3→v9 GP-evolution-speedup study; "rigidity finding" | Pure NumPy | ✅ Yes | Reference research / honest negative result |

---

## The version story (memory vs. what the notebooks say)

- **v1 — the base method & tech.** ✅ Production training pipeline. GP ensemble + a
  **logistic-regression head**, *accuracy-mode* fitness (members **are** predictors,
  trained via boosting). torch+numba engine, AOT-compiled to a ~19 KB dependency-free
  `.so` at sub-300 ns. Four embedded source files: `gp_ensemble`, `jit_kernel`,
  `bridge`, `lightning_aot`.

- **v2 — *not* new genome operators.** The notebook's actual headline contribution is
  **`predict_proba` emitted alongside `predict_one`** (real-valued probability scores,
  needed for top-K ranking in a syzkaller fuzzing hot path), plus the **"Gorilla sweep"**
  capacity finding (hitting the quality bar required *combining* more members + gens +
  population + depth, not any single knob). It imports an external `src/rapid_lightning/`
  package, so it is the **least self-contained** of the six.

- **v3 — head-aware (the key idea).** ✅ Members are evolved to maximize
  `corr(signal ⟂ pool, residual)²` — functional gradient boosting in feature space —
  i.e. each tree adds a coordinate the LR head can use, **not** a prediction of its own.
  Individually the trees are near-chance (measured: mean member acc ≈ 0.573, diversity
  ≈ 0.487); together they feed the head. The notebook calls this **"the best objective
  we found."** Pure NumPy reference.

- **v4 / Northside — gated hybrid, soft blend (not a hard fallback).** ✅ A
  *sparsely-gated mixture of two experts* — the head-aware GP and an XGBoost model —
  joined by a **competence gate**. The recommended serving mode (`proba_soft`) is a
  **soft blend**, a refinement of the "predict with trees, fall back to XGBoost"
  mental model. `RLN` = the gated hybrid; `RLNF` = same, with XGBoost compiled via
  XGForge into the nanosecond envelope.

## Are v3 and Northside the "best"? — Yes, with two clarifications

### 1. They are a stack, not rivals
Northside literally uses v3's head-aware GP as one of its two experts, then gates it
against XGBoost. So "v3 vs Northside" is really "the clean GP core" vs "the GP core +
XGBoost, gated."

### 2. They win on different axes

**Rapid Lightning v3 — best *original* method, clean IP**
- Project's own repeated verdict: head-aware is the strongest objective found.
- XGBoost is used **only as a baseline to beat** (optional import), never in the method.
- Measured (synthetic, head-aware reference budget): acc 0.886 / ROC-AUC 0.950 /
  PR-AUC 0.955 at 385 nodes; **Pareto-dominates the size-matched XGBoost** (0.875 acc
  @ 317 nodes) — smaller *and* better on continuous data. On heavily categorical data
  it ties `soft` / trails full XGBoost (a basis-match thing).
- Compiles to a ~15 KB `.so`, ~50 ns/row (~20 M inferences/sec/core).
- ⚠️ Caveat (the notebook's own honesty): this is the **NumPy reference at modest
  budget**, not the production engine. The bit-exact **production-AOT port at ship
  budget (N≥25)** is the open #1 lead — that's what turns these reference numbers into
  a shipped benchmark.

**Northside (v4) — best accuracy, most rigorous, most publication-ready**
- The **only** method here that actually **beats XGBoost** — conditionally.
- Across **19 OpenML binary datasets**: significantly beats XGBoost on **6**, loses on
  **1** (phoneme, −0.0024, negligible), ties on **12** (paired bootstrap **and** DeLong).
  Wins (+0.015 to +0.023) cluster where XGBoost has **headroom** (corr +0.76); the gate's
  separability does **not** predict the gain (corr −0.30).
- **Speed/size/energy win is unconditional**; the accuracy edge is conditional on headroom.
- Already ships the publication apparatus: a "will it help my data?" advisory, a repeated
  CV + paired-significance harness, and a speed shootout vs XGBoost / LightGBM /
  treelite-TL2cgen / Timber.

---

## Licensing picture

- **v1, v2, v3** — original code; the **methods and algorithms are unencumbered**. v3's
  only XGBoost touch is as an optional *baseline*, so the v3 method is genuinely
  XGBoost-free.
- **Northside (v4)** — released **AGPL-3.0**, *because it bundles XGBoost* (and uses
  LightGBM / treelite for its hybrid + benchmarks). The notebook already documents a
  dual-licensing path (AGPL for open use; a separate commercial license available).
- **XGForge** — depends on XGBoost; it is the compiler that produces Northside's `RLNF`
  fast-XGBoost mode.

**Implication for publishing:** the clean-IP showcase path is **v3** (and the shared GP +
AOT core beneath it). Northside is the strongest accuracy story but carries XGBoost's
copyleft footprint — keep that boundary explicit. (Structural folder/license separation
deferred to the rebuild; documented here for now.)

---

## Structural finding for the rebuild

The GP engine is **duplicated** across notebooks: v1 and v3 call it
`GPNode` / `GPIndividual`; Northside reimplements the same substrate as `Node` / `Ind`.
That copy-paste divergence is the core of the "chaos." **Step one of the rebuild is a
single shared GP + AOT core** that both v3 and Northside import, instead of three
hand-diverged copies.

## Suggested next steps

1. Extract a single shared core: `GP substrate + head-aware fitness + LR head + AOT (C) compiler`.
2. Rebuild **v3 (head-aware)** on top of it as the clean-IP reference method.
3. Rebuild **Northside** as `core + XGBoost expert + competence gate`, keeping the
   AGPL/XGBoost boundary explicit.
4. Land the v3 **production-AOT port at ship budget (N≥25)** — the open #1 lead.
5. Build the public-facing demonstration notebooks on the rebuilt core.

---

## Appendix — the other notebooks, briefly

- **XGForge** is a clean, narrowly-specialized **tree→C compiler** (binary + dense +
  fp32). It is explicitly **not novel** (compile-trees-to-C dates to Treelite, 2017);
  the value is that it beats the general-purpose tools (TL2cgen, lleaves) *in its niche*.
  Keep it as deployment **infrastructure**, not as a "method."
- **symbolic_regression_v9** is the honest close of a separate GP research arc (v3→v9):
  the "v5 mechanism" (lateral repositioning within neutral fibers) speeds up evolution on
  small algebraic toys with continuous fitness, but **fails to generalize** to real
  domains (the universal "single-field rigidity" finding). Reference value: the rigidity
  diagnostic, and a model of honest scope-bounding. Not a production method.
