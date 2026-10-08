# ⚡ Rapid Lightning

**An honest investigation of genetic-programming (GP) ensembles for tabular classification** —
a novel method, a from-scratch compile-to-C toolchain, and a rigorous benchmark.

Told straight: these GP ensembles compile to a **~15 KB, dependency-free C `.so`** with
tens-of-nanoseconds inference — clean, fun engineering — **but under rigorous evaluation
they do not beat properly-tuned gradient-boosted trees** on tabular classification. A small
complementary signal survives on a few small datasets; it's too marginal to matter, and that
regime is now owned by tabular foundation models (TabPFN). The full, cited writeup is the centerpiece:

> ### 📓 [The investigation notebook →](notebooks/methods/rapid_lightning_investigation.ipynb)
> 16 OpenML datasets · tuned baselines · repeated CV + paired significance — every number
> reproducible via [`benchmarks/gp_vs_gbdt.py`](benchmarks/gp_vs_gbdt.py). A clean, scoped negative result.

**Why publish a negative?** Because the rigor *is* the contribution: a hopeful idea, built
end-to-end, stress-tested against the right baselines, and reported honestly when it didn't
pan out — including catching that an early "win" was just the baseline overfitting.

> **Status: complete**, plus one follow-up (Oct 2026): [what if the generators are *trained* by gradient
> instead of evolved?](notebooks/methods/gradient_signal_generators.ipynb) A concluded investigation,
> preserved as a reproducible record — not a work in progress.

## Quickstart

```python
from rapidlightning import RapidLightningClassifier

clf = RapidLightningClassifier(n_members=10, generations=20).fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]      # head-aware GP + logistic head

fast = clf.compile()                          # -> a ~15 KB, dependency-free C .so
fast.predict_proba_one(x)                     # single-row inference in tens of nanoseconds
fast.benchmark_ns()                           # measured, timed inside C
```

Install (editable, from a clone): `pip install -e ".[demos]"` · needs a system `gcc` for the compile step.
See **[`demos/01_sensors_core.ipynb`](demos/01_sensors_core.ipynb)** for the full showcase — train, benchmark vs XGBoost, compile, and *measure* the latency, all with baked-in outputs.

## The finding

Across 16 OpenML datasets, repeated CV, and paired significance testing **against tuned
(early-stopped, regularized) GBDT baselines**:

- **Neither GP objective beats a tuned GBDT stack.** *v1* (members as predictors) adds a
  small, significant complementary signal on a few *small* datasets (mean **+0.002 AUC**,
  up to +0.015 on ionosphere); *v3* ("head-aware", once thought best) adds **essentially
  nothing**.
- **The compile-to-C toolchain isn't novel** — [treelite](https://treelite.readthedocs.io/),
  [m2cgen](https://github.com/BayesWitnesses/m2cgen), [lleaves](https://siboehm.com/articles/21/lleaves),
  and [Hummingbird](https://github.com/microsoft/hummingbird) have done it for years; a tuned
  GBDT compiles to fast C just as well.
- **An early "win" was a mirage:** v1 beating GBDTs by +0.03 on small data was the *default*
  GBDTs overfitting — regularize them and it collapses to non-significance.

Full reasoning, numbers, and citations in the **[investigation notebook](notebooks/methods/rapid_lightning_investigation.ipynb)**.
The earlier [`RECORD.md`](RECORD.md) (which crowned "best methods" *before* this benchmark) is
kept for provenance but is **superseded** by the investigation.

## Follow-up: gradient-trained signal generators (Oct 2026)

v3 *evolves* its signal generators with GP. Gradients beat evolution almost everywhere else, so
what if the generators were *trained* instead? Same 16 datasets, folds and tuned baselines;
defaults chosen on 14 *other* datasets — **[the follow-up notebook →](notebooks/methods/gradient_signal_generators.ipynb)**

- **Has it been tried?** Yes, for ~35 years: training ridge units one at a time against the
  head's residual is Cascade-Correlation (Fahlman & Lebiere, 1990); trained jointly with the
  head, "signal generators + linear head" is simply a one-hidden-layer MLP.
- **Better than GP: yes.** Gradient-tuning v3's own numbers helps on all 16 datasets (+0.016 AUC);
  a jointly trained small MLP (24/20 units, v3's head width) beats v3 by **+0.030 AUC** (13/16,
  p = 0.003) and trains ~100× faster.
- **Better than tuned GBDTs: no.** It is level with the tuned GBDT stack on the 10 small datasets
  (where plain logistic regression also beats that stack on 6) and still ~0.04 behind on the 6 large ones.
- **In the GBDT stack** it adds +0.006 AUC (p = 0.002), three times v1's +0.002, but plain logistic
  regression adds +0.005 by itself: the signal is mostly smooth structure trees approximate poorly.
- **The head-aware idea doesn't survive gradients:** joint training beats head-aware greedy growth,
  and v3's objective corr(g ⟂ pool, R)² turns out to be ill-posed under gradient ascent
  (scale-invariant), which is fixed by scoring generators with the head's exact Newton gain.

## Repository layout

```
rapidlightning/   the clean, installable core (GP engine + AOT-to-C compiler)
  gradgen.py         the follow-up: gradient-trained / numerically-built signal generators
benchmarks/
  gp_vs_gbdt.py      the deep, honest benchmark (tuned baselines · CV · paired significance)
  gradgen_vs_gp.py   the follow-up benchmark (same protocol), + gradgen_dev.py (default
                     selection on 14 other datasets), gradgen_stats.py, gradgen_latency.py
  results/           per-dataset (and per-fold) results the notebooks read
notebooks/
  methods/
    rapid_lightning_investigation.ipynb   ←  THE WRITEUP — start here
    gradient_signal_generators.ipynb      the follow-up: trained, not evolved, generators
    rapid_lightning_v3.ipynb     head-aware GP (the "signal-generator" objective)
    northside.ipynb              v4 gated GP + XGBoost hybrid
  foundations/
    rapid_lightning_v1.ipynb     the predictor-objective GP (the variant that fared best)
    rapid_lightning_v2.ipynb     adds predict_proba output + capacity findings
  tooling/
    XGForge.ipynb                tree→C compiler (a niche reimplementation of treelite/m2cgen)
  research/
    symbolic_regression_v9.ipynb GP-evolution-speedup study (another honest negative)
demos/            early illustrative notebooks — NOTE: their speed comparisons predate the
                  honest benchmark and used XGBoost's slow Python path; see the investigation
                  for the fair, compiled-vs-compiled story
tests/            gradient checks + C↔Python agreement for rapidlightning.gradgen
RECORD.md         early "best methods" account — superseded by the investigation
README.md         you are here
```

## What makes it fast

Each model — a bag of GP trees plus a logistic-regression head — is **AOT-compiled to
straight-line C99**: every tree expression and the head's dot product are inlined as
immediate constants, so a prediction is one function call with no opcode dispatch, no
runtime stack, and no preprocessing step. The result is a single self-contained `.so`
(no scikit-learn / XGBoost dependency at serve time) that a customer's request sees in
tens to hundreds of nanoseconds.

## License

Released under the **Apache License 2.0** — a permissive, non-copyleft license chosen to
maximize adoption and for its explicit **patent grant** (defensive cover for a novel
method). Use it freely, including in commercial and closed-source products; just preserve
the copyright/license notices. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Every dependency is permissive too — XGBoost and treelite/TL2cgen (Apache 2.0), LightGBM
(MIT), scikit-learn/NumPy (BSD) — and none is redistributed here, so nothing imposes
additional constraints. (Earlier drafts licensed Northside under AGPL-3.0; that's
superseded.)

This repository is also a **defensive publication** — see [`DISCLOSURE.md`](DISCLOSURE.md)
for the prior-art notice that keeps these methods free and unpatentable-by-others.
