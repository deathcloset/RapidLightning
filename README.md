# ⚡ Rapid Lightning

A family of **nanosecond-class tabular classifiers**. Genetic-programming ensembles
that compile to a **~15 KB, dependency-free C `.so`** and predict in **tens of
nanoseconds** on a single CPU core — while staying competitive with (and sometimes
beating) gradient-boosted trees on accuracy.

> **Status: rebuilding.** The six original notebooks are sorted out and written up in
> **[`RECORD.md`](RECORD.md)**. The best method (v3 head-aware) is now extracted into a
> clean, installable package — **[`rapidlightning/`](rapidlightning/)** — with the first
> runnable demo in **[`demos/`](demos/)**. Next: more demos (robotics, aerospace) and the
> Northside hybrid on the shared core.

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

## The two best methods

| | Method | Why | Licensing |
|---|---|---|---|
| 🥇 *original* | **[Rapid Lightning v3 — head-aware](notebooks/methods/rapid_lightning_v3.ipynb)** | Trees evolved as *signal generators* for a linear head (not predictors). The strongest objective found; Pareto-dominates size-matched XGBoost on continuous data. | Apache 2.0 · method is XGBoost-free |
| 🥇 *accuracy* | **[Northside (v4)](notebooks/methods/northside.ipynb)** | Gated mixture of the head-aware GP + XGBoost. The only method here that *beats* XGBoost (conditionally), with rigorous CV + significance testing. | Apache 2.0 · uses XGBoost (Apache 2.0) |

These two are a **stack, not rivals** — Northside contains v3's head-aware GP as one of
its experts. See [`RECORD.md`](RECORD.md) for the full reasoning, measured numbers, and
the licensing picture.

## Repository layout

```
rapidlightning/   the clean, installable core (head-aware engine + AOT-to-C compiler)
demos/
  01_sensors_core.ipynb        the domain-neutral showcase (RL vs XGBoost + the 15 KB / ~45 ns .so)
  02_droids_control_loop.ipynb robotics: RL balances a cart-pole — speed as a control capability
  03_spaceships_flight.ipynb   aerospace: tiny, deterministic, auditable inference for flight hardware
notebooks/
  methods/        the two "best" — show these off
    rapid_lightning_v3.ipynb     head-aware GP (clean IP)
    northside.ipynb              v4 gated GP+XGBoost hybrid
  foundations/    the base pipeline these grew from (superseded on method)
    rapid_lightning_v1.ipynb     GP+LR ensemble → AOT .so (torch+numba)
    rapid_lightning_v2.ipynb     adds predict_proba output + capacity findings
  tooling/
    XGForge.ipynb                XGBoost→C compiler (reusable deployment infra)
  research/
    symbolic_regression_v9.ipynb GP-evolution-speedup study (honest negative result)
RECORD.md         the canonical account of what each notebook is + the "best" verdict
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
