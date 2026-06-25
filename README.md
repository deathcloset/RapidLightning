# ⚡ Rapid Lightning

A family of **nanosecond-class tabular classifiers**. Genetic-programming ensembles
that compile to a **~15 KB, dependency-free C `.so`** and predict in **tens of
nanoseconds** on a single CPU core — while staying competitive with (and sometimes
beating) gradient-boosted trees on accuracy.

> **Status: organizing.** This repo currently collects the six notebooks from a
> chaotic, multi-context development process. The first job — sorting out what's what
> and which methods are the "best" — is done and written up in **[`RECORD.md`](RECORD.md)**.
> Next we extract a shared core and rebuild the best version(s), then build public
> demonstration notebooks.

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
