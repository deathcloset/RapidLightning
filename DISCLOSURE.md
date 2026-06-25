# Defensive Publication & Prior-Art Notice

**Author / inventor:** Ben Johnston  ·  benjohnston@gmail.com
**First public disclosure:** 2026-06-25
**Project:** Rapid Lightning — a family of nanosecond-class tabular classifiers
**Canonical public location:** https://github.com/deathcloset/RapidLightningPublishing
**License:** Apache License 2.0 (see [`LICENSE`](LICENSE))

## Purpose

This is a deliberate **defensive publication**. The methods, algorithms, and
artifacts in this repository are published openly to place them in the public
record as **prior art**, freely usable by anyone under the Apache License 2.0.

The intent is twofold: (1) to keep these inventions free for everyone to use, and
(2) to establish a **dated, public, reproducible disclosure** so that no party can
later claim them as novel in a patent. The notebooks in this repository are the
full enabling disclosure — they contain the working method, the code, and
reproducible measurements.

## What is disclosed

1. **Head-aware genetic-programming ensembles (Rapid Lightning v3).** The ensemble
   members are evolved not as predictors but as *signal generators* for a linear
   head: each member is selected to add a feature-space coordinate that maximizes
   `corr(signal ⟂ pool, residual)²` (functional gradient boosting in feature
   space). Individually the members are near-chance; collectively they feed a
   logistic head that classifies. → `notebooks/methods/rapid_lightning_v3.ipynb`

2. **Northside (Rapid Lightning v4).** A sparsely-gated mixture of two experts —
   the head-aware GP model and a gradient-boosted-tree model — joined by a
   per-sample *competence gate*, with a soft-blend serving mode.
   → `notebooks/methods/northside.ipynb`

3. **Ahead-of-time compilation of GP + linear-head ensembles to dependency-free
   C**, producing ~15 KB shared libraries with single-row inference in tens of
   nanoseconds. → `notebooks/foundations/`, `notebooks/tooling/XGForge.ipynb`

4. **The supporting record** — the v1/v2 foundations, the XGForge tree→C compiler,
   and a genetic-programming research arc — is included for completeness and
   provenance.

## Provenance

Authored by Ben Johnston, from ~30 years of self-directed work in technology and
science. The Git commit history in this repository provides the dated, immutable
record of authorship and public availability.

---
*This is a prior-art / defensive-publication notice. It is not legal advice; for
matters with real IP or financial stakes, consult a patent attorney.*
