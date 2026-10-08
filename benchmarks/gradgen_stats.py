"""
Statistics for the gradient-generator benchmark (reads results/gradgen_vs_gp.jsonl).

Per-dataset fold-level Wilcoxon p-values (what gp_vs_gbdt.py reports) are
optimistic — CV folds share training data, so they aren't independent. The
headline tests here are the ones recommended for comparing classifiers across
many datasets (Demšar, JMLR 2006): a Wilcoxon signed-rank test over the
per-dataset mean differences, and Friedman average ranks with the Nemenyi
critical difference.
"""
import json
import os

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "results", "gradgen_vs_gp.jsonl")
LABELS = {
    "g": "tuned GBDT stack", "lr": "logistic regression", "v1": "GP v1 (predictors)",
    "v3": "GP v3 (head-aware)", "v3t": "GP v3 + gradient-tuned numbers",
    "hac": "head-aware, gradient-trained (v3's corr² objective)",
    "ha": "head-aware, gradient-trained (Newton-gain objective)", "jt": "joint (small MLP)",
    "haj": "head-aware → joint fine-tune", "rnd": "head-aware, random search",
}
# Nemenyi q_0.05 (two-tailed) for k classifiers, Demšar (2006) Table 5
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load(path=SUMMARY):
    return [json.loads(line) for line in open(path)]


def matrix(rows, keys):
    """datasets x keys matrix of per-dataset mean AUC."""
    return np.array([[r["mean"][k] for k in keys] for r in rows])


def xds_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank over per-dataset means (Demšar 2006)."""
    d = np.asarray(a) - np.asarray(b)
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def wins(rows, a, b, which="solo"):
    """Count datasets where a beats / loses to b at fold-level p<0.05 (secondary)."""
    w = l = 0
    for r in rows:
        fa = np.array(r["folds_auc"][a]); fb = np.array(r["folds_auc"][b])
        d = fa - fb
        if not np.any(d != 0):
            continue
        p = wilcoxon(fa, fb).pvalue
        if p < 0.05:
            w += d.mean() > 0
            l += d.mean() < 0
    return int(w), int(l)


def friedman_ranks(rows, keys):
    M = matrix(rows, keys)
    # rank 1 = best (highest AUC), average ranks for ties
    ranks = np.array([np.argsort(np.argsort(-m, kind="stable"), kind="stable") + 1.0 for m in M])
    for i, m in enumerate(M):                       # proper tie handling
        for v in np.unique(m):
            idx = np.where(m == v)[0]
            if len(idx) > 1:
                ranks[i, idx] = ranks[i, idx].mean()
    k, n = len(keys), len(rows)
    stat, p = friedmanchisquare(*M.T)
    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    return dict(zip(keys, ranks.mean(axis=0))), float(p), float(cd)
