"""
The artifact check for the gradient-trained generators: do they keep what made
Rapid Lightning interesting to deploy — a small dependency-free C .so, tens-of-
nanoseconds single-row inference, and latency that doesn't depend on the input?

Fits each model on fold 0 of the benchmark protocol, compiles it to C, and
measures (on an otherwise idle machine) .so size, ns/row across 16 different
input rows (mean, spread, CoV — the input-dependent jitter), and C↔Python
agreement. The tuned (early-stopped) XGBoost, compiled with XGForge, is the
reference compiled tree model.

Run:  PYTHONPATH=. python3 benchmarks/gradgen_latency.py   → results/gradgen_latency.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from benchmarks.benchmark import load
from benchmarks.gp_vs_gbdt import DS, tuned_gbdts
from benchmarks.gradgen_vs_gp import splits, budget, HA, JT, V3T
from rapidlightning import RapidLightningClassifier
from rapidlightning.gradgen import RidgeHeadClassifier, TunedGPClassifier
from rapidlightning.xgforge import xgforge

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["diabetes", "magic", "spambase"]   # 8 features · 10 features · 57 features


def per_input_ns(so, rows):
    return np.array([so.benchmark_ns(x=r, n_warmup=3000, n_iters=200_000) for r in rows])


def main():
    out = []
    for name in DATASETS:
        sid, tier = DS[name]
        X, y = load(name, sid)
        Xf, yf, Xe, ye, Xbv, ybv, Xte, yte = next(splits(X, y, tier))[1]
        nm, gn, pp, K = budget(tier)
        rows = np.ascontiguousarray(Xte[:16], dtype=np.float32)

        v3 = RapidLightningClassifier(n_members=nm, generations=gn, population=pp, seed=1).fit(Xf, yf)
        models = {
            "v3 (GP head-aware)": (v3, v3.compile(), dict(nodes=v3.n_nodes)),
        }
        v3t = TunedGPClassifier(**V3T).fit_from(v3, Xf, yf, Xe, ye)
        models["v3t (GP + tuned numbers)"] = (v3t, v3t.compile(), dict(nodes=v3.n_nodes))
        for label, kw in [("ha (head-aware, gradient)", dict(HA, method="headaware")),
                          ("jt (joint / small MLP)", dict(JT, method="joint"))]:
            m = RidgeHeadClassifier(n_units=K, seed=1, **kw).fit(Xf, yf, Xe, ye)
            models[label] = (m, m.compile(), dict(generators=m.n_generators, params=m.n_params))

        # the reference compiled tree model: the benchmark's tuned XGBoost, refit at its
        # early-stopped size so XGForge compiles exactly the trees predict_proba uses
        xf = tuned_gbdts(Xf, yf, Xe, ye)[0]
        n_trees = int(xf.best_iteration) + 1
        xr = xgb.XGBClassifier(n_estimators=n_trees, max_depth=4, learning_rate=0.05, reg_lambda=2.0,
                               subsample=0.8, colsample_bytree=0.8, random_state=1, verbosity=0).fit(Xf, yf)
        models["XGBoost tuned (XGForge)"] = (xr, xgforge(xr), dict(trees=n_trees))

        for label, (m, so, info) in models.items():
            p_py = m.predict_proba(Xte)[:, 1]
            p_c = so.predict_proba(Xte)
            lat = per_input_ns(so, rows)
            r = dict(dataset=name, d=X.shape[1], model=label, kb=round(so.size_kb, 1),
                     ns_mean=round(float(lat.mean()), 1), ns_spread=round(float(lat.max() - lat.min()), 1),
                     cov_pct=round(float(100 * lat.std() / lat.mean()), 2),
                     auc_py=round(float(roc_auc_score(yte, p_py)), 4), auc_c=round(float(roc_auc_score(yte, p_c)), 4),
                     max_abs_dp=float(np.max(np.abs(p_py - p_c))), **info)
            out.append(r)
            print(f"{name:<9} {label:<27} {r['kb']:>6.1f} KB {r['ns_mean']:>8.1f} ns  spread {r['ns_spread']:>6.1f}  "
                  f"CoV {r['cov_pct']:>5.2f}%  AUC py {r['auc_py']:.4f} / C {r['auc_c']:.4f}", flush=True)
    path = os.path.join(HERE, "results", "gradgen_latency.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
