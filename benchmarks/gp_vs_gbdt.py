"""
Deep, honest benchmark: GP ensembles vs tuned gradient-boosted trees on tabular
classification. Produces the numbers for the writeup notebook.

For each dataset (repeated stratified CV):
  * tuned GBDT stack  = stack(XGBoost + LightGBM + HistGB), all early-stopped/regularized
  * RL v1 (predictors, AdaBoost) and RL v3 (head-aware) — solo, and added to the stack
  * TabPFN (the 2025/2026 small-tabular SOTA) as a reference baseline, if installed
The headline question: does either GP variant add signal a tuned GBDT stack lacks?
Paired Wilcoxon significance on per-fold AUC. Writes one JSON record per dataset
to RESULTS (resumable). Capped at 6000 rows — GP training is O(N).

Run:  PYTHONPATH=. python3 benchmarks/gp_vs_gbdt.py
Knobs: USE_TABPFN=1 to add TabPFN; BENCH_DATASETS=name,name to subset.
"""
import os, json, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import xgboost as xgb, lightgbm as lgb
import sys; sys.path.insert(0, "/home/user/RapidLightningPublishing")
from rapidlightning import RapidLightningClassifier
from rapidlightning.engine import EvolutionConfig, evolve_member, fit_lr_head, head_proba
from benchmarks.benchmark import load

# name -> (openml_id, tier). S = small (<1500, RL's claimed niche), B = big.
DS = {
    "diabetes": (37, "S"), "credit-g": (31, "S"), "ilpd": (1480, "S"), "kc2": (1063, "S"),
    "banknote": (1462, "S"), "blood-transfusion": (1464, "S"), "ionosphere": (59, "S"),
    "sonar": (40, "S"), "qsar-biodeg": (1494, "S"), "wdbc": (1510, "S"),
    "phoneme": (1489, "B"), "spambase": (44, "B"), "eeg-eye-state": (1471, "B"),
    "magic": (1120, "B"), "electricity": (151, "B"), "adult": (1590, "B"),
}
if os.environ.get("BENCH_DATASETS"):
    keep = set(os.environ["BENCH_DATASETS"].split(",")); DS = {k: v for k, v in DS.items() if k in keep}
USE_TABPFN = os.environ.get("USE_TABPFN") == "1"
RESULTS = os.environ.get("BENCH_RESULTS",
    "/tmp/claude-0/-home-user-RapidLightningPublishing/15124e30-91ee-54bd-9547-38675601fc85/scratchpad/gp_vs_gbdt.jsonl")

def tuned_gbdts(Xf, yf, Xe, ye):
    xf = xgb.XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.05, reg_lambda=2.0,
                           subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                           early_stopping_rounds=30, random_state=1, verbosity=0)
    xf.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
    lg = lgb.LGBMClassifier(n_estimators=800, max_depth=4, num_leaves=15, learning_rate=0.05,
                            reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                            random_state=1, verbose=-1)
    lg.fit(Xf, yf, eval_set=[(Xe, ye)], eval_metric="auc", callbacks=[lgb.early_stopping(30, verbose=False)])
    hg = HistGradientBoostingClassifier(max_iter=800, max_depth=4, learning_rate=0.05, l2_regularization=2.0,
                                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                        random_state=1).fit(Xf, yf)
    return [xf, lg, hg]

def _acc_eval(y, w):
    def ev(pop, X, y2, pars, _w=None):
        for ind in pop:
            try:
                c = (ind.classify(X) == y)
                ind.train_acc = float((c * w).sum() / w.sum()) if w is not None else float(c.mean())
                ind.fitness = ind.train_acc - ind.total_size() * pars
            except Exception:
                ind.fitness = 0.0; ind.train_acc = 0.0
    return ev

def _maj(M, X, nc):
    P = np.stack([m.classify(X) for m in M], axis=0)
    return np.array([np.bincount(P[:, j], minlength=nc).argmax() for j in range(X.shape[0])])

def rlv1_fit(X, y, nm, gn, pp):
    cfg = EvolutionConfig(); cfg.num_members = nm; cfg.gens = gn; cfg.pop_size = pp; cfg.seed = 1
    nc = int(y.max()) + 1; M = []
    for m in range(1, nm + 1):
        w = None
        if m > 3 and len(M) >= 3:
            wr = (_maj(M, X, nc) != y); a = np.ones(len(X)); a[wr] = 3.0; w = a / a.mean()
        M.append(evolve_member(X.astype(np.float32), y, cfg, _acc_eval(y, w), member_num=m))
    sc, lr = fit_lr_head(M, X.astype(np.float32), y)
    return lambda Z: head_proba(M, sc, lr, Z.astype(np.float32))

def tabpfn_proba(Xf, yf, Xte):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier().fit(Xf, yf).predict_proba(Xte)[:, 1]

def stack(bv, ybv, te):
    return LogisticRegression(max_iter=2000).fit(np.column_stack(bv), ybv).predict_proba(np.column_stack(te))[:, 1]

def pv(a, b):
    d = a - b
    try: return float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except Exception: return 1.0

def run(X, y, tier):
    folds = int(os.environ.get("BENCH_FOLDS") or (5))
    reps = int(os.environ.get("BENCH_REPEATS") or (3 if tier == "S" else 2))
    nm = int(os.environ.get("BENCH_NM") or (12 if tier == "S" else 10))
    gn = int(os.environ.get("BENCH_GN") or (20 if tier == "S" else 18))
    pp = 130 if tier == "S" else 120
    use_tp = USE_TABPFN and len(X) <= 4000
    A = {k: [] for k in ["v1", "v3", "g", "gv1", "gv3", "tp", "gtp", "gtpv1"]}
    for tr, te in RepeatedStratifiedKFold(n_splits=folds, n_repeats=reps, random_state=0).split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xf, Xr, yf, yr = train_test_split(Xtr, ytr, test_size=0.30, random_state=0, stratify=ytr)
        Xe, Xbv, ye, ybv = train_test_split(Xr, yr, test_size=0.5, random_state=0, stratify=yr)
        G = tuned_gbdts(Xf, yf, Xe, ye)
        v3 = RapidLightningClassifier(n_members=nm, generations=gn, population=pp, seed=1).fit(Xf, yf)
        v1 = rlv1_fit(Xf, yf, nm, gn, pp)
        gt = [m.predict_proba(Xte)[:, 1] for m in G]; gb = [m.predict_proba(Xbv)[:, 1] for m in G]
        v3t, v3b = v3.predict_proba(Xte)[:, 1], v3.predict_proba(Xbv)[:, 1]
        v1t, v1b = v1(Xte), v1(Xbv)
        A["v1"].append(roc_auc_score(yte, v1t)); A["v3"].append(roc_auc_score(yte, v3t))
        A["g"].append(roc_auc_score(yte, stack(gb, ybv, gt)))
        A["gv1"].append(roc_auc_score(yte, stack(gb + [v1b], ybv, gt + [v1t])))
        A["gv3"].append(roc_auc_score(yte, stack(gb + [v3b], ybv, gt + [v3t])))
        if use_tp:
            tpt = tabpfn_proba(Xf, yf, Xte); tpb = tabpfn_proba(Xf, yf, Xbv)
            A["tp"].append(roc_auc_score(yte, tpt))
            A["gtp"].append(roc_auc_score(yte, stack(gb + [tpb], ybv, gt + [tpt])))
            A["gtpv1"].append(roc_auc_score(yte, stack(gb + [tpb, v1b], ybv, gt + [tpt, v1t])))
    return {k: np.array(v) for k, v in A.items() if v}

def summarize(name, n, tier, A):
    r = dict(name=name, n=n, tier=tier, folds=len(A["g"]),
             gbdt=float(A["g"].mean()), v1=float(A["v1"].mean()), v3=float(A["v3"].mean()),
             d_v1=float(A["gv1"].mean() - A["g"].mean()), p_v1=pv(A["gv1"], A["g"]),
             d_v3=float(A["gv3"].mean() - A["g"].mean()), p_v3=pv(A["gv3"], A["g"]))
    if "tp" in A:
        r.update(tabpfn=float(A["tp"].mean()),
                 d_gtpv1=float(A["gtpv1"].mean() - A["gtp"].mean()), p_gtpv1=pv(A["gtpv1"], A["gtp"]))
    return r

if __name__ == "__main__":
    done = {}
    if os.path.exists(RESULTS):
        for ln in open(RESULTS): r = json.loads(ln); done[r["name"]] = r
    print(f"deep benchmark | {len(DS)} datasets | TabPFN={'on' if USE_TABPFN else 'off'} | resuming {list(done)}\n", flush=True)
    for name, (sid, tier) in DS.items():
        if name in done: r = done[name]
        else:
            d = load(name, sid)
            if d is None: continue
            t0 = time.time(); A = run(d[0], d[1], tier); r = summarize(name, len(d[0]), tier, A)
            r["secs"] = round(time.time() - t0)
            open(RESULTS, "a").write(json.dumps(r) + "\n"); done[name] = r
        tp = f" TabPFN={r['tabpfn']:.4f}" if "tabpfn" in r else ""
        print(f"[{r['tier']}] {r['name']:<16} n={r['n']:>4} | GBDT={r['gbdt']:.4f} v1={r['v1']:.4f} v3={r['v3']:.4f}{tp} "
              f"| +v1 Δ{r['d_v1']:+.4f}(p={r['p_v1']:.3f}) +v3 Δ{r['d_v3']:+.4f}(p={r['p_v3']:.3f})", flush=True)
    print("\nDEEP_DONE", flush=True)
