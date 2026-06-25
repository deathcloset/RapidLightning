"""
Rapid Lightning — honest, rigorous benchmark.

Compares, on real tabular datasets with repeated stratified CV + paired
significance testing:
  * Rapid Lightning (head-aware GP)            -> compiled .so
  * XGBoost                                     -> XGForge .so AND treelite/tl2cgen .so
  * RLBoost (soft-MoE of the two experts)       -> the headline model

The claim under test: RLBoost matches-or-beats XGBoost on AUC (significantly,
where there's headroom) at nanosecond-class single-row latency and a tiny
footprint. Accuracy is the headline; latency/size are reported per artifact on
its honest basis (RL/XGForge expose a single-row C entry point; treelite is
batch-amortized).

Knobs via env: BENCH_FOLDS, BENCH_REPEATS, BENCH_MEMBERS, BENCH_GENS, BENCH_DATASETS.
"""
import os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import xgboost as xgb
from rapidlightning import RapidLightningClassifier
from rapidlightning.xgforge import xgforge

FOLDS   = int(os.environ.get("BENCH_FOLDS", 5))
REPEATS = int(os.environ.get("BENCH_REPEATS", 2))
MEMBERS = int(os.environ.get("BENCH_MEMBERS", 8))
GENS    = int(os.environ.get("BENCH_GENS", 15))
POP     = int(os.environ.get("BENCH_POP", 120))
MAXN    = 6000   # subsample cap for speed

# (name, openml_id or 'sklearn:bc') — binary tabular classification
DATASETS = {
    "breast_cancer": "sklearn:bc",
    "diabetes": 37, "phoneme": 1489, "banknote": 1462,
    "credit-g": 31, "spambase": 44, "kc2": 1063, "ilpd": 1480,
}
if os.environ.get("BENCH_DATASETS"):
    keep = set(os.environ["BENCH_DATASETS"].split(","))
    DATASETS = {k: v for k, v in DATASETS.items() if k in keep}


def load(name, src):
    try:
        if src == "sklearn:bc":
            d = load_breast_cancer(); Xdf, y = d.data, d.target
            import pandas as pd
            Xdf = pd.DataFrame(Xdf, columns=d.feature_names)
        else:
            d = fetch_openml(data_id=src, as_frame=True, parser="auto")
            Xdf = d.data.copy(); ys = d.target
            # binary target -> 0/1 (minority = 1)
            cats = ys.astype("category").cat.codes.values
            y = (cats == np.bincount(cats).argmin()).astype(int) if len(np.unique(cats)) == 2 else (cats > 0).astype(int)
        import pandas as pd
        Xdf = Xdf.replace("?", np.nan)
        catc = Xdf.select_dtypes(include=["object", "category"]).columns.tolist()
        numc = Xdf.select_dtypes(include=["number"]).columns.tolist()
        for c in catc: Xdf[c] = Xdf[c].astype(str).fillna("NA")
        for c in numc: Xdf[c] = Xdf[c].fillna(Xdf[c].median())
        pre = ColumnTransformer([("n", StandardScaler(), numc),
                                 ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), catc)])
        X = pre.fit_transform(Xdf)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.int64)
        if len(X) > MAXN:
            rng = np.random.default_rng(0); idx = rng.choice(len(X), MAXN, replace=False); X, y = X[idx], y[idx]
        if len(np.unique(y)) != 2 or min(np.bincount(y)) < 30:
            return None
        return X, y
    except Exception as e:
        print(f"  [skip {name}: {type(e).__name__}: {str(e)[:60]}]", flush=True)
        return None


def cv_aucs(X, y):
    """RLBoost = a leakage-free learned stack of the two experts. Within each
    train fold, experts train on 80% and the meta-combiner is fit on the held-out
    20%; all three are scored on the untouched test fold. The XGB and RL baselines
    use the SAME 80% experts, so the comparison is apples-to-apples."""
    rl_a, xg_a, mo_a = [], [], []
    cvs = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=0)
    for tr, te in cvs.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xet, Xbv, yet, ybv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
        rl = RapidLightningClassifier(n_members=MEMBERS, generations=GENS, population=POP, seed=1).fit(Xet, yet)
        xf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=1, verbosity=0).fit(Xet, yet)
        p_rl, p_xg = rl.predict_proba(Xte)[:, 1], xf.predict_proba(Xte)[:, 1]
        # learned combiner, fit only on the blend-holdout expert predictions
        meta = LogisticRegression(max_iter=1000).fit(
            np.c_[rl.predict_proba(Xbv)[:, 1], xf.predict_proba(Xbv)[:, 1]], ybv)
        p_mo = meta.predict_proba(np.c_[p_rl, p_xg])[:, 1]
        rl_a.append(roc_auc_score(yte, p_rl)); xg_a.append(roc_auc_score(yte, p_xg)); mo_a.append(roc_auc_score(yte, p_mo))
    return np.array(rl_a), np.array(xg_a), np.array(mo_a)


def verdict(mo, xg):
    d = mo - xg
    try:
        p = wilcoxon(mo, xg).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = 1.0
    if p < 0.05 and d.mean() > 0: return f"WIN  (+{d.mean():.4f}, p={p:.3f})"
    if p < 0.05 and d.mean() < 0: return f"LOSS ({d.mean():.4f}, p={p:.3f})"
    return f"tie  ({d.mean():+.4f}, p={p:.3f})"


def latency_size(X, y):
    from rapidlightning.forge import compile_c  # noqa
    Xtr, Xte = X[:int(len(X) * .8)], X[int(len(X) * .8):]
    ytr = y[:int(len(X) * .8)]
    rl = RapidLightningClassifier(n_members=MEMBERS, generations=GENS, population=POP, seed=1).fit(Xtr, ytr)
    fast = rl.compile(); rl_ns = fast.benchmark_ns(Xte[0])
    xf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=1, verbosity=0).fit(Xtr, ytr)
    xfc = xgforge(xf); xf_ns = xfc.benchmark_ns(Xte[0])
    tl_kb, tl_ns = float("nan"), float("nan")
    try:
        import treelite, tl2cgen, tempfile
        m = treelite.frontend.from_xgboost(xf.get_booster())
        lp = os.path.join(tempfile.mkdtemp(), "tl.so")
        tl2cgen.export_lib(m, toolchain="gcc", libpath=lp, params={"parallel_comp": 4}, verbose=False)
        tl_kb = os.path.getsize(lp) / 1024
        pred = tl2cgen.Predictor(lp)
        B = np.ascontiguousarray(Xte, dtype=np.float32)
        for _ in range(3): pred.predict(tl2cgen.DMatrix(B))
        t0 = time.perf_counter_ns()
        for _ in range(20): pred.predict(tl2cgen.DMatrix(B))
        tl_ns = (time.perf_counter_ns() - t0) / 20 / len(B)   # batch-amortized
    except Exception as e:
        print(f"  [treelite skipped: {type(e).__name__}]", flush=True)
    return dict(rl_ns=rl_ns, rl_kb=fast.size_kb, xf_ns=xf_ns, xf_kb=xfc.size_kb, tl_ns=tl_ns, tl_kb=tl_kb,
                rlboost_ns=rl_ns + xf_ns, rlboost_kb=fast.size_kb + xfc.size_kb)


if __name__ == "__main__":
    print(f"config: {FOLDS}-fold x{REPEATS} | RL members={MEMBERS} gens={GENS} | datasets={list(DATASETS)}\n", flush=True)
    acc_rows, lat_rows = [], []
    for name, src in DATASETS.items():
        data = load(name, src)
        if data is None: continue
        X, y = data
        t0 = time.time()
        rl, xg, mo = cv_aucs(X, y)
        lat = latency_size(X, y)
        v = verdict(mo, xg)
        print(f"{name:<15} n={len(X):>5} pos={y.mean():.2f} | RL={rl.mean():.4f} XGB={xg.mean():.4f} RLBoost={mo.mean():.4f} | RLBoost vs XGB: {v}  [{time.time()-t0:.0f}s]", flush=True)
        acc_rows.append((name, rl.mean(), xg.mean(), mo.mean(), v))
        lat_rows.append((name, lat))

    print("\n=== LATENCY / SIZE (single-row C ns unless noted; treelite = batch-amortized) ===", flush=True)
    print(f"{'dataset':<15}{'RL ns':>8}{'XGForge ns':>11}{'treelite ns*':>13}{'RLBoost ns':>12}{'RLBoost KB':>12}", flush=True)
    for name, l in lat_rows:
        print(f"{name:<15}{l['rl_ns']:>8.0f}{l['xf_ns']:>11.0f}{l['tl_ns']:>13.0f}{l['rlboost_ns']:>12.0f}{l['rlboost_kb']:>12.1f}", flush=True)
    wins = sum(1 for *_, v in acc_rows if v.startswith("WIN"))
    ties = sum(1 for *_, v in acc_rows if v.startswith("tie"))
    loss = sum(1 for *_, v in acc_rows if v.startswith("LOSS"))
    print(f"\nRLBoost vs XGBoost across {len(acc_rows)} datasets:  {wins} WIN / {ties} tie / {loss} loss", flush=True)
    print("DONE", flush=True)
