"""
Choosing defaults for the gradient-trained signal generators — on DEV data only.

The 16 benchmark datasets are never touched here: every knob of
rapidlightning.gradgen is picked on 14 *other* OpenML binary datasets (9 in
stages 1-2; 5 high-dimensional ones added in stage 3), then frozen before
benchmarks/gradgen_vs_gp.py runs. Same split structure as the benchmark (fit 70% /
early-stop 15% / blend 15% of each training fold), 5-fold CV. Every config ever
tried is in CONFIGS and every result in results/gradgen_dev.jsonl.

Run:   PYTHONPATH=. python3 benchmarks/gradgen_dev.py          (then: ... gradgen_dev.py report)
Knobs: DEV_CONFIGS=glob,glob (subset of CONFIGS), DEV_RESULTS=path, DEV_WORKERS=4
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one core per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, time, pickle, fnmatch, warnings, numpy as np
import multiprocessing as mp
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from benchmarks.benchmark import load
from benchmarks.gp_vs_gbdt import tuned_gbdts, stack
from rapidlightning import RapidLightningClassifier
from rapidlightning.gradgen import RidgeHeadClassifier, TunedGPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = {  # name -> openml id; deliberately disjoint from the benchmark's 16
    "heart-statlog": 53, "climate": 1467, "australian": 40981, "tic-tac-toe": 50,
    "pc1": 1068, "kc1": 1067, "wilt": 40983, "churn": 40701, "bank-marketing": 1461,
    # added for stage 3: many features relative to rows (the regime where v3's
    # scale-invariant objective breaks under gradient ascent)
    "spectf": 1600, "kc3": 1065, "pc3": 1050, "ozone": 1487, "hill-valley": 1479,
}
RESULTS = os.environ.get("DEV_RESULTS", os.path.join(HERE, "results", "gradgen_dev.jsonl"))
CACHE = os.environ.get("DEV_CACHE", os.path.join(HERE, "results", ".v3cache"))
FOLDS = 5


def ridge(**kw):
    return lambda Xf, yf, Xe, ye, ctx: RidgeHeadClassifier(seed=1, **kw).fit(Xf, yf, Xe, ye)


def tuned(**kw):
    return lambda Xf, yf, Xe, ye, ctx: TunedGPClassifier(**kw).fit_from(ctx["v3"](), Xf, yf, Xe, ye)


CONFIGS = {
    "lr":   lambda Xf, yf, Xe, ye, ctx: LogisticRegression(max_iter=2000).fit(Xf, yf),
    "v3":   lambda Xf, yf, Xe, ye, ctx: ctx["v3"](),
    "v3t_tau.05": tuned(tau_frac=0.05),
    "v3t_tau.2":  tuned(tau_frac=0.2),
    "v3t_tau.5":  tuned(tau_frac=0.5),
}
# stage 1: activation x direct input->head links x regularization
for act in ("tanh", "relu"):
    for d in (0, 1):
        for a in (1, 10, 100):
            CONFIGS[f"ha_{act}_a{a}_d{d}"] = ridge(method="headaware", act=act, alpha_unit=a, direct=bool(d),
                                                   objective="corr")
        for a in (0.1, 1, 10):
            CONFIGS[f"jt_{act}_a{a}_d{d}"] = ridge(method="joint", act=act, alpha_joint=a, direct=bool(d))
        CONFIGS[f"rnd_{act}_d{d}"] = ridge(method="random", act=act, direct=bool(d), objective="corr")
# stage 2 (stage 1 picked tanh and no direct input->head links — the head sees only the
# generators, exactly as in v3 — around unit alpha ~10 / joint alpha ~1). Joint early stopping
# can quit early under L-BFGS's noisy first steps -> patience x weight decay; greedy+joint
# fine-tune; a wide MLP, to see whether width (not the method) is what binds.
for a in (0.3, 1, 3):
    for pat in (40, 200):
        CONFIGS[f"jt3_a{a}_p{pat}"] = ridge(method="joint", act="tanh", alpha_joint=a, patience=pat)
for a in (3, 30):
    CONFIGS[f"ha3_a{a}"] = ridge(method="headaware", act="tanh", alpha_unit=a, objective="corr")
for pat in (40, 200):
    CONFIGS[f"haj3_p{pat}"] = ridge(method="headaware+joint", act="tanh", alpha_unit=10, alpha_joint=1,
                                    patience=pat, objective="corr")
for a in (1, 3):
    CONFIGS[f"jtwide128_a{a}"] = ridge(method="joint", act="tanh", n_units=128, alpha_joint=a, patience=200)
for tau in (0.2, 0.5, 1.0):
    CONFIGS[f"v3t3_tau{tau}_p200"] = tuned(tau_frac=tau, patience=200, maxiter=600)
# stage 3: stages 1-2 ran the head-aware/random methods on v3's exact objective, corr(g ⟂ pool, R)²
# (objective="corr"). Under gradient ascent it is scale-invariant, so with many features per row it
# finds degenerate tiny-variance units (seen as constant predictions in a benchmark smoke run).
# "newton" scores a generator by its exact gain in the L2-regularized head instead; re-select it
# here, on the original 9 dev sets plus 5 high-dimensional ones.
for a in (1, 3, 10, 30):
    CONFIGS[f"ha4_a{a}"] = ridge(method="headaware", act="tanh", alpha_unit=a, objective="newton")
for a in (3, 10):
    CONFIGS[f"haj4_u{a}"] = ridge(method="headaware+joint", act="tanh", alpha_unit=a, alpha_joint=3,
                                  patience=40, objective="newton")
CONFIGS["rnd4"] = ridge(method="random", act="tanh", objective="newton")


def _splits(X, y):
    for i, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xf, Xr, yf, yr = train_test_split(Xtr, ytr, test_size=0.30, random_state=0, stratify=ytr)
        Xe, Xbv, ye, ybv = train_test_split(Xr, yr, test_size=0.5, random_state=0, stratify=yr)
        yield i, (Xf, yf, Xe, ye, Xbv, ybv, Xte, yte)


def _work(job):
    name, fold, todo = job
    X, y = load(name, DEV[name])
    Xf, yf, Xe, ye, Xbv, ybv, Xte, yte = dict(_splits(X, y))[fold]
    cache = os.path.join(CACHE, f"{name}_{fold}.pkl")
    memo = {}

    def v3():  # the GP model is the expensive part — fit once per fold, ever
        if "v3" not in memo:
            if os.path.exists(cache):
                memo["v3"] = pickle.load(open(cache, "rb"))
            else:
                memo["v3"] = RapidLightningClassifier(n_members=12, generations=20, population=130,
                                                      seed=1).fit(Xf, yf)
                os.makedirs(CACHE, exist_ok=True)
                pickle.dump(memo["v3"], open(cache, "wb"))
        return memo["v3"]

    out = []
    for cfg in todo:
        t0 = time.time()
        if cfg == "gbdt":
            G = tuned_gbdts(Xf, yf, Xe, ye)
            p = stack([m.predict_proba(Xbv)[:, 1] for m in G], ybv, [m.predict_proba(Xte)[:, 1] for m in G])
            k = None
        else:
            m = CONFIGS[cfg](Xf, yf, Xe, ye, dict(v3=v3))
            p = m.predict_proba(Xte)[:, 1]
            k = getattr(m, "n_generators", None)
        out.append(dict(name=name, fold=fold, cfg=cfg, auc=float(roc_auc_score(yte, p)),
                        secs=round(time.time() - t0, 2), k=k))
    return out


# the final selection: within each family, best mean rank (then mean AUC) over every dev dataset
FAMILIES = {
    "jt":  "jt3_*",                   # joint (small MLP)
    "ha":  "ha4_*",                   # head-aware, gradient-trained, Newton objective
    "haj": "haj4_*",                  # head-aware growth -> joint fine-tune
    "rnd": "rnd4",                    # head-aware selection from random candidates
    "v3t": "v3t3_*",                  # v3 + gradient-tuned numbers
}
REFERENCE = ["gbdt", "lr", "v3", "ha_tanh_a10_d0", "rnd_tanh_d0"]


def report():
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in open(RESULTS)])
    m = df.groupby(["name", "cfg"]).auc.mean().unstack()
    n = df.groupby(["name", "cfg"]).auc.count().unstack()
    m = m.loc[list(DEV)]
    print(f"dev selection over {len(m)} datasets: {', '.join(m.index)}\n")
    picks = {}
    for fam, pat in FAMILIES.items():
        cands = [c for c in m.columns if fnmatch.fnmatch(c, pat) and (n.loc[list(DEV), c] == FOLDS).all()]
        sub = m[cands]
        tab = pd.DataFrame({"meanAUC": sub.mean(), "meanRank": sub.rank(axis=1, ascending=False).mean()})
        tab = tab.sort_values(["meanRank", "meanAUC"], ascending=[True, False])
        picks[fam] = tab.index[0]
        print(f"[{fam}]\n{tab.round(4).to_string()}\n  -> {tab.index[0]}\n")
    final = REFERENCE + list(picks.values())
    show = m[final].copy()
    print(show.round(4).to_string())
    print("\nmean AUC:\n" + show.mean().round(4).to_string())
    return picks


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["report"]:
        report(); raise SystemExit
    pats = os.environ.get("DEV_CONFIGS", "*").split(",")
    cfgs = ["gbdt"] + [c for c in CONFIGS if any(fnmatch.fnmatch(c, p) for p in pats)]
    done = set()
    if os.path.exists(RESULTS):
        for ln in open(RESULTS):
            r = json.loads(ln); done.add((r["name"], r["fold"], r["cfg"]))
    jobs = []
    for name in DEV:
        for fold in range(FOLDS):
            todo = [c for c in cfgs if (name, fold, c) not in done]
            if todo:
                jobs.append((name, fold, todo))
    print(f"dev | {len(DEV)} datasets x {FOLDS} folds | {len(cfgs)} configs | {len(jobs)} fold-jobs to run", flush=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with mp.Pool(int(os.environ.get("DEV_WORKERS", 4)), maxtasksperchild=4) as pool:
        for rows in pool.imap_unordered(_work, jobs):
            with open(RESULTS, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"  {rows[0]['name']:<15} fold {rows[0]['fold']}  ({len(rows)} configs)", flush=True)
    print("DEV_DONE", flush=True)
