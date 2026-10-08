"""
Follow-up benchmark: does replacing GP evolution with gradient (or other
numerical) training of the signal generators get better results?

Same protocol as gp_vs_gbdt.py — the same 16 OpenML datasets, the same repeated
stratified folds, the same fit / early-stop / blend split of each training fold,
the same tuned GBDT stack and leakage-free stacker. Per fold it scores, solo and
added to the tuned GBDT stack:

  g      tuned GBDT stack (XGBoost + LightGBM + HistGB)        — the yardstick
  lr     plain logistic regression on the raw features        — the linear control
  v1     GP, members as predictors (AdaBoost)                 — re-run, for pairing
  v3     GP, head-aware signal generators                     — re-run, for pairing
  v3t    v3's evolved structure, every number gradient-tuned  — "tune the GP"
  hac    head-aware, generators trained by L-BFGS on v3's     — "replace GP by gradients",
         exact objective corr(g ⟂ pool, R)²                      literally
  ha     the same, on the head's exact Newton gain            — ... with the objective fixed
         (u·R)²/(|u|²+λ) — corr² is scale-invariant, which      for continuous optimization
         gradient ascent exploits (see gradgen.HeadAwarePool)
  jt     generators + head trained jointly (a small MLP)      — "drop head-aware too"
  haj    head-aware growth, then joint fine-tune              — greedy init + corrective step
  rnd    head-aware selection from random candidates          — no gradients, no evolution

Every gradient method gets exactly the GBDTs' data: fit on the 70% split,
early-stop / model-select on the 15% early-stopping split. Defaults were frozen
from benchmarks/gradgen_dev.py (disjoint datasets) before this ran.

Run:   PYTHONPATH=. python3 benchmarks/gradgen_vs_gp.py
Knobs: BENCH_DATASETS=name,name  BENCH_WORKERS=4  GG_RESULTS=dir
Writes results/gradgen_vs_gp_folds.jsonl (per fold, resumable) and
results/gradgen_vs_gp.jsonl (per dataset, with per-fold arrays + summary).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one core per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, time, warnings, numpy as np
import multiprocessing as mp
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from benchmarks.benchmark import load
from benchmarks.gp_vs_gbdt import DS, tuned_gbdts, stack, rlv1_fit
from rapidlightning import RapidLightningClassifier
from rapidlightning.gradgen import RidgeHeadClassifier, TunedGPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("GG_RESULTS", os.path.join(HERE, "results"))
FOLDS_PATH = os.path.join(OUT, "gradgen_vs_gp_folds.jsonl")
SUMMARY_PATH = os.path.join(OUT, "gradgen_vs_gp.jsonl")
METHODS = ["lr", "v1", "v3", "v3t", "hac", "ha", "jt", "haj", "rnd"]

# ---- frozen defaults: best mean within-family rank on the 14 dev datasets
# (`python3 benchmarks/gradgen_dev.py report`, results/gradgen_dev.jsonl), fixed before this
# benchmark ran. No direct input->head links: the head sees only the generators, exactly as
# in v3, so only "how the generators are made" differs between v3 and the gradient methods.
RIDGE = dict(act="tanh", direct=False, objective="newton")
HAC = dict(RIDGE, objective="corr", alpha_unit=10.0)       # dev: ha_tanh_a10_d0 (v3's exact objective)
HA = dict(RIDGE, alpha_unit=30.0)                          # dev: ha4_a30
JT = dict(RIDGE, alpha_joint=3.0, patience=40)             # dev: jt3_a3_p40
HAJ = dict(RIDGE, alpha_unit=10.0, alpha_joint=3.0, patience=40)   # dev: haj4_u10
RND = dict(RIDGE)                                          # dev: rnd4
V3T = dict(tau_frac=0.5, patience=200, maxiter=600)        # dev: v3t3_tau0.5_p200


def budget(tier):
    """GP budget exactly as gp_vs_gbdt.py; generators matched to v3's head width
    (each v3 member is 2 trees -> 2 head features)."""
    nm, gn, pp = (12, 20, 130) if tier == "S" else (10, 18, 120)
    return nm, gn, pp, 2 * nm


def splits(X, y, tier):
    reps = 3 if tier == "S" else 2
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=0)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xf, Xr, yf, yr = train_test_split(Xtr, ytr, test_size=0.30, random_state=0, stratify=ytr)
        Xe, Xbv, ye, ybv = train_test_split(Xr, yr, test_size=0.5, random_state=0, stratify=yr)
        yield i, (Xf, yf, Xe, ye, Xbv, ybv, Xte, yte)


def run_fold(job):
    name, fold = job
    sid, tier = DS[name]
    X, y = load(name, sid)
    Xf, yf, Xe, ye, Xbv, ybv, Xte, yte = dict(splits(X, y, tier))[fold]
    nm, gn, pp, K = budget(tier)
    rec = dict(name=name, fold=fold, tier=tier, n=len(X), secs={}, k={})

    t0 = time.time()
    G = tuned_gbdts(Xf, yf, Xe, ye)
    gt = [m.predict_proba(Xte)[:, 1] for m in G]
    gb = [m.predict_proba(Xbv)[:, 1] for m in G]
    rec["secs"]["g"] = round(time.time() - t0, 2)
    rec["g"] = float(roc_auc_score(yte, stack(gb, ybv, gt)))

    def score(key, predict, t_fit):
        pt, pb = predict(Xte), predict(Xbv)
        rec[key] = float(roc_auc_score(yte, pt))                              # solo
        rec["g+" + key] = float(roc_auc_score(yte, stack(gb + [pb], ybv, gt + [pt])))  # added to stack
        rec["secs"][key] = round(t_fit, 2)

    def timed(fn):
        t = time.time(); m = fn(); return m, time.time() - t

    m, t = timed(lambda: LogisticRegression(max_iter=2000).fit(Xf, yf))
    score("lr", lambda Z: m.predict_proba(Z)[:, 1], t)

    m, t = timed(lambda: rlv1_fit(Xf, yf, nm, gn, pp))
    score("v1", m, t)

    v3, t = timed(lambda: RapidLightningClassifier(n_members=nm, generations=gn, population=pp, seed=1).fit(Xf, yf))
    score("v3", lambda Z: v3.predict_proba(Z)[:, 1], t)
    rec["k"]["v3"] = 2 * nm

    m, t = timed(lambda: TunedGPClassifier(**V3T).fit_from(v3, Xf, yf, Xe, ye))
    score("v3t", lambda Z: m.predict_proba(Z)[:, 1], t + rec["secs"]["v3"])
    rec["k"]["v3t"] = 2 * nm

    for key, kw in [("hac", dict(HAC, method="headaware")), ("ha", dict(HA, method="headaware")),
                    ("jt", dict(JT, method="joint")), ("haj", dict(HAJ, method="headaware+joint")),
                    ("rnd", dict(RND, method="random"))]:
        m, t = timed(lambda: RidgeHeadClassifier(n_units=K, seed=1, **kw).fit(Xf, yf, Xe, ye))
        score(key, lambda Z: m.predict_proba(Z)[:, 1], t)
        rec["k"][key] = m.n_generators
    return rec


def pv(a, b):
    d = np.asarray(a) - np.asarray(b)
    try:
        return float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        return 1.0


def summarize(rows):
    rows = sorted(rows, key=lambda r: r["fold"])
    A = {k: np.array([r[k] for r in rows]) for k in ["g"] + METHODS + ["g+" + m for m in METHODS]}
    s = dict(name=rows[0]["name"], n=rows[0]["n"], tier=rows[0]["tier"], folds=len(rows),
             folds_auc={k: v.tolist() for k, v in A.items()},
             secs={k: float(np.mean([r["secs"][k] for r in rows])) for k in rows[0]["secs"]},
             k={k: float(np.mean([r["k"][k] for r in rows])) for k in rows[0]["k"]})
    s["mean"] = {k: float(v.mean()) for k, v in A.items()}
    s["d_stack"] = {m: float((A["g+" + m] - A["g"]).mean()) for m in METHODS}
    s["p_stack"] = {m: pv(A["g+" + m], A["g"]) for m in METHODS}
    s["d_vs_v3"] = {m: float((A[m] - A["v3"]).mean()) for m in METHODS}
    s["p_vs_v3"] = {m: pv(A[m], A["v3"]) for m in METHODS}
    return s


if __name__ == "__main__":
    ds = DS
    if os.environ.get("BENCH_DATASETS"):
        keep = set(os.environ["BENCH_DATASETS"].split(","))
        ds = {k: v for k, v in DS.items() if k in keep}
    os.makedirs(OUT, exist_ok=True)
    done = {}
    if os.path.exists(FOLDS_PATH):
        for ln in open(FOLDS_PATH):
            r = json.loads(ln); done[(r["name"], r["fold"])] = r
    jobs = []
    for name, (sid, tier) in ds.items():
        nf = 5 * (3 if tier == "S" else 2)
        jobs += [(name, f) for f in range(nf) if (name, f) not in done]
    # big datasets first so the long jobs don't straggle at the end
    jobs.sort(key=lambda j: ds[j[0]][1] != "B")
    print(f"gradgen vs gp | {len(ds)} datasets | {len(jobs)} fold-jobs to run ({len(done)} cached)", flush=True)
    t0 = time.time()
    with mp.Pool(int(os.environ.get("BENCH_WORKERS", 4)), maxtasksperchild=8) as pool:
        for i, r in enumerate(pool.imap_unordered(run_fold, jobs), 1):
            with open(FOLDS_PATH, "a") as f:
                f.write(json.dumps(r) + "\n")
            done[(r["name"], r["fold"])] = r
            print(f"  [{i}/{len(jobs)} {time.time()-t0:.0f}s] {r['name']:<16} fold {r['fold']:>2} | g={r['g']:.4f} "
                  + " ".join(f"{m}={r[m]:.4f}" for m in METHODS), flush=True)
    by = {}
    for (name, _), r in done.items():
        if name in ds:
            by.setdefault(name, []).append(r)
    with open(SUMMARY_PATH, "w") as f:
        for name in ds:
            if name in by:
                f.write(json.dumps(summarize(by[name])) + "\n")
    print("GRADGEN_DONE", flush=True)
