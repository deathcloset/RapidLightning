"""
Correctness checks for rapidlightning.gradgen — the hand-derived gradients are
verified against finite differences, the parametrized GP evaluator against the
engine's own, and the emitted C against the Python model.

Run:  PYTHONPATH=. python3 tests/test_gradgen.py      (or: pytest tests/)
"""
import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from rapidlightning import RapidLightningClassifier
from rapidlightning.engine import random_tree
from rapidlightning.gradgen import (
    HeadAwarePool, RidgeHeadClassifier, TunedGPClassifier, _ParamForest, _act, _dact,
)


def _data(n=400, d=6, seed=0):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=4, n_redundant=0, random_state=seed)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    return X, y.astype(np.int64)


def _fd_check(f, theta, n_probe=12, h=1e-6, seed=0):
    """Max relative error between the analytic gradient and central differences.
    The floor is relative to the gradient's scale, so components that are ~0
    (e.g. through a saturated sigmoid) don't fail on finite-difference round-off."""
    rng = np.random.default_rng(seed)
    _, g = f(theta)
    floor = 1e-4 * float(np.max(np.abs(g))) + 1e-12
    worst = 0.0
    for i in rng.choice(len(theta), size=min(n_probe, len(theta)), replace=False):
        e = np.zeros_like(theta)
        e[i] = h
        num = (f(theta + e)[0] - f(theta - e)[0]) / (2 * h)
        worst = max(worst, abs(num - g[i]) / max(floor, abs(num) + abs(g[i])))
    return worst


def test_headaware_objective_gradient():
    X, y = _data()
    X = X.astype(np.float64)
    rng = np.random.default_rng(1)
    H = np.tanh(X @ rng.normal(size=(6, 3)))
    d = X.shape[1]
    for objective, act in (("newton", "tanh"), ("newton", "softsign"), ("corr", "tanh")):
        pool = HeadAwarePool(H, y.astype(np.float64), objective=objective)
        def f(theta):
            z = X @ theta[:d] + theta[d]
            s = _act(z, act)
            J, gs = pool.score_grad(s)
            gz = gs * _dact(z, s, act)
            return J, np.concatenate([X.T @ gz, [gz.sum()]])
        theta = rng.normal(0, 0.5, d + 1)
        assert _fd_check(f, theta) < 1e-5, (objective, act)
        # score() and score_grad() agree
        s = _act(X @ theta[:d] + theta[d], act)
        assert abs(pool.score(s[:, None])[0] - pool.score_grad(s)[0]) < 1e-12


def test_joint_loss_gradient():
    X, y = _data()
    for direct in (False, True):
        clf = RidgeHeadClassifier(n_units=5, method="joint", act="tanh", direct=direct, joint_maxiter=1)
        clf.fit(X, y)
        captured = {}
        orig = clf._joint

        # re-enter _joint just to grab its objective via scipy's call
        import rapidlightning.gradgen as gg
        real_min = gg.minimize

        def spy(fun, x0, **kw):
            captured["f"], captured["x0"] = fun, x0
            return real_min(fun, x0, **kw)

        gg.minimize = spy
        try:
            orig(X.astype(np.float64), y.astype(np.float64), None, None, clf.W_, clf.b_, clf.v_, clf.c_)
        finally:
            gg.minimize = real_min
        assert _fd_check(captured["f"], captured["x0"] + 0.01) < 1e-5, direct


def test_param_forest_matches_engine_and_gradient():
    X, y = _data(n=300, d=5)
    random.seed(3)
    trees = [random_tree(6, X.shape[1]) for _ in range(6)]
    Xd = X.astype(np.float64)
    forest = _ParamForest(trees, Xd)
    S_hard, _ = forest.outputs(forest.theta0, Xd, hard=True)
    S_ref = np.column_stack([t.evaluate(X) for t in trees]).astype(np.float64)
    assert np.allclose(S_hard, S_ref, rtol=1e-4, atol=1e-3)

    rng = np.random.default_rng(0)
    G = rng.normal(size=S_hard.shape)

    def f(theta):
        S, V = forest.outputs(theta, Xd)
        return float((S * G).sum()), forest.backward(theta, Xd, V, G)

    assert _fd_check(f, forest.theta0 + rng.normal(0, 0.01, len(forest.theta0)), n_probe=20) < 1e-4


def test_compiled_ridge_matches_python():
    X, y = _data()
    for act in ("tanh", "relu", "softsign"):
        for direct in (False, True):
            clf = RidgeHeadClassifier(n_units=6, method="headaware", act=act, n_candidates=2,
                                      unit_maxiter=30, direct=direct).fit(X, y)
            so = clf.compile()
            p_py = clf.predict_proba(X[:200])[:, 1]
            p_c = so.predict_proba(X[:200])
            assert np.max(np.abs(p_py - p_c)) < 1e-4, (act, direct)


def test_tuned_gp_compiles_and_improves_train_loss():
    X, y = _data(n=500)
    Xf, Xv, yf, yv = X[:350], X[350:], y[:350], y[350:]
    rl = RapidLightningClassifier(n_members=3, generations=5, population=60, seed=1).fit(Xf, yf)
    tg = TunedGPClassifier(maxiter=60).fit_from(rl, Xf, yf)
    # the relaxed start is ~the evolved model; tuning should not hurt train AUC
    auc0 = roc_auc_score(yf, rl.predict_proba(Xf)[:, 1])
    auc1 = roc_auc_score(yf, tg.predict_proba(Xf)[:, 1])
    assert auc1 >= auc0 - 0.01
    so = tg.compile()
    p_py = tg.predict_proba(Xv)[:, 1]
    p_c = so.predict_proba(Xv)
    assert np.max(np.abs(p_py - p_c)) < 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
