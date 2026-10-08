"""
Rapid Lightning — gradient-trained signal generators (the follow-up experiment).

The head-aware ensemble (v3) *evolves* its signal generators with genetic
programming (GP): each new member is selected to maximize
corr(member ⟂ pool, residual)², then a logistic head combines them. Gradients
beat evolution at continuous optimization almost everywhere else, so: what if
the generators are *trained* instead of evolved?

Everything else stays fixed — standardized tabular input → K scalar signal
generators → logistic head → sigmoid, compiled ahead-of-time to C. Only the way
the generators are made changes. `RidgeHeadClassifier(method=...)`:

  "headaware"        Greedy, one generator at a time, each trained by L-BFGS on
                     a head-aware objective (exact analytic gradient), best of
                     several starts, head refit after each — either v3's own
                     corr(g ⟂ pool, R)² (objective="corr") or the head's exact
                     Newton gain (objective="newton", the default; corr² is
                     scale-invariant, which gradient ascent exploits — see
                     HeadAwarePool). This is Cascade-Correlation (Fahlman &
                     Lebiere, 1990) without the cascade — a constructive network
                     in the sense of Kwok & Yeung (1997).
  "joint"            All generators + head trained together on the actual
                     log-loss — i.e. a one-hidden-layer MLP — early-stopped.
  "headaware+joint"  Greedy head-aware growth, then joint fine-tuning (the
                     "corrective step" of GrowNet, Badirli et al., 2020).
  "random"           No training at all: each generator is picked from random
                     candidates by the head-aware objective (Stochastic
                     Configuration Networks, Wang & Li, 2017). The control for
                     "is the search method even the bottleneck?".

And keeping GP for structure but gradients for the numbers:

  `TunedGPClassifier`  The evolved v3 ensemble with every constant, a weight on
                     every input leaf and a temperature on every `lt` (relaxed
                     to a sigmoid) tuned jointly with the head by L-BFGS on
                     log-loss (cf. Kommenda et al., 2020, for GP + gradient-based
                     constant fitting).

Generators are ridge units  g(x) = act(w·x + b)  — branchless, so the compiled
artifact keeps the GP's input-independent latency. Pure NumPy + SciPy (already a
scikit-learn dependency); no torch.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---- activations -------------------------------------------------------------

def _act(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "softsign":
        return z / (1.0 + np.abs(z))
    raise ValueError(f"unknown activation {kind!r}")


def _dact(z, s, kind):
    """d act / dz, given z and s = act(z)."""
    if kind == "tanh":
        return 1.0 - s * s
    if kind == "relu":
        return (z > 0.0).astype(np.float64)
    if kind == "softsign":
        return 1.0 / (1.0 + np.abs(z)) ** 2
    raise ValueError(f"unknown activation {kind!r}")


def _logloss(f, y):
    """Mean binary log-loss from logits f (numerically stable)."""
    return float(np.mean(np.logaddexp(0.0, f) - y * f))


# ---- the head-aware objective, with its exact gradient -------------------------

class HeadAwarePool:
    """The state the next generator is scored against — the same construction as
    `engine.pool_setup`: fit the logistic head on the current pool, take its
    residual R = y − p, and the ridge projector onto span([1, pool]).

    `score(S)` scores each column of S, and `score_grad(s)` returns the score with
    dJ/ds, which is what lets gradient ascent replace evolution. Two objectives:

      "corr"    v3's fitness exactly: corr(s ⟂ pool, R)², head on standardized
                generators. Fine for GP's discrete search — but it is
                scale-invariant, so under continuous optimization a unit can shrink
                into a tiny-variance (near-linear) direction at no cost and fit the
                residual by plain OLS: with many features and few rows it finds
                degenerate minor-component units, which the standardized head then
                amplifies.
      "newton"  the exact gain of adding s to the L2-regularized logistic head (a
                Newton step), (u·R)² / (|u|² + λ_h) with λ_h = 1 / (C · mean p(1−p)) —
                the G²/(H + λ) form of XGBoost's split gain. λ_h → 0 recovers corr²;
                a tiny-variance unit now scores ~0. Head on raw generator outputs
                (tanh units are already bounded), so its L2 penalty acts on them."""

    def __init__(self, H, y, C=1.0, eps=1e-3, objective="newton"):
        N = len(y)
        cols = [np.ones((N, 1))]
        if H is not None and H.shape[1]:
            Hh = StandardScaler().fit_transform(H) if objective == "corr" else H
            cols.append(Hh)
            lr = LogisticRegression(max_iter=2000, C=C, solver="lbfgs").fit(Hh, y.astype(np.int64))
            p = lr.predict_proba(Hh)[:, 1]
        else:
            p = np.full(N, y.mean())
        self.B = np.concatenate(cols, axis=1)
        M = np.linalg.inv(self.B.T @ self.B + eps * np.eye(self.B.shape[1]))
        self.M = 0.5 * (M + M.T)
        R = y - p
        self.Rc = R - R.mean()
        self.Rn2 = float(self.Rc @ self.Rc) + 1e-12
        self.lam_h = 0.0 if objective == "corr" else 1.0 / (C * float(np.mean(p * (1.0 - p))) + 1e-12)

    def perp(self, s):
        """Component of s (vector or columns) orthogonal to the pool's span."""
        return s - self.B @ (self.M @ (self.B.T @ s))

    def score(self, S):
        Sp = self.perp(S)
        Sp = Sp - Sp.mean(axis=0)
        a = Sp.T @ self.Rc
        q = (Sp * Sp).sum(axis=0) + self.lam_h + 1e-12
        return a * a / (q * self.Rn2)

    def score_grad(self, s):
        u = self.perp(s)
        u = u - u.mean()
        a = float(u @ self.Rc)
        q = float(u @ u) + self.lam_h + 1e-12
        J = a * a / (q * self.Rn2)
        g = (2.0 * a / (q * self.Rn2)) * (self.Rc - (a / q) * u)   # dJ/du (already centered)
        return J, self.perp(g)                                      # chain through ⟂ (symmetric)


def _train_unit(X, pool, w0, b0, act, lam, maxiter):
    """Gradient ascent (L-BFGS) on  J(w, b) − lam·|w|²  for one ridge generator."""
    d = X.shape[1]

    def f(theta):
        w, b = theta[:d], theta[d]
        z = X @ w + b
        s = _act(z, act)
        J, gs = pool.score_grad(s)
        gz = gs * _dact(z, s, act)
        obj = J - lam * float(w @ w)
        grad = np.empty(d + 1)
        grad[:d] = X.T @ gz - 2.0 * lam * w
        grad[d] = gz.sum()
        return -obj, -grad

    res = minimize(f, np.concatenate([w0, [b0]]), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter))
    return res.x[:d].copy(), float(res.x[d]), float(-res.fun)


def _random_candidates(X, n, rng):
    """A data-aware random dictionary of ridge units: sparse (1–3 features) or dense
    directions, log-uniform steepness, and a hyperplane through a random training
    row so no unit is constant on the data."""
    N, d = X.shape
    W = np.zeros((n, d))
    for i in range(n):
        k = int(rng.choice([1, 2, 3, d])) if d > 3 else d
        idx = rng.choice(d, size=min(k, d), replace=False)
        u = rng.normal(size=len(idx))
        W[i, idx] = u / (np.linalg.norm(u) + 1e-12)
    W *= np.exp(rng.uniform(np.log(0.3), np.log(10.0), size=(n, 1)))
    b = -np.einsum("ij,ij->i", W, X[rng.integers(0, N, size=n)])
    return W, b


# ---- the classifier ----------------------------------------------------------

class RidgeHeadClassifier:
    """K ridge signal generators  act(w·x + b)  + a logistic head.

    Same contract as the GP version: binary y in {0,1}, standardized float input.
    `fit(X, y, X_val, y_val)` — the optional validation split is used only for
    early stopping / choosing how many generators to keep, exactly the role the
    early-stopping split plays for the tuned GBDT baselines."""

    def __init__(self, n_units=24, method="headaware", act="tanh", objective="newton",
                 n_candidates=6, unit_maxiter=100, alpha_unit=1.0, alpha_joint=1.0,
                 joint_maxiter=600, patience=40, n_random=256, direct=False, seed=1):
        self.n_units = n_units
        self.method = method
        self.objective = objective
        self.act = act
        self.n_candidates = n_candidates
        self.unit_maxiter = unit_maxiter
        self.alpha_unit = alpha_unit
        self.alpha_joint = alpha_joint
        self.joint_maxiter = joint_maxiter
        self.patience = patience
        self.n_random = n_random
        self.direct = direct
        self.seed = seed

    # -- greedy growth (headaware / random) --
    def _grow(self, X, y, rng):
        N, d = X.shape
        lam = self.alpha_unit / N
        W, b = np.zeros((0, d)), np.zeros(0)
        H = np.zeros((N, 0))
        for _ in range(self.n_units):
            pool = HeadAwarePool(np.hstack([X, H]) if self.direct else H, y, objective=self.objective)
            if self.method == "random":
                Wc, bc = _random_candidates(X, self.n_random, rng)
                J = pool.score(_act(X @ Wc.T + bc, self.act))
                k = int(np.argmax(J))
                w, bb = Wc[k], bc[k]
            else:
                # start 0: a linear direction for the ⟂-residual; the rest: random,
                # pre-activation ~ N(0, 1). (Under "corr", v3's objective, the start is
                # near-unregularized OLS — itself the degenerate solution when features
                # ~ rows; "newton" starts from the residual-correlation direction X⟂ᵀR.)
                starts = []
                Xp = pool.perp(X)
                if self.objective == "corr":
                    w_ls = np.linalg.lstsq(Xp.T @ Xp + 1e-3 * np.eye(d), Xp.T @ pool.Rc, rcond=None)[0]
                else:
                    w_ls = Xp.T @ pool.Rc
                z = X @ w_ls
                starts.append((w_ls / (z.std() + 1e-12), -z.mean() / (z.std() + 1e-12)))
                for _ in range(self.n_candidates - 1):
                    starts.append((rng.normal(0.0, 1.0 / np.sqrt(d), d), rng.normal(0.0, 0.5)))
                best = None
                for w0, b0 in starts:
                    w1, b1, obj = _train_unit(X, pool, w0, b0, self.act, lam, self.unit_maxiter)
                    if best is None or obj > best[2]:
                        best = (w1, b1, obj)
                w, bb = best[0], best[1]
            W = np.vstack([W, w])
            b = np.append(b, bb)
            H = np.column_stack([H, _act(X @ w + bb, self.act)])
        return W, b

    def _head_features(self, X, W, b):
        H = _act(X @ W.T + b, self.act)
        return np.hstack([X, H]) if self.direct else H

    def _fit_head(self, X, y, W, b):
        Z = self._head_features(X, W, b)
        if self.objective != "corr":       # raw bounded outputs: the L2 acts on the generators
            lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(Z, y)
            return lr.coef_[0].copy(), float(lr.intercept_[0])
        sc = StandardScaler().fit(Z)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(sc.transform(Z), y)
        # fold the scaler into (v, c) so the model is just  c + v·features
        v = lr.coef_[0] / sc.scale_
        c = float(lr.intercept_[0] - v @ sc.mean_)
        return v, c

    def _select_k(self, X, y, Xv, yv, W, b):
        """Keep the prefix of generators whose refit head has the best validation log-loss."""
        best = None
        for k in range(1, len(b) + 1):
            v, c = self._fit_head(X, y, W[:k], b[:k])
            ll = _logloss(self._head_features(Xv, W[:k], b[:k]) @ v + c, yv)
            if best is None or ll < best[0] - 1e-9:
                best = (ll, k, v, c)
        return best[1], best[2], best[3]

    # -- joint training (all generators + head on log-loss) --
    def _pack(self, W, b, v, c):
        return np.concatenate([W.ravel(), b, v, [c]])

    def _unpack(self, theta, K, d):
        W = theta[:K * d].reshape(K, d)
        b = theta[K * d:K * d + K]
        nv = K + (d if self.direct else 0)
        v = theta[K * d + K:K * d + K + nv]
        return W, b, v, float(theta[-1])

    def _joint(self, X, y, Xv, yv, W, b, v, c):
        N, d = X.shape
        K = len(b)
        lam = self.alpha_joint / N

        def logits(theta, A):
            W_, b_, v_, c_ = self._unpack(theta, K, d)
            return self._head_features(A, W_, b_) @ v_ + c_

        def f(theta):
            W_, b_, v_, c_ = self._unpack(theta, K, d)
            Zp = X @ W_.T + b_
            Hh = _act(Zp, self.act)
            Fe = np.hstack([X, Hh]) if self.direct else Hh
            fl = Fe @ v_ + c_
            loss = _logloss(fl, y) + 0.5 * lam * (float((W_ * W_).sum()) + float(v_ @ v_))
            r = (expit(fl) - y) / N
            gv = Fe.T @ r + lam * v_
            vh = v_[d:] if self.direct else v_
            gZ = np.outer(r, vh) * _dact(Zp, Hh, self.act)
            gW = gZ.T @ X + lam * W_
            return loss, np.concatenate([gW.ravel(), gZ.sum(axis=0), gv, [r.sum()]])

        theta0 = self._pack(W, b, v, c)
        if Xv is None:
            res = minimize(f, theta0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=self.joint_maxiter))
            return self._unpack(res.x, K, d)

        state = dict(best=_logloss(logits(theta0, Xv), yv), theta=theta0.copy(), stall=0)

        def cb(intermediate_result):
            th = intermediate_result.x
            ll = _logloss(logits(th, Xv), yv)
            if ll < state["best"] - 1e-7:
                state.update(best=ll, theta=th.copy(), stall=0)
            else:
                state["stall"] += 1
                if state["stall"] >= self.patience:
                    raise StopIteration

        minimize(f, theta0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=self.joint_maxiter))
        return self._unpack(state["theta"], K, d)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.float64)
        Xv = None if X_val is None else np.asarray(X_val, dtype=np.float64)
        yv = None if y_val is None else np.asarray(y_val).astype(np.float64)
        rng = np.random.default_rng(self.seed)
        N, d = X.shape
        self.n_features_ = d

        if self.method in ("headaware", "random", "headaware+joint"):
            W, b = self._grow(X, y, rng)
            if Xv is not None:
                k, v, c = self._select_k(X, y, Xv, yv, W, b)
                W, b = W[:k], b[:k]
            else:
                v, c = self._fit_head(X, y, W, b)
            if self.method == "headaware+joint":
                W, b, v, c = self._joint(X, y, Xv, yv, W, b, v, c)
        elif self.method == "joint":
            K = self.n_units
            W = rng.normal(0.0, 1.0 / np.sqrt(d), (K, d))
            b = rng.normal(0.0, 0.5, K)
            v = np.concatenate([np.zeros(d) if self.direct else np.zeros(0),
                                rng.normal(0.0, 1.0 / np.sqrt(K), K)])
            p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            c = float(np.log(p / (1 - p)))
            W, b, v, c = self._joint(X, y, Xv, yv, W, b, v, c)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self.W_, self.b_, self.v_, self.c_ = W, b, v, c
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._head_features(X, self.W_, self.b_) @ self.v_ + self.c_

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(np.int64)

    @property
    def n_generators(self):
        return len(self.b_)

    @property
    def n_params(self):
        return self.W_.size + self.b_.size + self.v_.size + 1

    def compile(self):
        from .forge import CompiledSource, emit_ridge_c
        return CompiledSource(emit_ridge_c(self.W_, self.b_, self.v_, self.c_, self.act,
                                           self.direct), self.n_features_)


# ---- GP structure, gradient-tuned numbers --------------------------------------

class _ParamForest:
    """GP trees flattened to postorder with tunable parameters: every constant,
    a multiplicative weight on every input leaf (Kommenda-style), and a log-
    temperature on every `lt`, which is relaxed to sigmoid((b − a)/τ). Exact
    reverse-mode gradients through the tree ops."""

    def __init__(self, trees, X, tau_frac=0.05):
        self.nodes, self.roots, params = [], [], []
        for t in trees:
            self.roots.append(self._flatten(t, params, X.shape[1]))
        self.theta0 = np.array(params, dtype=np.float64)
        # temperatures: a small fraction of each comparison's spread on the data,
        # so the relaxed model starts out ~identical to the evolved hard one
        V = self.forward(self.theta0, X, hard=True)
        for kind, _, kids, p in self.nodes:
            if kind == "lt":
                spread = float(np.std(V[kids[1]][0] - V[kids[0]][0]))
                self.theta0[p] = np.log(max(tau_frac * spread, 1e-4))
        self.n_lt = sum(1 for n in self.nodes if n[0] == "lt")

    def _flatten(self, node, params, n_features):
        kids = [self._flatten(c, params, n_features) for c in node.children]
        if node.ntype == "input":
            kind, val, p = "x", node.value % n_features, len(params)
            params.append(1.0)
        elif node.ntype == "const":
            kind, val, p = "c", None, len(params)
            params.append(float(node.value))
        else:
            kind, val, p = node.value, None, -1
            if kind == "lt":
                p = len(params)
                params.append(0.0)
        self.nodes.append((kind, val, kids, p))
        return len(self.nodes) - 1

    def forward(self, theta, X, hard=False):
        """Per node: (value, aux, clipped) — aux is what backward needs for `lt`,
        clipped marks rows hit by the GP evaluator's ±1e6 guard (or None)."""
        N = X.shape[0]
        V = [None] * len(self.nodes)
        for i, (kind, val, kids, p) in enumerate(self.nodes):
            aux, clipped = None, None
            if kind == "x":
                r = theta[p] * X[:, val]
            elif kind == "c":
                r = np.full(N, theta[p])
            else:
                a = V[kids[0]][0]
                bb = V[kids[1]][0] if len(kids) > 1 else None
                if kind == "add":     r = a + bb
                elif kind == "sub":   r = a - bb
                elif kind == "mul":   r = a * bb
                elif kind == "max":   r = np.maximum(a, bb)
                elif kind == "min":   r = np.minimum(a, bb)
                elif kind == "neg":   r = -a
                elif kind == "abs":   r = np.abs(a)
                elif kind == "clamp": r = np.clip(a, 0.0, 1.0)
                elif kind == "lt":
                    if hard:
                        r = (a < bb).astype(np.float64)
                    else:
                        tau = np.exp(theta[p])
                        u = (bb - a) / tau
                        r = expit(u)
                        aux = (u, tau)
                else:
                    r = a
                big = np.abs(r) > 1e6
                if big.any():
                    r = np.clip(r, -1e6, 1e6)
                    clipped = big
            V[i] = (r, aux, clipped)
        return V

    def outputs(self, theta, X, hard=False):
        V = self.forward(theta, X, hard)
        return np.column_stack([V[r][0] for r in self.roots]), V

    def backward(self, theta, X, V, G_roots):
        G = [None] * len(self.nodes)
        for r, g in zip(self.roots, G_roots.T):
            G[r] = g.copy()
        gt = np.zeros_like(theta)

        def acc(k, g):
            G[k] = g if G[k] is None else G[k] + g

        for i in range(len(self.nodes) - 1, -1, -1):
            g = G[i]
            if g is None:
                continue
            kind, val, kids, p = self.nodes[i]
            r, aux, clipped = V[i]
            if clipped is not None:
                g = g * (~clipped)
            if kind == "x":
                gt[p] += g @ X[:, val]
            elif kind == "c":
                gt[p] += g.sum()
            else:
                a = V[kids[0]][0]
                bb = V[kids[1]][0] if len(kids) > 1 else None
                if kind == "add":
                    acc(kids[0], g); acc(kids[1], g)
                elif kind == "sub":
                    acc(kids[0], g); acc(kids[1], -g)
                elif kind == "mul":
                    acc(kids[0], g * bb); acc(kids[1], g * a)
                elif kind == "max":
                    m = a >= bb
                    acc(kids[0], g * m); acc(kids[1], g * ~m)
                elif kind == "min":
                    m = a <= bb
                    acc(kids[0], g * m); acc(kids[1], g * ~m)
                elif kind == "neg":
                    acc(kids[0], -g)
                elif kind == "abs":
                    acc(kids[0], g * np.sign(a))
                elif kind == "clamp":
                    acc(kids[0], g * ((a > 0.0) & (a < 1.0)))
                elif kind == "lt":
                    u, tau = aux
                    ds = r * (1.0 - r)
                    acc(kids[0], -g * ds / tau)
                    acc(kids[1], g * ds / tau)
                    gt[p] += float((g * ds * (-u)).sum())
                else:
                    acc(kids[0], g)
        return gt


class TunedGPClassifier:
    """A trained v3 (head-aware GP) model whose numbers are then fine-tuned by
    gradient descent: structure from evolution, parameters from L-BFGS on the
    head's log-loss, early-stopped on a validation split."""

    def __init__(self, maxiter=400, patience=40, tau_frac=0.05):
        self.maxiter = maxiter
        self.patience = patience
        self.tau_frac = tau_frac

    def fit_from(self, rl, X, y, X_val=None, y_val=None):
        """`rl` is a fitted RapidLightningClassifier (members_, scaler_, lr_)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.float64)
        trees = [t for m in rl.members_ for t in m.trees]
        self.forest_ = _ParamForest(trees, X, self.tau_frac)
        self.mu_ = rl.scaler_.mean_.astype(np.float64)
        self.sd_ = rl.scaler_.scale_.astype(np.float64)
        v0 = rl.lr_.coef_[0].astype(np.float64)
        c0 = float(rl.lr_.intercept_[0])
        nt, T = len(self.forest_.theta0), len(trees)
        N = X.shape[0]
        lam = 1.0 / N          # = the head's sklearn C=1 L2 penalty, per-sample form
        self.n_features_ = X.shape[1]

        def split(th):
            return th[:nt], th[nt:nt + T], float(th[-1])

        def logits(th, A):
            tt, v, c = split(th)
            S, _ = self.forest_.outputs(tt, A)
            return ((S - self.mu_) / self.sd_) @ v + c

        def f(th):
            tt, v, c = split(th)
            S, V = self.forest_.outputs(tt, X)
            Z = (S - self.mu_) / self.sd_
            fl = Z @ v + c
            loss = _logloss(fl, y) + 0.5 * lam * float(v @ v)
            r = (expit(fl) - y) / N
            gS = np.outer(r, v / self.sd_)
            gtt = self.forest_.backward(tt, X, V, gS)
            return loss, np.concatenate([gtt, Z.T @ r + lam * v, [r.sum()]])

        theta0 = np.concatenate([self.forest_.theta0, v0, [c0]])
        if X_val is None:
            res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=self.maxiter))
            self.theta_ = res.x
            return self
        Xv = np.asarray(X_val, dtype=np.float64)
        yv = np.asarray(y_val).astype(np.float64)
        state = dict(best=_logloss(logits(theta0, Xv), yv), theta=theta0.copy(), stall=0)

        def cb(intermediate_result):
            th = intermediate_result.x
            ll = _logloss(logits(th, Xv), yv)
            if ll < state["best"] - 1e-7:
                state.update(best=ll, theta=th.copy(), stall=0)
            else:
                state["stall"] += 1
                if state["stall"] >= self.patience:
                    raise StopIteration

        minimize(f, theta0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=self.maxiter))
        self.theta_ = state["theta"]
        return self

    def _split(self):
        nt, T = len(self.forest_.theta0), len(self.forest_.roots)
        return self.theta_[:nt], self.theta_[nt:nt + T], float(self.theta_[-1])

    def decision_function(self, X):
        tt, v, c = self._split()
        S, _ = self.forest_.outputs(tt, np.asarray(X, dtype=np.float64))
        return ((S - self.mu_) / self.sd_) @ v + c

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def compile(self):
        from .forge import CompiledSource, emit_tuned_gp_c
        tt, v, c = self._split()
        return CompiledSource(emit_tuned_gp_c(self.forest_, tt, v, c, self.mu_, self.sd_),
                              self.n_features_)
