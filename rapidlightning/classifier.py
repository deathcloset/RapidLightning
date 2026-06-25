"""
Rapid Lightning — the user-facing classifier.

    from rapidlightning import RapidLightningClassifier

    clf = RapidLightningClassifier(n_members=10, generations=20).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]      # head-aware GP + logistic head
    fast  = clf.compile()                      # -> CompiledPredictor (~15 KB C .so)
    fast.predict_proba_one(x)                  # tens of nanoseconds
    fast.benchmark_ns()                        # measured, timed inside C

Binary tabular classification. Inputs are float32, standardized upstream
(StandardScaler on train is recommended). Multiclass uses the production
one-vs-rest head-aware path (not exposed in this reference build).
"""
import numpy as np

from .engine import (
    EvolutionConfig, evolve_ensemble_headaware, fit_lr_head, head_proba,
    member_diversity, F32,
)
from .forge import CompiledPredictor


class RapidLightningClassifier:
    def __init__(self, n_members=10, generations=20, population=160,
                 max_depth=8, num_islands=4, parsimony=1e-4, seed=1):
        self.n_members = n_members
        self.generations = generations
        self.population = population
        self.max_depth = max_depth
        self.num_islands = num_islands
        self.parsimony = parsimony
        self.seed = seed

    def _cfg(self):
        cfg = EvolutionConfig()
        cfg.num_members = self.n_members
        cfg.gens = self.generations
        cfg.pop_size = self.population
        cfg.max_depth = self.max_depth
        cfg.num_islands = self.num_islands
        cfg.parsimony = self.parsimony
        cfg.seed = self.seed
        return cfg

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=F32)
        y = np.asarray(y).astype(np.int64)
        if set(np.unique(y).tolist()) - {0, 1}:
            raise ValueError("RapidLightningClassifier (reference build) expects binary y in {0,1}.")
        self.n_features_ = X.shape[1]
        self.members_ = evolve_ensemble_headaware(X, y, self._cfg())
        self.scaler_, self.lr_ = fit_lr_head(self.members_, X, y)
        return self

    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype=F32)
        p1 = head_proba(self.members_, self.scaler_, self.lr_, X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int64)

    @property
    def n_nodes(self):
        """Total GP node count — what the compiled .so size and latency scale with."""
        return sum(m.total_size() for m in self.members_)

    def mean_member_accuracy(self, X, y):
        """Diagnostic: individual members should be near chance (they're signal generators)."""
        y = np.asarray(y)
        return float(np.mean([(m.classify(np.ascontiguousarray(X, dtype=F32)) == y).mean()
                              for m in self.members_]))

    def member_diversity(self, X):
        return member_diversity(self.members_, np.ascontiguousarray(X, dtype=F32))

    def compile(self):
        """Lower the trained ensemble to a compiled, dependency-free C predictor."""
        return CompiledPredictor(self.members_, self.scaler_, self.lr_, self.n_features_)
