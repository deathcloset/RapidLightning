"""
XGForge — compile a trained XGBoost model to a dependency-free C `.so`.

This is the *honest* baseline for latency comparisons: it lowers an
`xgboost.XGBClassifier` to nested-if/else C and compiles it with the same
`gcc -O3` flags Rapid Lightning uses, so we compare compiled-to-compiled
(one `.so` vs another) instead of against XGBoost's slow Python `predict_proba`.

Binary classification, dense fp32 input. AUC matches the source model (rank-
faithful). Requires `xgboost` (an optional/demos dependency) and a system gcc.

    from rapidlightning.xgforge import xgforge
    fast_xgb = xgforge(trained_xgb_classifier)   # -> CompiledXGB
    fast_xgb.predict_proba_one(x); fast_xgb.benchmark_ns(); fast_xgb.size_kb
"""
import os
import ctypes
import numpy as np

from .forge import _fmt_f, compile_c, _TIMING_C


class CompiledXGB:
    """A compiled XGBoost `.so` with the same interface as forge.CompiledPredictor."""

    def __init__(self, so_path, n_features):
        self.so_path = so_path
        self.n_features = n_features
        self.size_kb = os.path.getsize(so_path) / 1024
        lib = ctypes.CDLL(so_path)
        lib.predict_proba_one.restype = ctypes.c_float
        lib.predict_proba_one.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.predict_proba_time_ns.restype = ctypes.c_double
        lib.predict_proba_time_ns.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self._lib = lib

    def _ptr(self, x):
        xx = np.ascontiguousarray(x, dtype=np.float32)
        return xx.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def predict_proba_one(self, x):
        return float(self._lib.predict_proba_one(self._ptr(x)))

    def predict_proba(self, X):
        return np.array([self.predict_proba_one(r) for r in X], dtype=np.float64)

    def benchmark_ns(self, x=None, n_warmup=5000, n_iters=200000):
        if x is None:
            x = np.zeros(self.n_features, dtype=np.float32)
        return float(self._lib.predict_proba_time_ns(self._ptr(x), n_warmup, n_iters))


def emit_xgb_c(xgb_model):
    """Transcribe a trained XGBClassifier to straight C (nested if/else per tree)."""
    booster = xgb_model.get_booster()
    bs = booster.attr('base_score')
    base = float(bs) if bs else 0.5
    base_logit = float(np.log(base / (1 - base))) if 0 < base < 1 else base
    df = booster.trees_to_dataframe()
    tree_ids = sorted(df['Tree'].unique())

    lines = ['#include <math.h>', '#include <stdint.h>', '#include <time.h>', '']
    for tid in tree_ids:
        tdf = df[df['Tree'] == tid]
        nodes = {int(r['Node']): r for _, r in tdf.iterrows()}
        root = int(tdf['Node'].min())
        lines.append(f'static inline float tree_{tid}(const float* X) {{')

        def recurse(nid, indent):
            row = nodes[nid]
            feat = row['Feature']
            if feat == 'Leaf':
                lines.append('    ' * indent + f'return {_fmt_f(row["Gain"])[1:-1]};')
                return
            try:
                fi = int(str(feat).lstrip('f'))
            except ValueError:
                fi = 0
            yes = int(str(row['Yes']).split('-')[1])
            no = int(str(row['No']).split('-')[1])
            lines.append('    ' * indent + f'if (X[{fi}] < {_fmt_f(row["Split"])[1:-1]}) {{')
            recurse(yes, indent + 1)
            lines.append('    ' * indent + '} else {')
            recurse(no, indent + 1)
            lines.append('    ' * indent + '}')

        recurse(root, 1)
        lines.append('}\n')

    lines.append('float predict_proba_one(const float* X) {')
    lines.append(f'    float score = {_fmt_f(base_logit)[1:-1]};')
    for tid in tree_ids:
        lines.append(f'    score += tree_{tid}(X);')
    lines.append('    return 1.0f / (1.0f + expf(-score));')
    lines.append('}')
    lines.append(_TIMING_C)
    return '\n'.join(lines)


def xgforge(xgb_model, n_features=None):
    """Compile a trained binary XGBClassifier to a CompiledXGB `.so` (rank-faithful)."""
    if n_features is None:
        n_features = int(xgb_model.n_features_in_)
    so_path, _ = compile_c(emit_xgb_c(xgb_model))
    return CompiledXGB(so_path, n_features)
