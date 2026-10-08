"""
Rapid Lightning — ahead-of-time compiler (ensemble → C → .so).

Lowers a trained head-aware ensemble (GP trees → standardize → logistic head →
sigmoid) to straight-line C99, compiles it with gcc -O3, and loads it via ctypes.
The result is a ~15 KB, dependency-free shared library whose single-row inference
runs in tens of nanoseconds — and which you can read end to end.

The emitted `.so` exposes:
    float  predict_proba_one(const float* X);
    double predict_proba_time_ns(const float* X, int n_warmup, int n_iters);
"""
import os
import ctypes
import tempfile
import subprocess

import numpy as np


def _fmt_f(v):
    """Render a Python float as a valid, parenthesized C float literal."""
    v = float(v)
    s = f"{v:.9g}"
    if '.' not in s and 'e' not in s and 'E' not in s and 'n' not in s and 'i' not in s:
        s = f"{v:.1f}"
    return '(' + s + 'f)'  # parenthesize so negative literals never form `--` / `-+`


def _emit_node(n, n_features):
    if n.ntype == 'input':
        return f"X[{n.value % n_features}]"
    if n.ntype == 'const':
        return _fmt_f(n.value)
    a = [_emit_node(c, n_features) for c in n.children]
    op = n.value
    if   op == 'add':   return f"({a[0]}+{a[1]})"
    elif op == 'sub':   return f"({a[0]}-{a[1]})"
    elif op == 'mul':   return f"({a[0]}*{a[1]})"
    elif op == 'max':   return f"fmaxf({a[0]},{a[1]})"
    elif op == 'min':   return f"fminf({a[0]},{a[1]})"
    elif op == 'neg':   return f"(-({a[0]}))"
    elif op == 'abs':   return f"fabsf({a[0]})"
    elif op == 'clamp': return f"fminf(fmaxf({a[0]},0.0f),1.0f)"
    elif op == 'lt':    return f"(({a[0]}<{a[1]})?1.0f:0.0f)"
    return a[0]


_TIMING_C = '''
double predict_proba_time_ns(const float* X, int nw, int ni){
    volatile float sink=0; for(int i=0;i<nw;++i) sink+=predict_proba_one(X);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<ni;++i) sink+=predict_proba_one(X);
    clock_gettime(CLOCK_MONOTONIC,&t1); (void)sink;
    double dt=(double)(t1.tv_sec-t0.tv_sec)*1e9+(double)(t1.tv_nsec-t0.tv_nsec);
    return dt/(double)ni;
}'''


def emit_c_source(members, scaler, lr, n_features):
    """Emit the full straight-line C for the GP→standardize→logistic→sigmoid stack."""
    lines = ['#include <math.h>', '#include <time.h>', '',
             'float predict_proba_one(const float* X){']
    mean, scale = scaler.mean_, scaler.scale_
    coef, b = lr.coef_[0], float(lr.intercept_[0])
    k, terms = 0, []
    for mem in members:
        for t in mem.trees:
            lines.append(f"    float f{k} = (float)({_emit_node(t, n_features)});")
            c = coef[k] / scale[k]  # fold the scaler into the LR weight
            terms.append(f"({_fmt_f(c)}*(f{k}-{_fmt_f(mean[k])}))")
            k += 1
    lines.append(f"    float s = {_fmt_f(b)} + " + " + ".join(terms) + ";")
    lines.append("    return 1.0f/(1.0f+expf(-s));")
    lines.append("}")
    lines.append(_TIMING_C)
    return "\n".join(lines)


def compile_c(src, gcc_flags=('-O3', '-march=native', '-ffast-math')):
    """Write C source, compile to a shared object with gcc, return its path."""
    d = tempfile.mkdtemp(prefix="rapidlightning_")
    src_path = os.path.join(d, "predict.c")
    so_path = os.path.join(d, "predict.so")
    with open(src_path, "w") as f:
        f.write(src)
    # -lm after the source: the code calls libm (expf, tanhf), and on glibc libm.so also
    # pulls in libmvec for the vectorized variants gcc emits under -ffast-math
    subprocess.run(['gcc', *gcc_flags, '-shared', '-fPIC', '-o', so_path, src_path, '-lm'], check=True)
    return so_path, src_path


class CompiledSource:
    """A loaded `.so` built from emitted C: nanosecond single-row inference, with
    in-C timing. Any emitter that defines `predict_proba_one` + the timing helper
    can be loaded this way."""

    def __init__(self, c_source, n_features):
        self.n_features = n_features
        self.c_source = c_source
        self.so_path, self.c_path = compile_c(self.c_source)
        self.size_kb = os.path.getsize(self.so_path) / 1024
        lib = ctypes.CDLL(self.so_path)
        lib.predict_proba_one.restype = ctypes.c_float
        lib.predict_proba_one.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.predict_proba_time_ns.restype = ctypes.c_double
        lib.predict_proba_time_ns.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self._lib = lib

    def _ptr(self, x):
        xx = np.ascontiguousarray(x, dtype=np.float32)
        return xx.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def predict_proba_one(self, x):
        """P(y=1) for a single feature vector, via the compiled .so."""
        return float(self._lib.predict_proba_one(self._ptr(x)))

    def predict_proba(self, X):
        return np.array([self.predict_proba_one(row) for row in X], dtype=np.float64)

    def benchmark_ns(self, x=None, n_warmup=5000, n_iters=1_000_000):
        """Measured nanoseconds per single-row inference (timed inside C, no Python in the loop)."""
        if x is None:
            x = np.zeros(self.n_features, dtype=np.float32)
        return float(self._lib.predict_proba_time_ns(self._ptr(x), n_warmup, n_iters))


class CompiledPredictor(CompiledSource):
    """A compiled GP ensemble (GP trees → standardize → logistic head → sigmoid)."""

    def __init__(self, members, scaler, lr, n_features):
        super().__init__(emit_c_source(members, scaler, lr, n_features), n_features)


# ---- emitters for the gradient-trained generators (rapidlightning.gradgen) -----

_ACT_C = {
    "tanh": "tanhf({z})",
    "relu": "fmaxf({z},0.0f)",
    "softsign": "({z}/(1.0f+fabsf({z})))",
}


def _c_array(name, values):
    return f"static const float {name}[{len(values)}] = {{" + ",".join(_fmt_f(v)[1:-1] for v in values) + "};"


def emit_ridge_c(W, b, v, c, act, direct=False):
    """Ridge generators act(w·x + b) → logistic head → sigmoid, as branchless C.
    Weights live in static const arrays, stored transposed so the inner loop runs
    across generators: under -O3 -march=native the matrix-vector product becomes
    SIMD and the activation loop calls glibc's vector tanhf (libmvec). Any head
    standardization is already folded into (v, c)."""
    K, d = W.shape
    vx, vh = (v[:d], v[d:]) if direct else (None, v)
    lines = ["#include <math.h>", "#include <time.h>", "",
             _c_array("WT", W.T.ravel()), _c_array("B", b), _c_array("V", vh)]
    if direct:
        lines.append(_c_array("VX", vx))
    lines += ["", "float predict_proba_one(const float* X){",
              f"    float z[{K}];",
              f"    for(int k=0;k<{K};++k) z[k] = B[k];",
              f"    for(int j=0;j<{d};++j){{",
              f"        const float x = X[j]; const float* w = WT + j*{K};",
              f"        for(int k=0;k<{K};++k) z[k] += w[k]*x;",
              "    }",
              f"    float s = {_fmt_f(c)};"]
    if direct:
        lines.append(f"    for(int j=0;j<{d};++j) s += VX[j]*X[j];")
    lines += [f"    for(int k=0;k<{K};++k){{ s += V[k]*{_ACT_C[act].format(z='z[k]')}; }}",
              "    return 1.0f/(1.0f+expf(-s));",
              "}", _TIMING_C]
    return "\n".join(lines)


def emit_tuned_gp_c(forest, theta, v, c, mu, sd):
    """Gradient-tuned GP trees (leaf weights, tuned constants, sigmoid-relaxed `lt`)
    → standardize → logistic head → sigmoid, as straight-line C."""
    nodes = forest.nodes

    def emit(i):
        kind, val, kids, p = nodes[i]
        if kind == "x":
            return f"({_fmt_f(theta[p])}*X[{val}])"
        if kind == "c":
            return _fmt_f(theta[p])
        a = [emit(k) for k in kids]
        if kind == "add":   return f"({a[0]}+{a[1]})"
        if kind == "sub":   return f"({a[0]}-{a[1]})"
        if kind == "mul":   return f"({a[0]}*{a[1]})"
        if kind == "max":   return f"fmaxf({a[0]},{a[1]})"
        if kind == "min":   return f"fminf({a[0]},{a[1]})"
        if kind == "neg":   return f"(-({a[0]}))"
        if kind == "abs":   return f"fabsf({a[0]})"
        if kind == "clamp": return f"fminf(fmaxf({a[0]},0.0f),1.0f)"
        if kind == "lt":
            inv_tau = 1.0 / float(np.exp(theta[p]))
            return f"(1.0f/(1.0f+expf(-({a[1]}-{a[0]})*{_fmt_f(inv_tau)})))"
        return a[0]

    lines = ["#include <math.h>", "#include <time.h>", "", "float predict_proba_one(const float* X){"]
    terms = []
    for k, root in enumerate(forest.roots):
        lines.append(f"    float f{k} = (float)({emit(root)});")
        terms.append(f"({_fmt_f(v[k] / sd[k])}*(f{k}-{_fmt_f(mu[k])}))")
    lines.append(f"    float s = {_fmt_f(c)} + " + " + ".join(terms) + ";")
    lines += ["    return 1.0f/(1.0f+expf(-s));", "}", _TIMING_C]
    return "\n".join(lines)
