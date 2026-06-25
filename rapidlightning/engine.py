"""
Rapid Lightning — head-aware genetic-programming engine.

The core idea (Rapid Lightning v3): the ensemble members are evolved NOT as
predictors, but as *signal generators* for a linear head. Each new member is
selected to maximize the squared correlation between its (pool-orthogonalized)
output and the linear head's residual — functional gradient boosting in feature
space. Individually the members are near-chance; together they feed a logistic
head that does the actual classifying.

Pure NumPy + scikit-learn. No XGBoost, no torch. This is the clean extraction of
the validated v3 notebook engine, and the shared core the demos build on.
"""
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

F32 = np.float32

# GP operator set (arity). 'lt' (less-than) gives axis-aligned splits; the rest
# are smooth/algebraic. This is the production "tier-3" op set.
OPS = {'add': 2, 'mul': 2, 'sub': 2, 'max': 2, 'min': 2,
       'neg': 1, 'abs': 1, 'clamp': 1, 'lt': 2}
ACTIVE = list(OPS)


class GPNode:
    """A single node in a GP expression tree (input / const / operator)."""
    __slots__ = ['ntype', 'value', 'children']

    def __init__(self, ntype, value, children=None):
        self.ntype = ntype
        self.value = value
        self.children = children or []

    def evaluate(self, X):
        if self.ntype == 'input':
            return X[:, self.value % X.shape[1]]
        if self.ntype == 'const':
            return np.full(X.shape[0], self.value, dtype=F32)
        a = [c.evaluate(X) for c in self.children]
        op = self.value
        if   op == 'add':   r = a[0] + a[1]
        elif op == 'mul':   r = a[0] * a[1]
        elif op == 'sub':   r = a[0] - a[1]
        elif op == 'max':   r = np.maximum(a[0], a[1])
        elif op == 'min':   r = np.minimum(a[0], a[1])
        elif op == 'neg':   r = -a[0]
        elif op == 'abs':   r = np.abs(a[0])
        elif op == 'clamp': r = np.clip(a[0], 0.0, 1.0)
        elif op == 'lt':    r = (a[0] < a[1]).astype(F32)
        else:               r = a[0]
        return np.clip(r, -1e6, 1e6).astype(F32)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        return 1 if not self.children else 1 + max(c.depth() for c in self.children)

    def clone(self):
        return GPNode(self.ntype, self.value, [c.clone() for c in self.children])

    def all_nodes(self):
        out = [self]
        for c in self.children:
            out.extend(c.all_nodes())
        return out


class GPIndividual:
    """One ensemble member: a small bag of GP trees (one per class)."""
    __slots__ = ['trees', 'fitness', 'train_acc']

    def __init__(self, trees, fitness=0.0, train_acc=0.0):
        self.trees = trees
        self.fitness = fitness
        self.train_acc = train_acc

    def get_scores(self, X):
        return np.stack([t.evaluate(X) for t in self.trees], axis=1)

    def classify(self, X):
        return self.get_scores(X).argmax(axis=1)

    def total_size(self):
        return sum(t.size() for t in self.trees)

    def max_depth(self):
        return max(t.depth() for t in self.trees)

    def clone(self):
        return GPIndividual([t.clone() for t in self.trees], self.fitness, self.train_acc)


def random_tree(max_depth, n_inputs, d=0):
    if d >= max_depth or (d > 1 and random.random() < 0.3):
        if random.random() < 0.7:
            return GPNode('input', random.randint(0, n_inputs - 1))
        return GPNode('const', random.uniform(-2, 2))
    op = random.choice(ACTIVE)
    return GPNode('op', op, [random_tree(max_depth, n_inputs, d + 1) for _ in range(OPS[op])])


def random_individual(n_classes, max_depth, n_inputs):
    return GPIndividual([random_tree(max_depth, n_inputs) for _ in range(n_classes)])


def mutate(ind, max_depth, n_inputs):
    m = ind.clone()
    ti = random.randint(0, len(m.trees) - 1)
    nodes = m.trees[ti].all_nodes()
    if len(nodes) <= 1 or random.random() < 0.3:
        m.trees[ti] = random_tree(max_depth, n_inputs)
    else:
        t = random.choice(nodes)
        if   t.ntype == 'const': t.value += random.gauss(0, 0.5)
        elif t.ntype == 'input': t.value = random.randint(0, n_inputs - 1)
        else:
            n = random_tree(min(3, max_depth - 1), n_inputs)
            t.ntype, t.value, t.children = n.ntype, n.value, n.children
    return m


def crossover(p1, p2, max_depth, n_inputs):
    ch = p1.clone()
    ti = random.randint(0, len(ch.trees) - 1)
    n1, n2 = ch.trees[ti].all_nodes(), p2.trees[ti].all_nodes()
    if len(n1) > 1 and len(n2) > 1:
        tgt = random.choice(n1)
        don = random.choice(n2).clone()
        tgt.ntype, tgt.value, tgt.children = don.ntype, don.value, don.children
    if ch.trees[ti].depth() > max_depth + 2:
        ch.trees[ti] = random_tree(max_depth, n_inputs)
    return ch


def tournament(pop, size=7):
    best = None
    for _ in range(size):
        c = random.choice(pop)
        if best is None or c.fitness > best.fitness:
            best = c
    return best


class EvolutionConfig:
    """Hyperparameters for member evolution (islands + migration + elitism)."""
    pop_size = 160
    num_islands = 4
    max_depth = 8
    gens = 20
    migrate_every = 12
    migrate_count = 3
    tournament_size = 7
    crossover_rate = 0.85
    mutation_rate = 0.20
    parsimony = 1e-4
    elite_pct = 0.05
    fresh_pct = 0.03
    num_members = 10
    seed = 1


def evolve_member(X, y, cfg, eval_fn, member_num=0):
    """Evolve one member with an island GA against the supplied fitness `eval_fn`."""
    n_inputs = X.shape[1]
    n_classes = int(y.max()) + 1
    random.seed(cfg.seed + member_num * 1000)
    ppi = cfg.pop_size // cfg.num_islands
    islands = [[random_individual(n_classes, cfg.max_depth, n_inputs) for _ in range(ppi)]
               for _ in range(cfg.num_islands)]
    best, best_fit = None, -np.inf
    for gen in range(1, cfg.gens + 1):
        for i in range(len(islands)):
            pop = islands[i]
            eval_fn(pop, X, y, cfg.parsimony)
            pop.sort(key=lambda z: z.fitness, reverse=True)
            if pop[0].fitness > best_fit:
                best_fit = pop[0].fitness
                best = pop[0].clone()
            new = []
            ec = max(2, int(len(pop) * cfg.elite_pct))
            for j in range(ec):
                new.append(pop[j].clone())
            while len(new) < len(pop):
                p1 = tournament(pop, cfg.tournament_size)
                p2 = tournament(pop, cfg.tournament_size)
                ch = crossover(p1, p2, cfg.max_depth, n_inputs) if random.random() < cfg.crossover_rate else p1.clone()
                if random.random() < cfg.mutation_rate:
                    ch = mutate(ch, cfg.max_depth, n_inputs)
                new.append(ch)
            for j in range(int(len(pop) * cfg.fresh_pct)):
                ri = len(pop) - 1 - j
                if ri >= ec:
                    new[ri] = random_individual(n_classes, cfg.max_depth, n_inputs)
            islands[i] = new
        if gen % cfg.migrate_every == 0:
            ni = len(islands)
            for i in range(ni):
                src = sorted(islands[i], key=lambda z: z.fitness, reverse=True)
                tgt = sorted(islands[(i + 1) % ni], key=lambda z: z.fitness)
                for mm in range(min(cfg.migrate_count, len(src), len(tgt))):
                    tgt[mm] = src[mm].clone()
                islands[(i + 1) % ni] = tgt
    return best


# ---- head-aware fitness: the heart of the method ----------------------------

def stack(members, X):
    """Concatenate every member's per-class scores into the head's feature matrix."""
    return np.concatenate([m.get_scores(X) for m in members], axis=1).astype(np.float64)


def pool_setup(members, X, y, n_classes):
    """Fit the current linear head on the existing pool and return its residual
    R = (one-hot y - head_proba), plus the projector onto the pool's span. The
    next member is rewarded for explaining R in directions the pool can't."""
    N = X.shape[0]
    Y1 = np.eye(n_classes, dtype=np.float64)[y]
    cols = [np.ones((N, 1))]
    if members:
        Phi = stack(members, X)
        Phs = StandardScaler().fit_transform(Phi)
        cols.append(Phs)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(Phs, y)
        P = np.zeros((N, n_classes))
        for j, c in enumerate(lr.classes_):
            P[:, c] = lr.predict_proba(Phs)[:, j]
    else:
        P = np.tile(np.bincount(y, minlength=n_classes) / N, (N, 1))
    B = np.concatenate(cols, axis=1)
    Bproj = B @ np.linalg.inv(B.T @ B + 1e-3 * np.eye(B.shape[1]))
    return (Y1 - P), Bproj, B


def make_headaware_eval(R, Bproj, B):
    """Fitness = sum over classes of corr(member_output ⟂ pool, residual)²."""
    Rc = R - R.mean(0)
    Rn = np.sqrt((Rc ** 2).sum(0)) + 1e-12

    def eval_pop(pop, X, y, pars):
        for ind in pop:
            try:
                S = np.nan_to_num(ind.get_scores(X).astype(np.float64),
                                  nan=0.0, posinf=1e6, neginf=-1e6)
                S_perp = S - Bproj @ (B.T @ S)
                Sc = S_perp - S_perp.mean(0)
                Sn = np.sqrt((Sc ** 2).sum(0)) + 1e-12
                corr = (Sc * Rc).sum(0) / (Sn * Rn)
                ind.fitness = float((corr ** 2).sum()) - ind.total_size() * pars
                ind.train_acc = ind.fitness
            except Exception:
                ind.fitness = -1e9
                ind.train_acc = 0.0
    return eval_pop


def evolve_ensemble_headaware(X, y, cfg):
    """Grow the ensemble one head-aware member at a time."""
    n_classes = int(y.max()) + 1
    members = []
    for m in range(1, cfg.num_members + 1):
        R, Bproj, B = pool_setup(members, X, y, n_classes)
        members.append(evolve_member(X, y, cfg, make_headaware_eval(R, Bproj, B), member_num=m))
    return members


# ---- the linear head + diagnostics ------------------------------------------

def fit_lr_head(members, Xtr, ytr):
    """Fit StandardScaler + LogisticRegression on the stacked member scores."""
    Ztr = stack(members, Xtr)
    scaler = StandardScaler().fit(Ztr)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(scaler.transform(Ztr), ytr)
    return scaler, lr


def head_proba(members, scaler, lr, X):
    return lr.predict_proba(scaler.transform(stack(members, X)))[:, 1]


def member_diversity(members, X):
    """Mean pairwise disagreement of member argmax predictions (higher = more diverse)."""
    P = np.stack([m.classify(X) for m in members], axis=0)
    M = P.shape[0]
    tot, cnt = 0.0, 0
    for i in range(M):
        for j in range(i + 1, M):
            tot += float((P[i] != P[j]).mean())
            cnt += 1
    return tot / max(cnt, 1)
