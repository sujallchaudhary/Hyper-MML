"""
Soft-computing extensions for Hyper-MML.

Three families of techniques are provided, all optional and switched on from train.py:

1. Fuzzy-rough feature selection (FuzzyRoughSelector)
   Rough-set theory finds a *reduct*: a minimal feature subset that preserves the
   discernibility of the emotion classes.  Because EEG / audio / video features are
   continuous, the crisp rough set is replaced by its fuzzy-rough generalisation
   (fuzzy similarity relation + Kleene-Dienes implicator), so no discretisation is needed.
   Two reduct-search strategies are offered:
     * QuickReduct  - greedy forward search on the fuzzy-rough dependency degree
     * GA           - genetic-algorithm search (optimisation technique) maximising
                      dependency degree while minimising reduct size
   The fuzzy-rough positive-region membership of every training sample also tells us how
   "certain" that sample is; boundary-region (ambiguous) samples get a smaller loss weight.

2. Fuzzy modality-reliability gate (FuzzyModalityGate)
   A Takagi-Sugeno-Kang fuzzy inference system that decides, per sample, how much to trust
   each modality (EEG / audio / video) before hypergraph fusion.
   Inputs  : certainty (1 - normalised entropy of a modality-specific classifier) and
             agreement (cosine similarity with the other modalities).
   Fuzzy sets: LOW / MEDIUM / HIGH Gaussian membership functions (learnable).
   Rules   : 3 x 3 = 9 rules, product t-norm, learnable zero-order consequents.

3. Particle Swarm Optimisation (PSO)
   A generic PSO used by tune_pso.py to tune hyper-parameters.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================
# 1. Fuzzy-rough feature selection
# =====================================================================================
def _similarity(col, sigma):
    """Fuzzy similarity relation of a single (standardised) feature: max(0, 1-|a(x)-a(y)|/sigma)."""
    d = np.abs(col[:, None] - col[None, :])
    return np.maximum(0.0, 1.0 - d / sigma).astype(np.float32)


def _gamma_from_relation(R, diff_mask):
    """
    Fuzzy-rough dependency degree for a crisp decision attribute.
    mu_POS(x) = inf_{y : d(y)!=d(x)} I(R(x,y), 0) = 1 - max_{y : d(y)!=d(x)} R(x,y)   (Kleene-Dienes)
    gamma = mean_x mu_POS(x)
    """
    worst = np.where(diff_mask, R, 0.0).max(axis=1)
    mu = 1.0 - worst
    return float(mu.mean()), mu


class FuzzyRoughSelector:
    """
    Parameters
    ----------
    sigma        : width of the fuzzy similarity relation (in standard deviations)
    pool_size    : number of candidate features kept after the ANOVA-F pre-filter
    max_features : upper bound on reduct size
    tol          : QuickReduct stops when gamma >= gamma(all pool features) - tol
    max_samples  : training samples are (stratified) sub-sampled to this size for the search
    """

    def __init__(self, sigma=0.5, pool_size=200, max_features=64, tol=1e-3,
                 max_samples=500, seed=0):
        self.sigma = sigma
        self.pool_size = pool_size
        self.max_features = max_features
        self.tol = tol
        self.max_samples = max_samples
        self.seed = seed
        self.selected_ = None

    # ------------------------------------------------------------------ helpers
    def _prepare(self, X, y):
        rng = np.random.RandomState(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-8
        Z = (X - self.mean_) / self.std_
        if len(y) > self.max_samples:  # stratified subsample keeps the search tractable
            keep = []
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                k = max(2, int(round(self.max_samples * len(idx) / len(y))))
                keep.extend(rng.choice(idx, min(k, len(idx)), replace=False))
            keep = np.array(sorted(keep))
            Z, y = Z[keep], y[keep]
        return Z, y

    @staticmethod
    def _anova_f(Z, y):
        """One-way ANOVA F statistic per feature (between-class / within-class variance)."""
        classes = np.unique(y)
        grand = Z.mean(0)
        ssb = np.zeros(Z.shape[1]); ssw = np.zeros(Z.shape[1])
        for c in classes:
            Zc = Z[y == c]
            ssb += len(Zc) * (Zc.mean(0) - grand) ** 2
            ssw += ((Zc - Zc.mean(0)) ** 2).sum(0)
        return (ssb / (len(classes) - 1)) / (ssw / (len(y) - len(classes)) + 1e-12)

    # ------------------------------------------------------------------ search
    def fit(self, X, y, method="quickreduct", ga_generations=40, ga_pop=30, verbose=True):
        Z, y = self._prepare(X, y)
        n, d = Z.shape
        diff = y[:, None] != y[None, :]

        # filter step: keep the individually most class-discriminative features as the search pool
        scores = self._anova_f(Z, y)
        pool = np.argsort(-scores)[: min(self.pool_size, d)]
        Rs = np.stack([_similarity(Z[:, j], self.sigma) for j in pool])  # [p, n, n]
        p = len(pool)

        R_all = Rs.min(axis=0)
        gamma_full, _ = _gamma_from_relation(R_all, diff)
        self.gamma_full_ = gamma_full

        qr_sel, qr_hist = self._quickreduct(Rs, diff, gamma_full)
        if method == "quickreduct":
            sel, hist = qr_sel, qr_hist
        elif method == "ga":
            sel, hist = self._ga(Rs, diff, seed_sel=qr_sel, generations=ga_generations, pop=ga_pop)
        else:
            raise ValueError("method must be 'quickreduct' or 'ga'")

        self.selected_ = np.sort(pool[sel])
        self.history_ = hist
        self.gamma_ = self._gamma_of(Rs, diff, sel)
        if verbose:
            print(f"[fuzzy-rough/{method}] {d} features -> pool {p} -> reduct {len(sel)} | "
                  f"gamma(reduct)={self.gamma_:.4f} gamma(pool)={gamma_full:.4f}")
        return self

    @staticmethod
    def _gamma_of(Rs, diff, sel):
        if len(sel) == 0:
            return 0.0
        return _gamma_from_relation(Rs[list(sel)].min(axis=0), diff)[0]

    def _quickreduct(self, Rs, diff, gamma_full):
        p = Rs.shape[0]
        R = np.ones_like(Rs[0])
        sel, hist, cur = [], [], 0.0
        remaining = set(range(p))
        while remaining and len(sel) < self.max_features:
            best_j, best_g = None, -1.0
            for j in remaining:
                g, _ = _gamma_from_relation(np.minimum(R, Rs[j]), diff)
                if g > best_g:
                    best_j, best_g = j, g
            if best_g <= cur + 1e-6:  # no improvement
                break
            sel.append(best_j)
            remaining.discard(best_j)
            R = np.minimum(R, Rs[best_j])
            cur = best_g
            hist.append(cur)
            if cur >= gamma_full - self.tol:
                break
        return np.array(sel, dtype=int), hist

    def _ga(self, Rs, diff, seed_sel, generations, pop, w_gamma=0.9, p_mut=None, elite=2):
        """Genetic algorithm over bit-strings: fitness = w*gamma + (1-w)*(1 - |B|/p)."""
        rng = np.random.RandomState(self.seed + 1)
        p = Rs.shape[0]
        p_mut = p_mut or 1.0 / p
        cache = {}

        def fitness(bits):
            key = bits.tobytes()
            if key not in cache:
                idx = np.where(bits)[0]
                if len(idx) == 0 or len(idx) > self.max_features:
                    cache[key] = 0.0
                else:
                    g = self._gamma_of(Rs, diff, idx)
                    cache[key] = w_gamma * g + (1 - w_gamma) * (1 - len(idx) / p)
            return cache[key]

        P = (rng.rand(pop, p) < min(0.1, self.max_features / (2.0 * p))).astype(np.uint8)
        P[0] = 0
        P[0, seed_sel] = 1  # hybrid: warm-start with the greedy reduct
        hist = []
        for _ in range(generations):
            fit = np.array([fitness(b) for b in P])
            order = np.argsort(-fit)
            hist.append(float(fit[order[0]]))
            new = [P[i].copy() for i in order[:elite]]
            while len(new) < pop:
                a = max(rng.choice(pop, 3), key=lambda i: fit[i])  # tournament selection
                b = max(rng.choice(pop, 3), key=lambda i: fit[i])
                mask = rng.rand(p) < 0.5  # uniform crossover
                child = np.where(mask, P[a], P[b])
                flip = rng.rand(p) < p_mut  # bit-flip mutation
                child = np.where(flip, 1 - child, child).astype(np.uint8)
                new.append(child)
            P = np.array(new)
        fit = np.array([fitness(b) for b in P])
        return np.where(P[int(np.argmax(fit))])[0], hist

    # ------------------------------------------------------------------ use
    def transform(self, X):
        return np.asarray(X)[:, self.selected_]

    def positive_region_membership(self, X, y):
        """Fuzzy-rough positive-region membership (in [0,1]) of every sample w.r.t. the reduct."""
        Z = ((np.asarray(X, dtype=np.float64) - self.mean_) / self.std_)[:, self.selected_]
        y = np.asarray(y).astype(int).ravel()
        diff = y[:, None] != y[None, :]
        R = np.ones((len(y), len(y)), dtype=np.float32)
        for j in range(Z.shape[1]):
            R = np.minimum(R, _similarity(Z[:, j], self.sigma))
        return _gamma_from_relation(R, diff)[1]


def rough_sample_weights(mu, floor=0.3):
    """Map positive-region membership to loss weights: boundary samples -> `floor`, certain -> 1."""
    mu = np.asarray(mu, dtype=np.float64)
    span = mu.max() - mu.min()
    scaled = (mu - mu.min()) / span if span > 1e-9 else np.ones_like(mu)
    return (floor + (1.0 - floor) * scaled).astype(np.float32)


# =====================================================================================
# 2. Fuzzy modality-reliability gate (Takagi-Sugeno-Kang)
# =====================================================================================
class FuzzyModalityGate(nn.Module):
    LEVELS = ("LOW", "MEDIUM", "HIGH")

    def __init__(self, dim, n_classes, n_mod=3, mod_names=("EEG", "AUDIO", "VIDEO")):
        super().__init__()
        self.n_mod, self.n_classes, self.mod_names = n_mod, n_classes, mod_names
        # modality-specific auxiliary classifiers -> "certainty" input
        self.aux = nn.ModuleList([nn.Linear(dim, n_classes) for _ in range(n_mod)])
        # Gaussian membership functions for the 2 fuzzy variables (certainty, agreement)
        self.centers = nn.Parameter(torch.tensor([[0.0, 0.5, 1.0]] * 2))
        self.log_sigma = nn.Parameter(torch.full((2, 3), math.log(0.25)))
        # TSK zero-order consequents (logits); prior: weight grows with certainty (60%) and agreement (40%)
        lvl = torch.tensor([0.0, 0.5, 1.0])
        prior = 0.25 + 0.5 * (0.6 * lvl[:, None] + 0.4 * lvl[None, :])
        self.consequent = nn.Parameter(torch.log(prior / (1 - prior)))
        self.last_weights = None  # [N, n_mod], kept for interpretability
        self.aux_log_probs = []

    def memberships(self, v, i):
        """v: [N] in [0,1] -> [N,3] membership degrees of fuzzy variable i."""
        sigma = self.log_sigma[i].exp()
        return torch.exp(-0.5 * ((v[:, None] - self.centers[i]) / sigma) ** 2)

    def forward(self, feats):
        assert len(feats) == self.n_mod
        total = sum(feats)
        weights, self.aux_log_probs = [], []
        for m, f in enumerate(feats):
            logits = self.aux[m](f)
            logp = F.log_softmax(logits, dim=-1)
            self.aux_log_probs.append(logp)
            p = logp.exp().detach()
            entropy = -(p * logp.detach()).sum(-1) / math.log(self.n_classes)
            certainty = (1.0 - entropy).clamp(0, 1)
            others = (total - f) / (self.n_mod - 1)
            agreement = (F.cosine_similarity(f.detach(), others.detach(), dim=-1) + 1) / 2
            mu_c, mu_g = self.memberships(certainty, 0), self.memberships(agreement, 1)
            firing = mu_c[:, :, None] * mu_g[:, None, :]  # [N,3,3] rule strengths (product t-norm)
            out = (firing * torch.sigmoid(self.consequent)).sum((1, 2)) / (firing.sum((1, 2)) + 1e-6)
            weights.append(out)
        w = torch.stack(weights, dim=1)  # [N, M]
        w = self.n_mod * w / (w.sum(1, keepdim=True) + 1e-6)  # mean weight 1 -> keeps feature scale
        self.last_weights = w.detach()
        return [f * w[:, m:m + 1] for m, f in enumerate(feats)]

    def aux_loss(self, label, loss_fn):
        return sum(loss_fn(lp, label) for lp in self.aux_log_probs) / len(self.aux_log_probs)

    def describe(self):
        """Human-readable fuzzy sets and rule base (for the report)."""
        names = ("certainty", "agreement")
        lines = []
        for i, nm in enumerate(names):
            for k, lv in enumerate(self.LEVELS):
                lines.append(f"  {nm:<9} {lv:<6}: Gaussian(c={self.centers[i, k].item():.2f}, "
                             f"sigma={self.log_sigma[i, k].exp().item():.2f})")
        cons = torch.sigmoid(self.consequent)
        for a, la in enumerate(self.LEVELS):
            for b, lb in enumerate(self.LEVELS):
                lines.append(f"  IF certainty is {la:<6} AND agreement is {lb:<6} THEN trust = {cons[a, b].item():.2f}")
        return "\n".join(lines)


# =====================================================================================
# Loss with per-sample (rough-set) weights
# =====================================================================================
def weighted_nll(log_prob, label, class_weight=None, sample_weight=None):
    """Weighted NLL; reduces to nn.NLLLoss(weight=class_weight) when sample_weight is None."""
    nll = -log_prob.gather(1, label.view(-1, 1)).squeeze(1)
    w = torch.ones_like(nll)
    if class_weight is not None:
        w = w * class_weight.to(nll.device)[label]
    if sample_weight is not None:
        w = w * sample_weight.to(nll.device)
    return (w * nll).sum() / w.sum().clamp_min(1e-8)


# =====================================================================================
# 3. Particle Swarm Optimisation
# =====================================================================================
class PSO:
    """Maximises f over a box.  `evaluate_batch(list_of_positions) -> list_of_fitness`
    lets the caller evaluate a whole swarm in parallel (e.g. on Modal)."""

    def __init__(self, bounds, n_particles=6, iters=4, w=0.7, c1=1.5, c2=1.5, seed=0):
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.n, self.iters, self.w, self.c1, self.c2 = n_particles, iters, w, c1, c2
        self.rng = np.random.RandomState(seed)

    def run(self, evaluate_batch, log=print):
        d = len(self.lo)
        span = self.hi - self.lo
        x = self.lo + self.rng.rand(self.n, d) * span
        v = (self.rng.rand(self.n, d) - 0.5) * span * 0.2
        fit = np.array(evaluate_batch([xi.copy() for xi in x]), dtype=float)
        pbest, pfit = x.copy(), fit.copy()
        g = int(np.argmax(pfit))
        gbest, gfit = pbest[g].copy(), float(pfit[g])
        history = [gfit]
        log(f"[PSO] iter 0 best={gfit:.4f}")
        for it in range(1, self.iters + 1):
            r1, r2 = self.rng.rand(self.n, d), self.rng.rand(self.n, d)
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            x = np.clip(x + v, self.lo, self.hi)
            fit = np.array(evaluate_batch([xi.copy() for xi in x]), dtype=float)
            better = fit > pfit
            pbest[better], pfit[better] = x[better], fit[better]
            g = int(np.argmax(pfit))
            if pfit[g] > gfit:
                gbest, gfit = pbest[g].copy(), float(pfit[g])
            history.append(gfit)
            log(f"[PSO] iter {it} best={gfit:.4f}")
        return gbest, gfit, history
