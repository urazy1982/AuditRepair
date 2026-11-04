"""Core of the Audit-Repair-Abstain pilot.

Inputs everywhere: an (n, m) integer matrix L of source labels (-1 = missing),
K classes, and an oracle that reveals gold labels (each reveal is charged).

Contents
  dawid_skene        multiclass Dawid-Skene EM, with optional clamped (audited) items
  WorBettingTest     one-sided anytime-valid test for a finite-population mean,
                     sampling without replacement (Waudby-Smith & Ramdas, Prop. 4 / eq. 34)
  candidate_groups   observable audit groups: (source s, current label c, alternative b)
  run_*              the policies compared in the pilot
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Dawid-Skene
# ---------------------------------------------------------------------------
def dawid_skene(L: np.ndarray, K: int, clamp: dict[int, int] | None = None,
                n_iter: int = 50, smooth: float = 0.01, tol: float = 1e-6, *, observation_codes=None) -> np.ndarray:
    """Return posterior P(y_i = k) of shape (n, K).

    clamp: {item index: gold label}. Clamped items keep a one-hot posterior,
    so audited labels also inform every source's confusion matrix.
    """
    n, m = L.shape
    clamp = clamp or {}
    # init: soft majority vote
    T = np.full((n, K), smooth)
    for j in range(m):
        ok = L[:, j] >= 0
        T[np.where(ok)[0], L[ok, j]] += 1.0
    T /= T.sum(1, keepdims=True)
    idx = np.fromiter(clamp.keys(), int, len(clamp))
    lab = np.fromiter(clamp.values(), int, len(clamp))
    if len(idx):
        T[idx] = 0.0
        T[idx, lab] = 1.0

    observations = L if observation_codes is None else observation_codes
    width = K if observation_codes is None else max(1, int(observations.max()) + 1)
    m = observations.shape[1]
    prev = None
    for _ in range(n_iter):
        prior = T.mean(0) + 1e-12
        # confusion matrices pi[j, true, given]
        pi = np.full((m, K, width), smooth)
        for j in range(m):
            ok = observations[:, j] >= 0
            np.add.at(pi[j].T, observations[ok, j], T[ok])  # pi[j][:, given] += T
        pi /= pi.sum(2, keepdims=True)
        logT = np.tile(np.log(prior), (n, 1))
        for j in range(m):
            ok = observations[:, j] >= 0
            logT[ok] += np.log(pi[j][:, observations[ok, j]]).T
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT)
        T /= T.sum(1, keepdims=True)
        if len(idx):
            T[idx] = 0.0
            T[idx, lab] = 1.0
        if prev is not None and np.abs(T - prev).max() < tol:
            break
        prev = T
    return T


def majority_vote(L: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n, m = L.shape
    C = np.zeros((n, K))
    for j in range(m):
        ok = L[:, j] >= 0
        np.add.at(C, (np.where(ok)[0], L[ok, j]), 1)
    C += rng.random(C.shape) * 1e-3  # random tie-break
    return C.argmax(1)


# ---------------------------------------------------------------------------
# Anytime-valid one-sided test, sampling without replacement
# ---------------------------------------------------------------------------
@dataclass
class WorBettingTest:
    """Tests H0: mean of a finite population (values in [0,1]) <= m0.

    Capital K_t = prod_i (1 + lam_i (X_i - m_i^WoR)), with
    m_i^WoR = (N m0 - sum_{j<i} X_j) / (N - i + 1)   (Waudby-Smith & Ramdas, eq. 33-34).
    Under H0 (mean <= m0), E[X_i | past] <= m_i^WoR, so with lam_i >= 0 the capital
    is a nonnegative supermartingale; Ville's inequality gives
    P(sup_t K_t >= 1/alpha) <= alpha  -> valid under optional stopping.
    Betting fraction: aGRAPA-style predictable plug-in, clipped to [0, c / m_i].
    """
    N: int
    m0: float = 0.5
    alpha: float = 0.05
    c: float = 0.5
    t: int = 0
    s: float = 0.0           # sum of X
    ss: float = 0.0          # sum of squared deviations (for variance)
    log_cap: float = 0.0
    xs: list = field(default_factory=list)

    def m_wor(self) -> float:
        rem = self.N - self.t
        if rem <= 0:
            return self.m0
        return (self.N * self.m0 - self.s) / rem

    def update(self, x: float) -> None:
        mw = self.m_wor()
        eps = 1e-12
        if mw < -eps:
            # observed sum already exceeds N*m0: H0 (mean <= m0) is impossible
            self.log_cap = math.inf
        elif mw <= eps:
            # boundary: H0 stays possible only if every remaining value is 0,
            # so a positive observation refutes it; a zero changes nothing
            if x > eps:
                self.log_cap = math.inf
        elif mw < 1:
            mu_hat = (0.5 + self.s) / (self.t + 1)
            var_hat = (0.25 + self.ss) / (self.t + 1)
            gap = mu_hat - mw
            lam = gap / (var_hat + gap * gap) if gap > 0 else 0.0
            lam = min(max(lam, 0.0), self.c / mw)
            self.log_cap += math.log1p(lam * (x - mw))
        # mw >= 1 means H0 can't be rejected any more; capital frozen
        mu_prev = (0.5 + self.s) / (self.t + 1)
        self.t += 1
        self.s += x
        self.ss += (x - mu_prev) ** 2
        self.xs.append(x)

    @property
    def rejected(self) -> bool:
        return self.log_cap >= math.log(1.0 / self.alpha)

    @property
    def mean(self) -> float:
        return self.s / max(self.t, 1)


class WorLowerCS:
    """Time-uniform lower confidence sequence for the mean mu of a finite population in [0, 1],
    sampled uniformly without replacement, built from WorBettingTest-type tests on a grid
    m_0 = 0 < m_1 < ... < m_G = 1 (spacing delta).

    For each grid point m_j, H0_j: mu <= m_j is tested by a nonnegative capital process;
    "rejected by time t" is sticky. The bound is
        lower(t) = m_{k_t - 1},   k_t = smallest index never rejected up to t   (0 if k_t = 0).

    Validity (one test suffices, no monotonicity across grid points needed):
    let j+ be the smallest index with m_{j+} >= mu. H0_{j+} is true, so by Ville's inequality
    P(H0_{j+} is ever rejected) <= alpha. On the complement, j+ is never rejected, hence
    k_t <= j+ for all t and lower(t) <= m_{j+ - 1} < mu (minimality of j+), or lower(t) = 0 <= mu.
    So P(exists t: lower(t) > mu) <= alpha, i.e. the bound holds at all t simultaneously and
    optional stopping is allowed. Discretization costs at most one grid step (delta).

    remainder_gain_lb() turns this into a bound on the quantity a repair actually changes:
    total gain on the NOT-yet-audited items = N * mean_gain - sum of audited gains,
    with gain d = 2x - 1 in {-1, 0, 1}; on the same event it is a valid lower bound.
    """

    def __init__(self, N: int, alpha: float = 0.05, grid: int = 401, c: float = 0.5):
        self.N, self.alpha, self.c = N, alpha, c
        self.m = np.linspace(0.0, 1.0, grid)
        self.log_cap = np.zeros(grid)
        self.t, self.s, self.ss = 0, 0.0, 0.0
        self._lower = 0.0
        self._thr = math.log(1.0 / alpha)
        self._rejected = np.zeros(grid, dtype=bool)   # sticky rejections

    def update(self, x: float) -> None:
        rem = self.N - self.t
        mw = (self.N * self.m - self.s) / rem          # m_t^WoR for every grid null
        mu_hat = (0.5 + self.s) / (self.t + 1)
        var_hat = (0.25 + self.ss) / (self.t + 1)
        eps = 1e-12
        impossible = mw < -eps                          # observed sum already exceeds N*m
        boundary = (~impossible) & (mw <= eps)          # only zeros remain possible
        live = (~impossible) & (~boundary) & (mw < 1)
        gap = mu_hat - mw
        lam = np.where(gap > 0, gap / (var_hat + gap * gap), 0.0)
        lam = np.clip(lam, 0.0, self.c / np.where(mw > eps, mw, 1.0))
        inc = np.zeros_like(self.m)
        inc[live] = np.log1p(lam[live] * (x - mw[live]))
        self.log_cap += inc
        self.log_cap[impossible] = np.inf
        if x > eps:
            self.log_cap[boundary] = np.inf
        mu_prev = mu_hat
        self.t += 1
        self.s += x
        self.ss += (x - mu_prev) ** 2
        self._rejected |= self.log_cap >= self._thr
        never = np.where(~self._rejected)[0]           # m = 1 is a true null for every population
        k = int(never[0]) if len(never) else len(self.m)
        self._lower = float(self.m[k - 1]) if k > 0 else 0.0

    @property
    def lower(self) -> float:
        return self._lower

    def remainder_gain_lb(self) -> float:
        audited_gain = 2 * self.s - self.t
        return self.N * (2 * self._lower - 1) - audited_gain

    @property
    def mean(self) -> float:
        return self.s / max(self.t, 1)


# ---------------------------------------------------------------------------
# Observable audit groups
# ---------------------------------------------------------------------------
@dataclass
class Group:
    source: int
    cur: int        # current aggregated label of the group's items
    alt: int        # label the dissenting source proposes
    items: np.ndarray
    prior_score: float


def candidate_groups(L: np.ndarray, cur: np.ndarray, post: np.ndarray,
                     min_size: int = 30) -> list[Group]:
    """Group = items whose current label is c while source s says b != c.

    Defined only from observable labels, fixed before any audit.
    prior_score = |group| * mean DS posterior of the alternative (used only to
    order audits; it does not affect validity).
    """
    groups = []
    n, m = L.shape
    for s in range(m):
        dis = (L[:, s] >= 0) & (L[:, s] != cur)
        for c in np.unique(cur[dis]):
            for b in np.unique(L[dis & (cur == c), s]):
                items = np.where(dis & (cur == c) & (L[:, s] == b))[0]
                if len(items) >= min_size:
                    score = len(items) * post[items, b].mean()
                    groups.append(Group(s, int(c), int(b), items, float(score)))
    groups.sort(key=lambda g: -g.prior_score)
    return groups


def candidate_groups_disjoint(L: np.ndarray, cur: np.ndarray, post: np.ndarray,
                              min_size: int = 30) -> list[Group]:
    """Fixed, DISJOINT candidate groups from observable data only (in run_group_repair: after the
    warm-up audits, before any group-certification sampling).

    Every item whose current label c is contradicted by at least one source gets exactly one
    candidate repair c -> b, where b is the most frequent dissenting label (ties -> smallest
    label id). Items are grouped by (c, b, set of sources voting b). Each hypothesis therefore
    has a predefined replacement label and no item appears in two groups.
    `source` holds the dissenting-source signature as a tuple.
    """
    n, m = L.shape
    keys: dict = {}
    for i in range(n):
        votes = [(j, L[i, j]) for j in range(m) if L[i, j] >= 0 and L[i, j] != cur[i]]
        if not votes:
            continue
        cnt: dict = {}
        for _, lab in votes:
            cnt[lab] = cnt.get(lab, 0) + 1
        b = min(cnt, key=lambda k: (-cnt[k], k))
        sig = tuple(j for j, lab in votes if lab == b)
        keys.setdefault((int(cur[i]), int(b), sig), []).append(i)
    groups = []
    for (c, b, sig), items in keys.items():
        if len(items) >= min_size:
            items = np.array(items)
            groups.append(Group(sig, c, b, items, float(len(items) * post[items, b].mean())))
    groups.sort(key=lambda g: -g.prior_score)
    return groups


# ---------------------------------------------------------------------------
# Oracle with budget accounting
# ---------------------------------------------------------------------------
class Oracle:
    def __init__(self, gold: np.ndarray, budget: int):
        self.gold = gold
        self.budget = budget
        self.revealed: dict[int, int] = {}

    @property
    def left(self) -> int:
        return self.budget - len(self.revealed)

    def query(self, i: int) -> int:
        if i not in self.revealed:
            if self.left <= 0:
                raise RuntimeError("budget exhausted")
            self.revealed[i] = int(self.gold[i])
        return self.revealed[i]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def uncertainty_fill(L, K, oracle: Oracle, clamp: dict, batch: int = 50,
                     exclude: set | None = None):
    """Spend the remaining budget on the most uncertain DS items, re-fitting DS per batch."""
    exclude = exclude or set()
    post = dawid_skene(L, K, clamp)
    while oracle.left > 0:
        unc = 1.0 - post.max(1)
        order = np.argsort(-unc, kind="stable")  # ties (common: unanimous items) broken by the seeded item order, not by the NumPy/CPU sort path
        picked = [i for i in order if i not in oracle.revealed and i not in exclude][: min(batch, oracle.left)]
        if not picked:
            break
        for i in picked:
            clamp[i] = oracle.query(i)
        post = dawid_skene(L, K, clamp)
    return post


def run_ds_uncertainty(L, K, gold, budget):
    """Baseline: DS + audits on the most uncertain items, DS re-fit with audited labels."""
    oracle = Oracle(gold, budget)
    clamp: dict = {}
    post = uncertainty_fill(L, K, oracle, clamp)
    return post.argmax(1), {"audited": len(oracle.revealed), "repaired": 0, "c2i": 0}


def run_ds_same_audits(L, K, gold, audited: dict):
    """Decisive baseline: DS re-fit with EXACTLY the labels another policy audited."""
    post = dawid_skene(L, K, dict(audited))
    return post.argmax(1), {"audited": len(audited), "repaired": 0, "c2i": 0}


def run_group_repair(L, K, gold, budget, alpha=0.05, k_max=80,
                     futility_after=10, futility_mean=0.45, ref_share=0.0,
                     group_share=0.7, min_size=30, max_groups=10, stop_after_fails=6, seed=0,
                     disjoint=True, order="size", max_audit_frac=0.5):
    """Audit-Repair-Abstain (heuristic allocation policy, see README).

    1. Optional warm-up: spend ref_share of the budget on DS uncertainty audits.
    2. FREEZE the reference vector `ref` = DS argmax (with warm-up audits clamped).
       Every certificate compares the repair c -> b against exactly these labels.
    3. Candidate groups from observable labels and `ref`, fixed and disjoint by default
       (candidate_groups_disjoint), each with a predefined replacement label; shortlist of
       max_groups fixed AFTER the warm-up and BEFORE any group-certification sampling;
       Bonferroni alpha / max_groups.
    4. Audit each group in uniformly random order; x = (1[y=b] - 1[y=ref] + 1) / 2.
       guarded:                    repair iff the time-uniform lower bound on the gain over the
                                   UNAUDITED remainder (WorLowerCS.remainder_gain_lb) is > 0
       unguarded (same evidence):  repair iff the audited mean of x > 0.5 at the same stopping time
       Both decisions come from the SAME audited items, isolating the safeguard.
    5. Remaining budget -> uncertainty audits; DS re-fit with all audits.
       Final labels: audited -> gold; unaudited items in repaired groups -> b
       (the re-fit never overwrites them, so the realized change is exactly the certified one);
       everything else -> re-fit DS.
    Guarantee (false certification): with probability at least 1 - alpha, NO accepted
    candidate repair on the shortlist has non-positive net gain on the items it actually
    changes (the unaudited remainder of its group at the moment of acceptance), relative to
    the frozen reference labels. It says nothing about which repaired labels survive a later
    PAC stage, which may replace some of them with expert labels.
    Per group: the lower confidence sequence holds at all audit sizes simultaneously, so
    stopping at acceptance is valid; Bonferroni over the pre-registered shortlist. Each test is
    valid conditionally on earlier groups (fresh uniform order, population fixed at its start).
    It is not a statement that an already accepted repair is harmful with conditional
    probability <= alpha/|shortlist|, and it promises nothing about the re-fitted DS labels
    or downstream training loss.

    The unguarded result is a matched-evidence REPLAY of the guarded trajectory; it never
    influences which items are audited.
    Returns {"guarded": (labels, info), "unguarded": (labels, info)}.
    """
    if not (0 <= ref_share <= 1 and 0 <= group_share <= 1):
        raise ValueError("ref_share and group_share must lie in [0, 1]")
    oracle = Oracle(gold, budget)
    clamp: dict = {}
    warm = int(ref_share * budget)
    if warm:
        sub = Oracle(gold, warm)
        uncertainty_fill(L, K, sub, clamp)
        for i, y in sub.revealed.items():
            oracle.revealed[i] = y
    post_ref = dawid_skene(L, K, clamp)
    ref = post_ref.argmax(1)
    for i, y in clamp.items():
        ref[i] = y
    make = candidate_groups_disjoint if disjoint else candidate_groups
    groups = make(L, ref, post_ref, min_size=min_size)
    if order == "size":       # largest leverage first; fixed before any audit
        groups = sorted(groups, key=lambda g: -len(g.items))
    elif callable(order):
        groups = sorted(groups, key=order, reverse=True)
    groups = groups[:max_groups]
    a_g = alpha / max(len(groups), 1)
    rng = np.random.default_rng(seed)
    rep = {"guarded": {}, "unguarded": {}}
    conf = {"guarded": {}, "unguarded": {}}
    claimed: set = set()          # items already in a group the guarded rule repaired
    # group phase may use group_share of what is LEFT after warm-up -> never exceeds budget
    group_budget = warm + int(group_share * (budget - warm))
    fails, log = 0, []

    for g in groups:
        if len(oracle.revealed) >= group_budget or fails >= stop_after_fails:
            break
        # Warm-up labels equal ref, so any unsampled warm-up item has
        # nonpositive gain. Earlier group audits can have positive gain
        # under an overlapping candidate and must be excluded up front.
        pop = np.array([i for i in g.items
                        if i not in claimed and (i in clamp or i not in oracle.revealed)],
                       dtype=int)
        if len(pop) < min_size:
            continue
        test = WorLowerCS(N=len(pop), alpha=a_g)
        decision = "abstain"
        stop_reason = "population_exhausted"
        history = []
        known_at_start = sorted(int(i) for i in pop if int(i) in oracle.revealed)
        n_alt = 0
        for t, i in enumerate(rng.permutation(pop), 1):
            if len(oracle.revealed) >= group_budget or (int(i) not in oracle.revealed and oracle.left <= 0):
                stop_reason = "budget"
                break
            y = oracle.query(int(i))
            n_alt += int(y == g.alt)
            z = int(y == g.alt) - int(y == ref[i])
            test.update((z + 1) / 2)
            history.append({"item": int(i), "gold": int(y), "gain": int(z)})
            # certify the action actually taken: gain on the UNAUDITED remainder must be > 0
            if test.t < len(pop) and test.remainder_gain_lb() > 0:
                decision = "repair"
                stop_reason = "certified"
                break
            if t >= futility_after and test.mean < futility_mean:
                decision = "reject"
                stop_reason = "futility"
                break
            if t >= k_max or t >= max_audit_frac * len(pop):
                stop_reason = "audit_cap" if t >= k_max else "fraction_cap"
                break             # little left to propagate: abstain, keep the audited corrections
        dec = {"guarded": decision == "repair", "unguarded": test.t > 0 and test.mean > 0.5}
        p_alt = n_alt / max(test.t, 1)   # audited share that truly is `alt`
        for k, ok in dec.items():
            if ok:
                for i in pop:
                    rep[k][int(i)] = g.alt
                    conf[k][int(i)] = p_alt
        if dec["guarded"]:   # populations follow the guarded run only
            claimed.update(int(i) for i in pop)
        fails = 0 if dec["guarded"] else fails + 1
        true_G = float(np.mean((gold[pop] == g.alt).astype(int) - (gold[pop] == ref[pop]).astype(int)))
        unaud = np.array([i for i in pop if int(i) not in oracle.revealed], dtype=int)
        true_rem = int(((gold[unaud] == g.alt).astype(int) - (gold[unaud] == ref[unaud]).astype(int)).sum()) if len(unaud) else 0
        members = [int(i) for i in pop]
        log.append({"source": g.source, "cur": g.cur, "alt": g.alt, "size": len(pop),
                    "membership_id": hashlib.sha256(np.asarray(members, dtype="<i8").tobytes()).hexdigest(),
                    "members": members, "known_at_start": known_at_start,
                    "audit_history": history, "alpha": a_g, "stop_reason": stop_reason,
                    "lower_mean": test.lower, "remainder_lb_exact": test.remainder_gain_lb(),
                    "audited": test.t, "audit_mean_x": round(test.mean, 3),
                    "guarded": dec["guarded"], "unguarded": dec["unguarded"],
                    "true_G_vs_ref": round(true_G, 3), "true_remainder_gain": true_rem,
                    "remainder_lb": round(test.remainder_gain_lb(), 1)})

    # The primary (guarded) trajectory is independent of the ablation: only ITS repairs are
    # excluded from the remaining uncertainty audits. The unguarded rule is then REPLAYED on
    # exactly this evidence (same audited items, same re-fit), so it never steers allocation.
    clamp = dict(oracle.revealed)
    post = uncertainty_fill(L, K, oracle, clamp, exclude=set(rep["guarded"]))
    refit = post.argmax(1)
    out = {}
    for k in ("guarded", "unguarded"):
        final = refit.copy()
        for i, b in rep[k].items():
            if i not in oracle.revealed:
                final[i] = b
        idx = np.array([i for i in rep[k] if i not in oracle.revealed], dtype=int)
        c2i = int(((ref[idx] == gold[idx]) & (final[idx] != gold[idx])).sum()) if len(idx) else 0
        i2c = int(((ref[idx] != gold[idx]) & (final[idx] == gold[idx])).sum()) if len(idx) else 0
        # harmful = the change actually applied (unaudited remainder) does not improve labels
        harmful = sum(1 for g in log if g[k] and g["true_remainder_gain"] <= 0)
        out[k] = (final, {"audited": len(oracle.revealed), "repaired": len(idx), "c2i": c2i,
                          "i2c": i2c, "groups_repaired": sum(g[k] for g in log),
                          "harmful_group_repairs": harmful, "log": log,
                          "audited_labels": dict(oracle.revealed),
                          "repair_conf": {i: c for i, c in conf[k].items() if i not in oracle.revealed},
                          "post": post})
    return out
