"""PAC Labels (Candes, Ilyas & Zrnic, 2025), Algorithm 1 with uniform weights (pi_i = 1).

Given predictions Yhat, uncertainties U and gold labels behind an oracle, returns labels with
  P( mean_i 1[Ytilde_i != Y_i] <= eps ) >= 1 - alpha.
meanUB: a one-sided betting test of H0: L^u >= eps for every candidate u at once
(capital prod_t (1 + lam_t (eps - Z_t)), lam_t predictable in [0, c/(1-eps)]);
"L_hat^u(alpha) > eps" <=> H0 not rejected at level alpha.
"""
from __future__ import annotations

import numpy as np


def pac_labels(pred, U, gold, eps=0.05, alpha=0.1, m=500, c=0.5, already=None, seed=0):
    """already: {i: y} labels the user already owns (audits); they cost nothing extra here
    and are treated as correct with U = 0."""
    rng = np.random.default_rng(seed)
    n = len(pred)
    pred = pred.copy()
    U = U.astype(float).copy()
    already = already or {}
    for i, y in already.items():
        pred[i] = y
        U[i] = -1.0                      # known-correct, always below any threshold
    us = np.unique(U[U >= 0])            # candidate thresholds (ascending)
    draws = rng.integers(0, n, m)
    loss = (pred[draws] != gold[draws]).astype(float)
    # Z_t(u) = loss_t * 1[U_t <= u]   (vectorised over all candidate u)
    log_cap = np.zeros(len(us))
    s = np.zeros(len(us)); ss = np.zeros(len(us))
    lam_max = c / (1 - eps)
    for t in range(m):
        z = loss[t] * (U[draws[t]] <= us)
        mu = (eps / 2 + s) / (t + 1)               # regularised running mean (predictable)
        var = (0.25 + ss) / (t + 1)
        gap = eps - mu
        lam = np.clip(np.where(gap > 0, gap / (var + gap ** 2), 0.0), 0.0, lam_max)
        log_cap += np.log1p(lam * (eps - z))
        ss += (z - mu) ** 2
        s += z
    certified = log_cap >= np.log(1 / alpha)        # confident L^u < eps
    fail = np.where(~certified)[0]
    u_hat = us[fail[0]] if len(fail) else np.inf    # min{U_i : bound > eps}
    final = pred.copy()
    to_label = np.where(U >= u_hat)[0]
    final[to_label] = gold[to_label]
    drawn = np.unique(draws)
    final[drawn] = gold[drawn]                      # step 7: use collected labels
    new_labels = set(to_label.tolist()) | set(drawn.tolist())
    new_labels -= set(already)
    calibration_new = set(drawn.tolist()) - set(already)
    return final, {"expert_new": len(new_labels), "u_hat": float(u_hat),
                   "calibration_new": len(calibration_new),
                   "threshold_new": len(new_labels - calibration_new),
                   "error": float((final != gold).mean())}
