"""PAC Labels threshold-reduced exact-sequence evaluator.

Algebraic optimization: Z_t(u) = loss_t 1[U_draw[t] <= u] can change with u
only at an observed error-draw score. Between breakpoints every t's observed
loss sequence is identical; the predictable stake and terminal capital are
identical. Test the first candidate threshold of each equivalence class.
All arithmetic and boundary conventions otherwise replicate frozen pac_labels.
"""
from __future__ import annotations
import numpy as np


def pac_labels_fast(pred, U, gold, eps=.05, alpha=.1, m=500, c=.5, already=None, seed=0):
    rng=np.random.default_rng(seed)
    pred=np.asarray(pred).copy(); U=np.asarray(U,dtype=float).copy()
    n=len(pred);gold=np.asarray(gold)
    already=already or {}
    for i,y in already.items():
        pred[i]=y;U[i]=-1.0
    us=np.unique(U[U>=0])
    draws=rng.integers(0,n,m)
    loss=(pred[draws]!=gold[draws]).astype(float)
    # Only these candidate scores produce a new sample-loss sequence.
    if len(us):
        observed_indices=np.searchsorted(us,U[draws[loss>0]],side='left')
        observed_indices=observed_indices[observed_indices<len(us)]
        idx=np.unique(np.r_[0,observed_indices])
        chosen_us=us[idx]
    else:
        chosen_us=us
    logcap=np.zeros(len(chosen_us));s=np.zeros(len(chosen_us));ss=np.zeros(len(chosen_us))
    lam_max=c/(1-eps)
    for t in range(m):
        z=loss[t]*(U[draws[t]]<=chosen_us)
        mu=(eps/2+s)/(t+1)
        var=(.25+ss)/(t+1)
        gap=eps-mu
        lam=np.clip(np.where(gap>0,gap/(var+gap**2),0.0),0.0,lam_max)
        logcap+=np.log1p(lam*(eps-z))
        ss+=(z-mu)**2
        s+=z
    certified=logcap>=np.log(1/alpha)
    failing=np.where(~certified)[0]
    uhat=float(chosen_us[failing[0]]) if len(failing) else float('inf')
    final=pred.copy()
    reviewed=np.flatnonzero(U>=uhat)
    final[reviewed]=gold[reviewed]
    drawn=np.unique(draws)
    final[drawn]=gold[drawn]
    new=set(int(i) for i in reviewed)|set(int(i) for i in drawn)
    new.difference_update(already)
    cal=set(int(i) for i in drawn)-set(already)
    return final,{'expert_new':len(new),'u_hat':uhat,'calibration_new':len(cal),'threshold_new':len(new-cal),
                  'error':float((final!=gold).mean())}
