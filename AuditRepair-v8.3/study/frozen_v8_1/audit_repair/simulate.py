"""Simulated LLM annotators with correlated, source-concentrated errors.

Each source (an LLM x prompt configuration) has a base accuracy with uniform
random errors. On top, "systematic confusions" flip a hidden slice of class a
to class b for a chosen SET of sources at once -> correlated errors that
majority vote and Dawid-Skene tend to absorb as the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Confusion:
    true_cls: int
    wrong_cls: int
    slice_frac: float      # fraction of true_cls items in the hidden slice
    sources: tuple         # sources that share this confusion (correlation)
    flip_prob: float = 0.85


def simulate(n=5000, K=4, base_acc=(0.88, 0.85, 0.82, 0.86), confusions=None, seed=0):
    rng = np.random.default_rng(seed)
    m = len(base_acc)
    gold = rng.integers(0, K, n)
    L = np.empty((n, m), int)
    for j, acc in enumerate(base_acc):
        correct = rng.random(n) < acc
        noise = (gold + rng.integers(1, K, n)) % K
        L[:, j] = np.where(correct, gold, noise)
    if confusions is None:
        confusions = default_confusions()
    for cf in confusions:
        members = np.where(gold == cf.true_cls)[0]
        sl = members[rng.random(len(members)) < cf.slice_frac]
        for j in cf.sources:
            flip = sl[rng.random(len(sl)) < cf.flip_prob]
            L[flip, j] = cf.wrong_cls
    return L, gold


def default_confusions():
    return [
        Confusion(0, 1, 0.35, (0, 1, 2)),   # 3 of 4 sources share it: majority is wrong
        Confusion(2, 3, 0.30, (1, 3)),      # 2 of 4 share it: ties
        Confusion(3, 0, 0.25, (2,)),        # one source only: aggregation fixes it
    ]


SCENARIOS = {
    # clear, source-concentrated systematic errors
    "clear": dict(confusions=default_confusions()),
    # null: no systematic errors at all -> every group repair is harmful
    "null": dict(confusions=[]),
    # borderline: correlated confusions that are weaker and mixed with noisier sources,
    # so several candidate groups have net benefit close to zero
    "borderline": dict(base_acc=(0.80, 0.78, 0.75, 0.79), confusions=[
        Confusion(0, 1, 0.12, (0, 1, 2), flip_prob=0.7),
        Confusion(2, 3, 0.15, (1, 3), flip_prob=0.7),
        Confusion(1, 2, 0.10, (0, 2), flip_prob=0.6),
    ]),
}
