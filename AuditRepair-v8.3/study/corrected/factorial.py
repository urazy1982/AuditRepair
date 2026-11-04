"""Frozen, CPU-only E2/E3 common-control factorial preprocessing.

Public API
----------
``build_features(texts, ids) -> (X, neighbors)``
    Fit the prespecified TF-IDF/SVD representation without labels. ``X`` and
    ``neighbors`` retain input row order. Neighbours exclude the query itself;
    exact distance ties use the lexicographic order of ``str(id)``. Build these
    once per pool, and permute/remap them with the pool if its rows are reordered.

``preprocess_generator(L, K, gold, features, neighbors, generator, seed,
                       warm_indices=None) -> dict``
    Run G0_uniform, G1, or G2 with a common 120-item uniform warm-up, a total
    budget of 400 unique labels, and at most 196 additional group-stage audits.
    ``seed`` can be an integer or SeedSequence; reusing it does not mutate it.
    Explicit ``warm_indices`` must contain exactly 120 distinct indices. They
    let the orchestrator provide the same externally sampled warm-up to all
    generators. No PAC sampling or real-data evaluation is performed here.

Important returned fields
-------------------------
``base``, ``post``
    DS predictions and posterior fitted on exactly ``audits``.
``pred``
    ``base`` with certified replacements on unaudited group remainders.
``ref``, ``post_ref``, ``initial_ref``
    The frozen post-warm-up reference/posterior, and the pre-warm-up DS labels.
``scores['S0']``, ``scores['S1']``
    One read-only score vector per source, reused unchanged for ``base`` and
    ``pred``. S0 = 1 - max(post). S1 = 1 - q(base | text), where q is trained
    only on warm-up gold labels. These arrays do not apply the PAC audited-item
    sentinel; pass ``audits`` to PAC so it can account for already known labels.
``audits``, ``warm_indices``, ``warm_audits``, ``text_probabilities``
    Charged labels, warm-up order/labels, and the text-only class probabilities.
``groups``, ``group_log``
    All frozen shortlisted hypotheses and the actually tested hypotheses.
    Group members exclude warm-up items and are pairwise disjoint. Every log
    records its audit history, stopping-time remainder, and certificate bound.
    Gold-dependent diagnostics are added only after the audit trajectory ends.
``candidate_labels``, ``oracle_candidate_labels``
    The first applies every shortlisted proposal to its final unaudited
    remainder. The second applies only proposals with positive net gain over
    ``base`` on that remainder. This is a free *group-action* oracle, not an
    oracle correcting each incorrect item individually. Neither steers audits.
``diagnostics``
    Budget, coverage, and gain counts. Nonbeneficial/harmful accepted groups
    relative to the FROZEN reference are separate from effects relative to
    same-audit DS. The certificate guarantees the former comparison only.

This design tests sensitivity to architecture, not statistical independence or
a uniquely identified causal coupling effect. The vendored Oracle, WorLowerCS,
DS fit, and uncertainty_fill implementations retain the frozen v8.1 rules.
Only the final strict remainder-acceptance arithmetic and its logging change in
v8.2; the betting sequence, audit policy and fixed score estimand are preserved.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

if __package__:
    from .audit_repair.core import (
        Oracle, WorLowerCS, candidate_groups_disjoint, dawid_skene,
        uncertainty_fill,
    )
else:
    from audit_repair.core import (
        Oracle, WorLowerCS, candidate_groups_disjoint, dawid_skene,
        uncertainty_fill,
    )


AUDIT_BUDGET = 400
WARMUP_AUDITS = 120
GROUP_AUDITS = 196
MIN_GROUP_SIZE = 30
MAX_GROUPS = 10
ALPHA_REPAIR = 0.05
MAX_GROUP_DRAWS = 80
FUTILITY_AFTER = 10
FUTILITY_MEAN = 0.45
STOP_AFTER_FAILURES = 10
NEIGHBORS = 15
FEATURE_SEED = 20261030
CLASS_SMOOTHING = 1e-6


def build_features(
    texts: Sequence[str], ids: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the prespecified label-free representation and stable neighbours.

    Degenerate vocabularies fail explicitly rather than silently changing
    min_df or the frozen representation. Individual zero-feature rows remain
    valid zero vectors; their distance ties use IDs. Cosine distances are
    computed in bounded-size blocks; no full n-by-n matrix is retained.
    """
    texts = list(texts)
    ids = [str(item) for item in ids]
    n = len(texts)
    if n != len(ids) or n < 2:
        raise ValueError("texts and ids must have the same length, at least two")
    if len(set(ids)) != n:
        raise ValueError("ids must be unique after conversion to strings")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("every text must be a string")

    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_features=20000, sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    dimension = min(128, n - 1, tfidf.shape[1] - 1)
    if dimension < 1:
        raise ValueError("frozen SVD requires at least two retained text features")
    svd = TruncatedSVD(n_components=dimension, random_state=FEATURE_SEED)
    features = normalize(svd.fit_transform(tfidf), norm="l2")
    if not np.isfinite(features).all():
        raise ValueError("nonfinite values in the frozen text representation")

    id_order = np.argsort(np.asarray(ids), kind="stable")
    id_rank = np.empty(n, dtype=np.int64)
    id_rank[id_order] = np.arange(n)
    k = min(NEIGHBORS, n - 1)
    neighbors = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, 256):
        block = cosine_distances(features[start:start + 256], features)
        for offset, distance in enumerate(block):
            i = start + offset
            distance[i] = np.inf
            neighbors[i] = np.lexsort((id_rank, distance))[:k]
    return np.asarray(features, dtype=np.float64), neighbors


def _seed_streams(seed: Any) -> tuple[np.random.Generator, np.random.Generator]:
    if isinstance(seed, np.random.SeedSequence):
        # SeedSequence.spawn mutates its child counter. Reconstruct so the same
        # caller-owned object gives an identical warm-up across generators.
        sequence = np.random.SeedSequence(
            seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size,
        )
    elif seed is None or isinstance(seed, np.random.Generator):
        raise ValueError("seed must be an explicit seed or SeedSequence")
    else:
        sequence = np.random.SeedSequence(seed)
    warm_seed, group_seed = sequence.spawn(2)
    return np.random.default_rng(warm_seed), np.random.default_rng(group_seed)


def _text_probabilities(
    features: np.ndarray, warm: np.ndarray, labels: np.ndarray, K: int
) -> tuple[np.ndarray, dict[str, Any]]:
    observed = np.unique(labels)
    q = np.zeros((len(features), K), dtype=float)
    if len(observed) == 1:
        q[:, int(observed[0])] = 1.0
        info = {"observed_classes": observed.tolist(), "n_iter": [],
                "one_class_fallback": True}
    else:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(features[warm], labels)
        except ConvergenceWarning as error:
            raise RuntimeError(
                "frozen text classifier failed to converge within 1000 iterations"
            ) from error
        if np.any(model.n_iter_ >= 1000):
            raise RuntimeError(
                "frozen text classifier exhausted its 1000-iteration limit"
            )
        q[:, model.classes_.astype(int)] = model.predict_proba(features)
        info = {"observed_classes": model.classes_.astype(int).tolist(),
                "n_iter": model.n_iter_.astype(int).tolist(),
                "one_class_fallback": False}
    q = (q + CLASS_SMOOTHING) / (1.0 + K * CLASS_SMOOTHING)
    return q, info


@dataclass(frozen=True)
class _Group:
    cur: int
    alt: int
    items: np.ndarray
    source: tuple[int, ...] | str
    prior_score: float = 0.0
    confusion_count: int = 0


def _generate_groups(
    L: np.ndarray, ref: np.ndarray, post_ref: np.ndarray,
    neighbors: np.ndarray, q: np.ndarray, warm: np.ndarray,
    warm_labels: np.ndarray, initial_ref: np.ndarray, generator: str, K: int,
) -> list[_Group]:
    eligible = np.ones(len(ref), dtype=bool)
    eligible[warm] = False
    if generator == "G0_uniform":
        groups = []
        for group in candidate_groups_disjoint(L, ref, post_ref, min_size=1):
            items = group.items[eligible[group.items]].astype(np.int64)
            if len(items) >= MIN_GROUP_SIZE:
                groups.append(_Group(
                    group.cur, group.alt, items, tuple(int(s) for s in group.source),
                    float(len(items) * post_ref[items, group.alt].mean()),
                ))
        return sorted(groups, key=lambda g: (
            -len(g.items), -g.prior_score, g.cur, g.alt, g.source,
        ))

    if generator == "G1":
        alternative = np.array([
            np.bincount(ref[row], minlength=K).argmax() for row in neighbors
        ], dtype=int)
        confusion = np.zeros((K, K), dtype=int)
        source = "text_neighbors"
    else:  # G2: caller validates the generator name.
        alternative = q.argmax(axis=1)
        confusion = np.zeros((K, K), dtype=int)
        np.add.at(confusion, (initial_ref[warm], warm_labels), 1)
        source = "warmup_text_classifier"

    members: dict[tuple[int, int], list[int]] = {}
    for i in np.flatnonzero(eligible & (alternative != ref)):
        key = (int(ref[i]), int(alternative[i]))
        members.setdefault(key, []).append(int(i))
    groups = [
        _Group(c, b, np.asarray(items, dtype=np.int64), source,
               confusion_count=int(confusion[c, b]))
        for (c, b), items in members.items() if len(items) >= MIN_GROUP_SIZE
    ]
    if generator == "G1":
        return sorted(groups, key=lambda g: (-len(g.items), g.cur, g.alt))
    return sorted(groups, key=lambda g: (
        -g.confusion_count, -len(g.items), g.cur, g.alt,
    ))


def _group_record(group: _Group, rank: int) -> dict[str, Any]:
    return {
        "group_index": rank, "cur": group.cur, "alt": group.alt,
        "source": list(group.source) if isinstance(group.source, tuple) else group.source,
        "size": len(group.items), "members": group.items.tolist(),
        "membership_id": hashlib.sha256(
            np.asarray(group.items, dtype="<i8").tobytes()
        ).hexdigest(),
        "prior_score": group.prior_score,
        "warmup_confusion_count": group.confusion_count,
    }


def _gain(gold: np.ndarray, old: np.ndarray, alt: int, items: np.ndarray) -> int:
    return int(np.sum((gold[items] == alt).astype(int) -
                      (gold[items] == old[items]).astype(int)))


def _validate_inputs(L, K, gold, features, neighbors) -> None:
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    if L.ndim != 2 or len(L) < WARMUP_AUDITS or L.shape[1] < 1:
        raise ValueError("L must be a matrix with at least 120 rows and one source")
    if not np.issubdtype(L.dtype, np.integer) or np.any((L < -1) | (L >= K)):
        raise ValueError("source labels must be integers in [-1, K)")
    if gold.shape != (len(L),) or not np.issubdtype(gold.dtype, np.integer):
        raise ValueError("gold must be a row-aligned integer vector")
    if np.any((gold < 0) | (gold >= K)):
        raise ValueError("gold labels must be in [0, K)")
    if features.ndim != 2 or features.shape[0] != len(L) or features.shape[1] < 1:
        raise ValueError("features must be a row-aligned nonempty matrix")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    if neighbors.shape != (len(L), min(NEIGHBORS, len(L) - 1)):
        raise ValueError("neighbors must contain 15 nonself indices per row")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError("neighbor indices must be integers")
    if np.any((neighbors < 0) | (neighbors >= len(L))):
        raise ValueError("neighbor index is outside the pool")
    if np.any(neighbors == np.arange(len(L))[:, None]):
        raise ValueError("self-neighbors are not allowed")
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise ValueError("neighbor indices must be unique within each row")


def preprocess_generator(
    L: np.ndarray, K: int, gold: np.ndarray, features: np.ndarray,
    neighbors: np.ndarray, generator: str, seed: Any,
    warm_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return a completed, pre-PAC state under the fixed common-control policy.

    ``G0`` is accepted as an alias, but is always reported as ``G0_uniform``.
    The original stop rule is retained: acceptance is checked before futility
    and caps; the fraction cap stops at t >= N/2, i.e. ceil(N/2) for odd N.
    """
    generator = {"G0": "G0_uniform"}.get(generator, generator)
    if generator not in {"G0_uniform", "G1", "G2"}:
        raise ValueError("generator must be G0_uniform (or G0), G1, or G2")
    L, gold = np.asarray(L), np.asarray(gold)
    features, neighbors = np.asarray(features), np.asarray(neighbors)
    _validate_inputs(L, K, gold, features, neighbors)
    n = len(L)
    rng_warm, rng_groups = _seed_streams(seed)
    if warm_indices is None:
        warm = rng_warm.permutation(n)[:WARMUP_AUDITS]
    else:
        warm = np.asarray(warm_indices)
        if (warm.shape != (WARMUP_AUDITS,) or
                not np.issubdtype(warm.dtype, np.integer) or
                len(np.unique(warm)) != WARMUP_AUDITS or
                np.any((warm < 0) | (warm >= n))):
            raise ValueError("warm_indices must contain 120 distinct valid integers")
        warm = warm.astype(np.int64, copy=True)

    # All learning labels are obtained through the charged Oracle.
    initial_ref = dawid_skene(L, K).argmax(axis=1)
    oracle = Oracle(gold, AUDIT_BUDGET)
    for i in warm:
        oracle.query(int(i))
    warm_audits = dict(oracle.revealed)
    warm_labels = np.asarray([warm_audits[int(i)] for i in warm], dtype=int)
    post_ref = dawid_skene(L, K, dict(warm_audits))
    ref = post_ref.argmax(axis=1)
    ref[warm] = warm_labels
    q, classifier_info = _text_probabilities(features, warm, warm_labels, K)
    eligible_groups = _generate_groups(
        L, ref, post_ref, neighbors, q, warm, warm_labels,
        initial_ref, generator, K,
    )
    groups = eligible_groups[:MAX_GROUPS]
    alpha_group = ALPHA_REPAIR / max(len(groups), 1)
    group_records = [_group_record(g, j) for j, g in enumerate(groups)]

    # Freeze a disjoint, completely unaudited population for every hypothesis.
    all_members = [int(i) for group in groups for i in group.items]
    if len(set(all_members)) != len(all_members) or set(all_members) & set(warm_audits):
        raise AssertionError("shortlist populations must be disjoint and exclude warm-up")

    accepted: dict[int, int] = {}
    guarded_populations: set[int] = set()
    logs: list[dict[str, Any]] = []
    failures = 0
    for rank, group in enumerate(groups):
        if (len(oracle.revealed) >= WARMUP_AUDITS + GROUP_AUDITS or
                failures >= STOP_AFTER_FAILURES):
            break
        population = group.items
        if any(int(i) in oracle.revealed for i in population):
            raise AssertionError("a frozen group contains previously audited items")
        test = WorLowerCS(N=len(population), alpha=alpha_group)
        history: list[dict[str, int]] = []
        decision, stop_reason = "abstain", "population_exhausted"
        for i in rng_groups.permutation(population):
            if len(oracle.revealed) >= WARMUP_AUDITS + GROUP_AUDITS:
                stop_reason = "budget"
                break
            i = int(i)
            y = oracle.query(i)
            gain = int(y == group.alt) - int(y == ref[i])
            test.update((gain + 1) / 2)
            history.append({"item": i, "gold": int(y), "gain": gain})
            if test.remainder_is_positive():
                decision, stop_reason = "repair", "certified"
                break
            if test.t >= FUTILITY_AFTER and test.mean < FUTILITY_MEAN:
                decision, stop_reason = "reject", "futility"
                break
            if test.t >= MAX_GROUP_DRAWS or test.t >= len(population) / 2:
                stop_reason = ("audit_cap" if test.t >= MAX_GROUP_DRAWS
                               else "fraction_cap")
                break
        remainder = np.asarray([
            int(i) for i in population if int(i) not in oracle.revealed
        ], dtype=np.int64)
        guarded = decision == "repair"
        if guarded:
            guarded_populations.update(int(i) for i in population)
            accepted.update((int(i), group.alt) for i in remainder)
        failures = 0 if guarded else failures + 1
        logs.append({
            **group_records[rank], "alpha": alpha_group,
            "decision": decision, "guarded": guarded,
            "stop_reason": stop_reason, "audited": test.t,
            "known_at_start": [], "audit_history": history,
            "remainder_members_at_stop": remainder.tolist(),
            "lower_mean": float(test.lower),
            "remainder_lb": float(test.remainder_gain_lb()),
            **test.remainder_certificate_record(),
            "audit_mean_x": float(test.mean),
            "fraction_cap_draws": math.ceil(len(population) / 2),
        })

    group_audits = len(oracle.revealed) - WARMUP_AUDITS
    clamp = dict(oracle.revealed)
    post = uncertainty_fill(L, K, oracle, clamp, exclude=guarded_populations)
    audits = dict(oracle.revealed)
    base = post.argmax(axis=1)
    for i, label in audits.items():
        base[i] = label
    pred = base.copy()
    for i, label in accepted.items():
        if i in audits:
            raise AssertionError("uncertainty fill entered a guarded remainder")
        pred[i] = label

    # Scores are functions of the common same-audit baseline, never the repairs.
    scores = {
        "S0": 1.0 - post.max(axis=1),
        "S1": 1.0 - q[np.arange(n), base],
    }
    for vector in scores.values():
        vector.setflags(write=False)

    # Full-gold evaluation begins only after all adaptive choices are complete.
    candidate_labels, oracle_candidate_labels = base.copy(), base.copy()
    oracle_groups = []
    for rank, group in enumerate(groups):
        remainder = np.asarray([
            int(i) for i in group.items if int(i) not in audits
        ], dtype=np.int64)
        candidate_labels[remainder] = group.alt
        gain_vs_base = _gain(gold, base, group.alt, remainder)
        gain_vs_ref = _gain(gold, ref, group.alt, remainder)
        if gain_vs_base > 0:
            oracle_candidate_labels[remainder] = group.alt
            oracle_groups.append(rank)
        group_records[rank].update({
            "final_remainder_members": remainder.tolist(),
            "final_remainder_size": len(remainder),
            "true_population_gain_vs_ref": _gain(gold, ref, group.alt, group.items),
            "proposed_final_remainder_gain_vs_ref": gain_vs_ref,
            "proposed_final_remainder_gain_vs_same_audit": gain_vs_base,
        })
    for log in logs:
        remainder = np.asarray(log["remainder_members_at_stop"], dtype=np.int64)
        log["true_remainder_gain_vs_frozen_ref"] = _gain(
            gold, ref, log["alt"], remainder,
        )
        log["true_remainder_gain"] = log["true_remainder_gain_vs_frozen_ref"]
        log["effect_vs_same_audit"] = (
            _gain(gold, base, log["alt"], remainder) if log["guarded"] else 0
        )
        log["nonbeneficial_accepted_vs_ref"] = bool(
            log["guarded"] and log["true_remainder_gain_vs_frozen_ref"] <= 0
        )
        log["negative_accepted_vs_ref"] = bool(
            log["guarded"] and log["true_remainder_gain_vs_frozen_ref"] < 0
        )
        log["negative_effect_vs_same_audit"] = bool(
            log["guarded"] and log["effect_vs_same_audit"] < 0
        )

    applied = np.asarray(sorted(accepted), dtype=np.int64)
    c2i_ref = int(np.sum((ref[applied] == gold[applied]) & (pred[applied] != gold[applied])))
    i2c_ref = int(np.sum((ref[applied] != gold[applied]) & (pred[applied] == gold[applied])))
    c2i_base = int(np.sum((base == gold) & (pred != gold)))
    i2c_base = int(np.sum((base != gold) & (pred == gold)))
    diagnostics = {
        "audit_budget": AUDIT_BUDGET, "audited": len(audits),
        "warmup_audits": WARMUP_AUDITS, "group_phase_audits": group_audits,
        "fill_audits": len(audits) - WARMUP_AUDITS - group_audits,
        "unused_budget": AUDIT_BUDGET - len(audits),
        "eligible_groups": len(eligible_groups),
        "eligible_items": sum(len(g.items) for g in eligible_groups),
        "shortlisted_groups": len(groups), "shortlisted_items": len(all_members),
        "tested_groups": len(logs), "certified_groups": sum(g["guarded"] for g in logs),
        "beneficial_shortlisted_groups_vs_ref": sum(
            g["true_population_gain_vs_ref"] > 0 for g in group_records
        ),
        "oracle_beneficial_groups_vs_same_audit": len(oracle_groups),
        "harmful_accepted_vs_ref": sum(g["nonbeneficial_accepted_vs_ref"] for g in logs),
        "negative_accepted_vs_ref": sum(g["negative_accepted_vs_ref"] for g in logs),
        "negative_accepted_effect_vs_same_audit": sum(
            g["negative_effect_vs_same_audit"] for g in logs
        ),
        "repaired_unaudited_items": len(accepted),
        "changed_items_vs_same_audit": int(np.sum(pred != base)),
        "i2c_vs_ref": i2c_ref, "c2i_vs_ref": c2i_ref,
        "net_gain_vs_ref": i2c_ref - c2i_ref,
        "i2c_vs_same_audit": i2c_base, "c2i_vs_same_audit": c2i_base,
        "net_gain_vs_same_audit": i2c_base - c2i_base,
    }
    return {
        "generator": generator, "pred": pred, "base": base, "post": post,
        "audits": audits, "ref": ref, "post_ref": post_ref,
        "initial_ref": initial_ref, "warm_indices": warm.copy(),
        "warm_audits": warm_audits, "text_probabilities": q,
        "classifier_info": classifier_info, "scores": scores,
        "groups": group_records, "group_log": logs,
        "candidate_labels": candidate_labels,
        "oracle_candidate_labels": oracle_candidate_labels,
        "oracle_candidate_groups": oracle_groups,
        "repair_labels": dict(accepted),
        "guarded_populations": sorted(guarded_populations),
        "diagnostics": diagnostics,
    }
