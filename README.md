# AuditRepair — certified group repair of LLM labels and expert-review cost

This repository holds the code, frozen inputs, result records and verification scripts for a study of **when certified repair of LLM-generated labels reduces the number of expert labels a threshold certifier (PAC Labels) has to buy**. It covers experiments E1–E8:

| Block | Question | Records |
|---|---|---:|
| E1 | How much would oracle corrections save, and what do the shortlisted group repairs realise? | 12,600 |
| E2/E3 | Do other candidate generators or uncertainty scores change the cost effect of repair? | 26,600 |
| E4 | Synthetic intervention on the alignment between scores and source disagreement | 3,000 |
| E5 | Repair with and without rescoring the repaired predictions (A/B/C) | 8,400 |
| E6 | Synthetic positive controls: 54 cells × 100 populations, three ranking policies | 81,000 |
| E7 | A four-family annotation panel (Qwen, Llama, Mistral, Phi) on the same AG News items | 10,000 PAC arms |
| E8 | Bounded exact/high-precision reference checks of the betting and PAC arithmetic | 585 + 36 cases |

These are paired evaluation records on fixed annotation panels, not independent datasets; see [`docs/STATISTICAL_CONTRACTS.md`](docs/STATISTICAL_CONTRACTS.md) and [`docs/PROVENANCE_AND_LIMITS.md`](docs/PROVENANCE_AND_LIMITS.md).

**[Reproduce the results](REPRODUCE.md)** · **[Data and redistribution notes](docs/DATA_AND_RELEASE_RIGHTS.md)** · **[Environments](docs/WORKSPACE_AND_ENVIRONMENTS.md)**

## Quick verification

With Python 3.12 and the pinned packages in `AuditRepair-v8.3/requirements.txt`, from the repository root:

```bash
python -m pip install -r AuditRepair-v8.3/requirements.txt
bash VERIFY_COMPLETE.sh --quick   # hashes, study grids, result accounting, E7 identities (seconds)
bash VERIFY_COMPLETE.sh --full    # adds source tests, regression tests and the E8 arithmetic audit
```

Verification checks the supplied files and the accounting of the stored results. It does not rerun the experiments; [REPRODUCE.md](REPRODUCE.md) explains how to recompute each block into a new directory. Neither replaces a mathematical proof of the guarantees.

## Repository layout

| Need | Path |
|---|---|
| Main E1–E4 study: code, input CSVs, TF-IDF/SVD caches, 320 checkpoints, 42,200 records | [`AuditRepair-v8.3/study/frozen_v8_1/`](AuditRepair-v8.3/study/frozen_v8_1/) |
| Corrected exact-integer acceptance gate, acceptance replay, active-repair analysis, power study | [`AuditRepair-v8.3/study/`](AuditRepair-v8.3/study/) |
| Cross-workstation (WSL) reproduction evidence | [`AuditRepair-v8.3/validation/`](AuditRepair-v8.3/validation/) |
| E5 rescoring, E6 synthetic controls, E8 arithmetic audit, E7 gold-blind worklists | [`AuditRepair-v8.4/`](AuditRepair-v8.4/) |
| E7 collected responses and qualified joined panels | [`e7_collected/`](e7_collected/), [`e7_collected_rc2/`](e7_collected_rc2/) |
| E7 analysis results and checkpoints | [`results/`](results/) |
| E7 analysis code and frozen tasks | [`code/`](code/), [`e7_rc2/`](e7_rc2/), [`audit/`](audit/) |
| E7 GPU collection launchers | [`E7_launch_kit/`](E7_launch_kit/), [`E7_v2_amendment_kit/`](E7_v2_amendment_kit/) |
| Earlier v7.9 code, data and results (historical supporting analyses) | [`historical_support/`](historical_support/) |
| Statistical contracts, provenance, data rights, environments | [`docs/`](docs/) |

The versioned folder names (`v8.3`, `v8.4`, `frozen_v8_1`) are provenance identifiers. They are kept unchanged so that archived protocol, checkpoint and code hashes still point at the right files. Frozen directories should be treated as read-only; every recompute script writes to a new output directory.

## Main findings, with their scope

- **Opportunity vs. realisation (E1).** On six of seven fixed panels, correcting every error for free would save expert labels worth 40–69% of the pool, yet the shortlisted group repairs realise none of it.
- **Factorial (E2/E3).** In 32 of 42 cells no repair was accepted, in seven accepted repairs never changed cost, and in three cost changed only occasionally. Of 620 accepted-state comparisons, 557 cost the same, 33 saved labels and 30 added labels. The uncertainty score moved pipeline cost far more than repair did.
- **Coupling (E4).** Aligning scores more closely with source disagreement *increased* savings in the informative synthetic settings, contrary to a simple coupling explanation.
- **Rescoring (E5).** Recomputing scores after repair moves cost in both directions; no stable holdout gain.
- **Positive controls (E6).** A certifiable beneficial group can save hundreds of labels or none, depending on where its scores sit and how the audit budget is shared.
- **Second panel (E7).** A four-family panel on the same AG News items yields more accepted repairs but no practically important holdout saving. The Mistral/Phi free-format collection failed its pre-registered format gate; the constrained recollection and the failed attempt are both kept.
- **Arithmetic (E8).** Independent Fraction/Decimal references match on a bounded set of cases; this is not a general finite-precision proof.

All results are conditional on the fixed panels, the tested generators and scores, and one uniform-calibration certifier. Benchmark labels stand in for expert labels.

## Frozen identities

- Main-study protocol: `AuditRepair-v8.3/study/frozen_v8_1/protocol_v8_1.json`, SHA-256 `2931a624b1ce970ce15ce2a6d4205b3992aa42606fc4d972fb5a3db38b042f61`.
- Amendment protocol: `AuditRepair-v8.3/study/protocol_v8_2.json`, SHA-256 `6695e33de9eb88ccf74726d3015bc7cc4b2baf650cdb4cb1b2696f3c04f53516`.
- E7-RC2 analysis protocol: `results/E7_RC2_ANALYSIS_PROTOCOL.json` (hash in `results/E7_RC2_ANALYSIS_PROTOCOL.sha256`).

The protocol files are kept byte-identical, so they still name source documents that are not part of this repository (an earlier planning document and the original v7.9 archive). The v7.9 code, data and results are included as `historical_support/AuditRepair_v7_9_code_data_results.zip`; its contents are listed with SHA-256 sums in the archive's own `SHA256SUMS`. The original archive is available from the author on request.

## Requirements

- CPU verification and recomputation: Python 3.12 with `AuditRepair-v8.3/requirements.txt` (NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8). No GPU, network or model weights are needed.
- Optional new annotation collection: a CUDA GPU, a pinned vLLM container and access to the pinned open-weight checkpoints (see `docs/E7_COLLECTION_AND_GOLD_ISOLATION.md`). Model weights are not distributed.

## License and data

Project code is under GPL-3.0 (`LICENSE`). The benchmark texts and labels (AG News, Yelp Review Full, DBpedia-14, IMDb, Emotion, BANKING77) and the stored model outputs keep their own terms; see [`docs/DATA_AND_RELEASE_RIGHTS.md`](docs/DATA_AND_RELEASE_RIGHTS.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
