# Version lineage, provenance, and residual limitations

| Stage | Reused data or new evidence | Current executable location | Scope / frozen status |
|---|---|---|---|
| v8.1 E1–E4 | Cached seven original annotation panels | `AuditRepair-v8.3/study/frozen_v8_1/` | 320 checkpoints / 42,200 rows; names preserved for exact hashes. |
| v8.2 amendment | Correction to strict acceptance, active repair and operational power | `AuditRepair-v8.3/study/` | 55 recorded accepted decisions retained; prior WSL numeric reproduction available in validation summaries. |
| v8.3 | GitHub packaging of prior experiments | `AuditRepair-v8.3/` | Integration release, not new data. |
| v8.4 E5/E6/E8 | Cached-panel joint rescoring; fresh synthetic populations; bounded numerical tests | `AuditRepair-v8.4/` | E5 8,400; E6 81,000; E8 585/36; no new LLM annotations at this stage. |
| v8.5 | Compacted distribution | `COMPACTION_MANIFEST.json` | Duplicate WSL checkpoints and E6 synthetic checkpoint shards intentionally removed; canonical results retained. |
| E7 original + RC2 | New response panel on existing AG News examples; failed initial gate and amended qualified panel | `e7_collected/`, `e7_collected_rc2/`, `e7_rc2/`, `results/` | Not a controlled isolated model-family experiment; same benchmark items with mixed response contracts. |
| Complete GitHub distribution | Preserves source and raw evidence under one root | This repository | Repackaging only. |

**Authoritative parent archive hashes and source/evidence mappings** are recorded in `AuditRepair-v8.3/study/frozen_v8_1/protocol_v8_1.json`, `AuditRepair-v8.4/results/E6_full/e6_manifest.json` and E7-RC2 manifests. This distribution's own `RELEASE_SHA256SUMS` covers every included file and is not a replacement for internal experiment hashes.

## Remaining scientific limitations

- Conditional results from fixed annotated panels; benchmark labels proxy the expert gold standard.
- E7 same-data response panel and changed format contracts do not identify causal model diversity.
- Original E5 uses only 20 preprocessing seeds; rare event conclusions are limited, and zero-interval bootstrap widths do not imply true zero expected effects.
- E6 is controlled synthetic evidence, not a real-world estimate of candidate ranking effectiveness.
- E8 independent Fraction/Decimal checks are finite and do not replace a proof that all finite-precision executions preserve theoretical guarantees.
- Models, Docker images, and any external private credentials are not redistributed.
- Public release requires rights/attribution checks for each third-party source.
