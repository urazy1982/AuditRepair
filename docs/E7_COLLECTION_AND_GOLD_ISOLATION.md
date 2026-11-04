# E7 original / RC2 GPU collection and gold-blind gates

## Frozen evidence in this release

1. `AuditRepair-v8.4/E7_tasks` is the original 30,400-request, gold-blind four-family worklist, with eight input hashes and model revisions.
2. `e7_collected/` preserves all original response files and their provenance; original pre-gold gate **failed** for Mistral/Phi, although Qwen/Llama passed.
3. `E7_v2_amendment_kit/` contains the versioned, constrained class-name recollection code. It collects another **15,200** Mistral/Phi responses and does not alter Qwen/Llama evidence.
4. `e7_collected_rc2/` preserves RC2 response files, server evidence, eight passing pre-gold gates and two joined panels.
5. `e7_rc2/E7_RC2_MANIFEST.json`, `results/E7_RC2_ANALYSIS_PROTOCOL.json` and the data audit establish the protocol and file hashes. `VERIFY_E7_RC2.sh` checks all supplied evidence without serving models.

The RC2 output change is **not** an identical prompt/response-contract replication. The same examples, labels and model family assignments are used. Gold labels are accessed only after pre-gold format/coverage checks pass; joining is done by canonical item identifier and frozen text.

## Optional local GPU collection

The cached results are adequate for CPU replication. A fresh GPU collection is a **new run** and should use a separate output folder and provenance record:

```bash
# With pinned vLLM Docker image digest, model checkpoint revisions and sufficient GPU capacity:
export VLLM_IMAGE='vllm/vllm-openai@sha256:ACTUAL_VALIDATED_DIGEST'
export E7_HOST_PORT=18080
# First-attempt launcher: use E7_launch_kit/run_one_family.sh <family> one by one;
# stop an E7 container only after collection succeeds.
# The original format gate is expected to be sensitive for Mistral/Phi.
# RC2 amended protocol (requires original Qwen/Llama response files):
bash E7_v2_amendment_kit/run_all.sh
```

Never use that digest placeholder literally. The launch kit checks the immutable model revisions, image provenance, readiness and request identities; **a served model name alone is not revision proof**. If the host port is occupied, choose a free port and ensure the collector and Docker mapping agree. Do not alter the qualified cached response records in place.

When a new collection is generated, run the pre-gold validation and compare with the frozen protocol before analysis. Report any response-contract changes, failed gates, and how a qualified panel differs from the previous one.
