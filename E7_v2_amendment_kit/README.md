# E7-RC2 — gold-blind response-contract amendment

**Scientific state:** E7 initial 30,400-response collection failed the preset 5% per-family and per-pool format gate for Mistral and Phi. The original failed collection must remain unchanged and reported. This kit collects **15,200 new Mistral/Phi responses** using constrained class-name output. Qwen and Llama (7,600 each) are retained. The new response protocol changes collection conditions; E7-RC2 is a **new versioned panel**, not strict reproduction of the original prompt contract.

## Requirements

- Freshly extracted `AuditRepair_v8_5_GitHub_Compact.zip`, with `AuditRepair-v8.3/` and `AuditRepair-v8.4/` present.
- The originally collected `e7_collected/` directory including Qwen and Llama response files. **Do not overwrite it.** If needed, extract `E7_collected_results.zip` to restore it.
- Working WSL Ubuntu 24.04, Docker GPU, a vLLM image pinned by full sha256 repository digest, model revisions frozen in the original worklists.
- `python3` (stdlib for preparing and collecting); `pandas` required for joining after the pre-gold gate.

Set up environment once (inside WSL):

```bash
cd /mnt/c/AuditRepair_v8_5_GitHub
sudo apt-get update
sudo apt-get install -y python3-venv
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/auditrepair-e7rc2"
source "$HOME/.venvs/auditrepair-e7rc2/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r AuditRepair-v8.3/requirements.txt
```

The launcher automatically reuses the *pinned* vLLM image digest recorded in each family’s original `e7_collected/provenance/*_start.json`. If needed you can override `VLLM_IMAGE` with another independently verified immutable digest; do not use a floating `latest` tag.

```bash
export E7_HOST_PORT=18080
```

Then run:

```bash
bash E7_v2_amendment_kit/run_all.sh
```

Each family runs sequentially; its E7-RC2 Docker container is stopped after successful collection. `run_all.sh` stops if any stage fails. A failed server is left for inspection rather than silently replaced.

Outputs: `e7_rc2/` frozen amended worklists + their hashes, `e7_collected_rc2/` response files, per-model provenance, pre-gold validation, and (only after passing the gate) `e7_rc2_four_family_*.csv` merged panels. After validation, the script packages `E7_RC2_qualified_results.zip` for uploading here.

**No new response collection has been executed by this release kit.** Only deterministic preparation and simulated endpoint tests can be performed here; your RTX 5090 executes the real 15,200 requests.

## Methods note

- The original model families, pinned revisions, item IDs, and AG News text are retained.
- Both originally assigned prompt styles receive an explicit final class-name-only override, and constrained decoding enforces `choice=["World","Sports","Business","Sci/Tech"]`.
- The output parser accepts only one exact class label after trimming surrounding whitespace and requires `finish_reason="stop"`. There is **no fallback to an unconstrained or permissive parser**.
- The first request acts as a constrained-decoding compatibility probe; unsupported vLLM versions are rejected before the collection proceeds. vLLM v0.12+ accepts `structured_outputs.choice`; older versions using `guided_choice` are deliberately not substituted automatically.
- Response caches and amendments are separated. Request identity hashes, class vocabulary, model revision, and image provenance are retained.
- The same 5% unusable-response gate applies to *every* family × panel; it is evaluated without gold access and written to disk before the frozen benchmark labels are opened.
- A passing format gate does not establish label accuracy. Compare model accuracy and repair outcomes only after the gate, and report the original failed collection as an excluded protocol attempt.
