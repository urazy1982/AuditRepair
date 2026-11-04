# E7 four-family AG News annotation — GPU execution companion

This small companion uses the **unaltered v8.5 E7 task manifest**, with 30,400 gold-blind prompts (Qwen, Llama, Mistral, Phi × 5,000 development and 2,600 holdout). It does **not** simulate or claim new annotations. Run from WSL/Ubuntu, using a single RTX 5090, one model at a time. The original v8.5 reference and frozen data remain unchanged.

## A. Install the companion

Extract this folder as `E7_launch_kit/` inside the extracted v8.5 repository root (containing `AuditRepair-v8.3` and `AuditRepair-v8.4`). Change to the repository root in WSL. Use your existing Python venv with pandas. Docker + NVIDIA GPU access must already work (`docker run --rm --gpus all ... nvidia-smi` or `nvidia-smi` inside existing GPU containers).

## B. Pin image and model revisions

The launcher obtains exact model IDs and revisions from `E7_GOLD_BLIND_MANIFEST.json`. It passes `--revision` **and** `--tokenizer-revision` to vLLM; it will reject an unpinned image. Determine the digest of a tested vLLM Docker image and set:

```bash
export VLLM_IMAGE='vllm/vllm-openai@sha256:<FULL_IMAGE_DIGEST>'
```

Never use this placeholder literally. For example, after pulling an image tag that works with your RTX 5090, inspect its exact immutable digest with `docker image inspect <image-tag> --format '{{json .RepoDigests}}'`. Record the digest. For gated model repos, export `HF_TOKEN` privately; do not put it in scripts or uploaded archives. The launcher records container image ID, model revision and server command but not the token.

Because the original protocol did not pin this *new* inference environment yet, freeze the chosen image digest, driver/CUDA/vLLM versions and model IDs before the first substantive request. A server reporting the right model name is **not** sufficient evidence of the correct weights. Verify the container command and exact revision, and retain the `provenance/` directory.

## C. Execute sequentially

Run from the repository root. Each family uses port 8000, so remove the existing E7 container explicitly between families. Avoid using port 8000 concurrently with other vLLM projects.

```bash
bash E7_launch_kit/run_one_family.sh qwen
docker rm -f auditrepair-e7-qwen
bash E7_launch_kit/run_one_family.sh llama
docker rm -f auditrepair-e7-llama
bash E7_launch_kit/run_one_family.sh mistral
docker rm -f auditrepair-e7-mistral
bash E7_launch_kit/run_one_family.sh phi
docker rm -f auditrepair-e7-phi
bash E7_launch_kit/validate_all.sh
```

To control the output location: `export E7_OUTPUT_ROOT="$PWD/e7_collected"`. Successful collection yields eight `*_responses.jsonl` files. Each family is restartable; cached received rows are reused after a verified request-hash check. **Failures are also cached** by the original collector, so do not treat re-running a failed family as a retry of failures. Preserve the original failure records before any manual retry, and document any change to the retry policy. The strict 5% unusable/failed gate is evaluated *before* gold labels are opened; the validator refuses incomplete worklists.

## D. Package the collected evidence

After successful `validate_all.sh`, package `e7_collected/responses/`, `e7_collected/provenance/`, and `e7_collected/validated/`. Do not add HF tokens, downloaded model weights, or private caches to the archive. The next stage computes model quality and annotation disagreement and compares the panel against the original using the frozen study and matched PAC evaluation. The existing `e7_validate_and_join.py` checks response format only and then joins the fixed benchmark truth; it does not run the downstream E1–E5 analyses by itself.

## Key research limitations

The two pools share the same underlying dataset and their historical reference labels are known from previous work; the new panel is an **independent model-annotation collection**, not a newly blind held-out dataset. Alternating direct/reason prompts differs across model families by position and is not a pure family-only causal test. New models must not be silently substituted after observing results. References: v8.5 manifest, vLLM `serve --revision` docs.
