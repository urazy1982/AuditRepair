# Dataset and model-output redistribution notes

This repository **does not certify permission to redistribute every included third-party artifact.** Users and redistributors should check the following.

| Material | Supplied in this repository | What to check |
|---|---|---|
| AG News, Yelp Review Full, DBpedia-14, IMDb, Emotion, BANKING77 text and source labels | Frozen CSVs, text worklists and joined E7 panels | Check each original source license, citation/attribution and redistribution constraints. |
| Model-generated source/annotation responses | Cached source labels and E7 original + amended response JSONL | Verify applicable model output terms and any disclosure obligations. |
| This project's original code | GPL-3.0 license file at root | GPL terms apply only to material owned/licensed accordingly, not to the third-party data and styles. |
| Open-weight model weights | **Not included** | Reproduction of GPU collection needs obtaining checkpoints under model-provider terms. |
| API tokens / credentials | Not included | None are required to verify or reproduce the CPU results. |

If any item cannot be redistributed in a given context, replace it with an acquisition/preparation step and checksum verification, and revalidate before claiming identical reproduction.

Other cautions:
- These are public dataset benchmark labels used as *simulated* expert truth, not an error-free independently adjudicated ground truth.
- There are no new human participant annotations in the release.
- Original first-attempt E7 Mistral/Phi responses are kept to document why RC2 was needed, not to treat them as qualified labels.
- See `AuditRepair-v8.3/docs/DATA_AND_LICENSES.md` for per-dataset notes.
