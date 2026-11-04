# Data, upstream material, and release notice

This repository includes **cached annotation tables and text** originating from AG News, Yelp Review Full, DBpedia-14, IMDb, Emotion, and BANKING77. Original dataset/collection metadata accompany each data file. The project-owned source code used GPL-3.0 licensing in the prior public repository; that license does **not** replace the original datasets' licenses or model providers' terms. Some generated files may include content subject to third-party copyright or dataset redistribution conditions.

Before redistributing this material, check each upstream dataset license and appropriate attribution, whether raw text/annotations may be redistributed, and any model-output redistribution restrictions. If redistribution is disallowed, replace the affected raw-data file with a lawful download/preparation script and a SHA-256 gate, **without claiming it is identical until that substitution is tested**.

The supplied original source pool IDs, annotation provenance and hashes are recorded in `study/frozen_v8_1/input_provenance.json` and the frozen per-pool metadata. `historical_support/` (repository root) contains the earlier v7.9 code, data and results, included to keep the historical supporting analyses traceable. No API keys, private model weights, or Hugging Face access tokens are intentionally packaged.

**Important scientific disclosure:** Publication readiness of licensing and redistributability has not been certified by this build. The scientific computing artifacts are included for reproduction; the owner remains responsible for rights checks before making them public.
