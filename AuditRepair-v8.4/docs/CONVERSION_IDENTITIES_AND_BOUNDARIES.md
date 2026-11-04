# Exact conversion accounting and its scope (v8.4 explanatory note)

These are reusable, implementation-independent algebraic observations for a **threshold-local, fixed-score** certifier. They are not presented as foundational new risk-control theorems.

Let `A` be the set of already purchased audits, `D` the *distinct* calibration draw IDs, `U` the fixed score vector, and `u0,u1` the thresholds returned on baseline and repaired predictions. Review sets are `R_k={i: U_i >= u_k}`, including equal-score boundary items. The total expert cost is `C_k=|A ∪ D ∪ R_k|`. Then exactly

`C0 - C1 = sum_{i not in A∪D} (1{U_i>=u0} - 1{U_i>=u1})`.

**Proof:** Outside `A∪D`, an item is purchased if and only if its score exceeds or equals the selected review threshold. Items already in `A∪D` are paid for in both arms and cancel. The identity is set algebra and holds even with ties, different thresholds, or a threshold of infinity. It holds for E5 A/B with fixed scores. For E5 A/C, score vectors differ; replace the right-hand expression with `1{U0_i>=u0}-1{Ur_i>=ur}` instead, and do not interpret it as an isolated label effect.

Let `M` denote the **fixed** set of k changed prediction indices (not scores), chosen before and independently of calibration sampling. Both arms have the same `U,A` and use the same m independent uniform draws with replacement from N items. If none of the draws intersects `M`, all sampled loss sequences at *every* candidate u are identical; under threshold-local predictable testing the selected threshold and cost are identical. Consequently,

`P(C0 != C1 | fixed pre-calibration state) <= 1-(1-k/N)^m`.

The bound is potentially loose; it does not count where changed items fall in uncertainty space or whether a sampled change actually modifies a loss, and it says nothing about arm C with recomputed scores. A stronger diagnostic records whether a sampled changed label influences any tested prefix and whether the selected threshold or unique review union changes.

Both statements rely on *frozen candidate selection and scores before the PAC draws*. They are not licences to use gold information or a threshold calculated from the same future draws when choosing repairs. The PAC error guarantee remains a separate statistical statement requiring the assumptions of the selected terminal-capital test; these identities concern purchases, not coverage.

## Stage-specific estimands

- **Candidate access:** proportion of independently randomized preprocessing states with at least one retrospective positive-net frozen candidate. This uses gold only **after** freezing and is not deployable prediction.
- **Certification:** proportion of preprocessing states/groups whose audit procedure accepts under the current per-group level and shared budget, with failed-audit expenditure counted.
- **Conversion:** probability of a paired cost change and its signed magnitude, conditional on a fixed panel and a certified action, plus the unconditional expected saving including inactive states.

For E5 and E6, seed-level summaries are the primary inferential units. Nested PAC draws provide a conditional distribution of certification cost given a preprocessing state; they do not multiply the number of independent discovered repair opportunities. The current CI estimates remain conditional on the fixed annotation panels or the specified synthetic population generator, respectively.
