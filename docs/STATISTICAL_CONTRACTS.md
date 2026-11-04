# Probability, comparisons and implementation contracts

- **Remainder correction:** for group size N, grid index j on a 401-point grid and audited gain sum S, strict acceptance uses the exact integer sign `N*(2*j-400)-400*S > 0` and requires at least one unaudited item. The independently audited floating-point betting process is not an interval-arithmetic implementation.
- **Group rejection:** sticky first-crossing of grid-null capital boundaries, enabling a time-uniform lower bound at arbitrary stopping times. Group tests use the frozen shortlist and per-group alpha `alpha_repair/g`.
- **Final PAC:** terminal (after all m draws) betting-capital boundary, not a sticky maximum-over-time test; candidate threshold equality is consequential. Automatic label retention uses `U < u_hat`, and reviewed items satisfy `U >= u_hat`; test losses include `U <= u`.
- **End-point:** total number of *distinct* purchased benchmark labels `|A union D union R|`. This proxies annotation volume; it is not money or expert-hours.
- **Guarantee scope:** with assumptions, final error target epsilon=0.03 has `alpha_final=0.10`; no harmful group acceptance relative to frozen reference has `alpha_repair=0.05`; their conjunction has the union-bound lower probability 0.85. Neither is a claim about the conditional probability of a harmful repair *given that* one is accepted.
- **Comparators:** group certificate improvement is against frozen warm-up labels, not necessarily later DS refitted on all audits; the cost comparator is later same-audit DS. Keep these distinct.
- **Inference:** primary comparison is at the preprocessing seed level; 10 nested PAC repeats do not create 10 independent preprocessing events. Pointwise 95% seed bootstrap intervals conditional on fixed annotation panels and reported probabilities are descriptive unless justified by the frozen sampling design.
- **New panel:** E7 label collection was quality-gated *without benchmark gold labels*; two families were recollected with modified output constraints. A pre-gold quality pass is not an accuracy guarantee.
- **E8:** 585 finite WoR paths and 36 PAC Decimal comparisons passed. Finite tests do not establish universal implementation-level validity.

The canonical code for numerical handling is in `AuditRepair-v8.3/study/corrected`, the procedure specification in `AuditRepair-v8.3/study/docs/CURRENT_PROCEDURE.txt`, the operational-level power protocol in `AuditRepair-v8.3/study/docs/POWER_PROTOCOL.txt`, and the E8 tests in `AuditRepair-v8.4/code`.
