# SUPERSEDED — do not use

Replaced by `analysis/tsla36_2026-08-09/`.

This analysis is invalid for three independent reasons, any one of which is disqualifying:

1. **It includes runs from before `36f4f88`.** `embed_relationship_chains` had no remote
   branch, so on every cluster run the relationship chains were extracted and then
   discarded at the embedding step — the path feature block was silently all zeros. Any
   conclusion here involving path features, `chain_hops`, or `feature_mode` is measuring
   a dead feature block.
2. **It pools MSFT and TSLA**, and pools `candidates=3` runs from the abandoned design
   with `candidates=2` runs from the current one. The balance guarantee holds for neither
   pooled set.
3. **It is partial** — n = 43 of a 72-config target, with `steps=7` entirely absent, so
   the design's marginals do not exist.

The numbers in `SUMMARY.md` and `headline_numbers.txt` here should not be quoted.
Kept only as a record of what was known on 2026-08-07.
