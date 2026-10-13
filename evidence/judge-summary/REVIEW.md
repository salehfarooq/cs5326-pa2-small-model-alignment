# Saved judging results review

Kaggle version 357378242 completed successfully in 4630.8 seconds. All four banks contain 300 judgments, with zero unknowns, failures or pending rows. Judge: google/gemini-3.6-flash, SDK 0.6.1.

| Comparison | DPO wins | Ties | SFT wins | Tie-adjusted | Decisive | 95% tie-adjusted CI |
|---|---:|---:|---:|---:|---:|---|
| pairs-hh-alpaca_eval | 101 | 98 | 101 | 50.0% | 50.0% | 45.3%–54.5% |
| pairs-hh-hh | 115 | 47 | 138 | 46.2% | 45.5% | 38.9%–53.2% |
| pairs-uf-alpaca_eval | 91 | 63 | 146 | 40.8% | 38.4% | 36.2%–45.7% |
| pairs-uf-hh | 103 | 24 | 173 | 38.3% | 37.3% | 31.0%–45.7% |

HH-DPO does not demonstrate improvement: both intervals overlap 50%. UltraFeedback-DPO is worse in this single-judge evaluation: both intervals fall below 50%. HH bootstrap accounts for 185 canonical instruction clusters among 300 rows. Intervals do not measure judge bias or training-seed variability.

Verification scope: saved expanded comparisons.json viewer and completed notebook logs checked; counts summed and both rates independently recalculated. Full raw per-row/journal audit remains pending: Chrome blocked the archive download. Local comparisons.json is a structured extraction of visible saved metrics, not the byte-identical original file. comparisons-view.txt preserves the DOM evidence.

SDK warning after completion: task returned float while this utility task expects None/Unknown. All 1200 judgments and comparison files were saved before the warning; notebook run is successful. Correct the task return before a future utility run; do not rerun completed judging solely for this warning.

Next separate assignment stage: persona LoRA DPO. Do not tune the current models using these final-bank scores.
