# AI-assisted implementation work

On 2026-10-09, Saleh requested an implementation session using OpenCode's `opencode/longcat-2.5-preview-free`, with Codex specifying and reviewing the work. Coding conventions were calibrated against the Saleh-style skill and historical local coursework examples. Correctness takes precedence over stylistic similarity.

The working specification and reviewer evidence are stored beside this checkout in `../implementation-session/`. This document records assistance; it does not assert compliance with any unstated course AI-use policy or sole human authorship. The implementation remains subject to student understanding, course requirements, independent review and real experiment validation.

GPU training, model outputs, judge results and final report claims must come from actual saved runs. No results are implied by implementing or passing offline tests.

During data integration, LongCat's free endpoint rate-limited the final revision. Saleh explicitly authorized Codex to finish the remaining fixes. Codex completed evaluation-bank exclusions, streaming-reader cleanup, fixed-bank path handling, the source-specific UltraFeedback prompt-group rule, and the pinned-tokenizer eligibility filter, then ran the independently authored tests and a real artifact/source audit. The final data gate is 228 passing checks. Dataset acquisition, packaging and audit scripts beside the checkout are Codex orchestration work. LongCat's earlier logs and the final reviewer specifications are retained.

On the next authorized continuation, Codex implemented the custom SFT/DPO loops,
CUDA CLI, bounded real-data optimizer pilot and development generation helpers.
Codex added optimizer-oracle, identity, failure and generation boundary tests;
the current local gate is 263 passed. This continues the user-authorized role
change. CUDA pilot results and full model quality require separately saved
runtime evidence.

The bounded Kaggle pilot then made two real SFT updates and one per DPO track,
with saved metrics/checkpoints and reviewed development outputs. Both references
remained unchanged. Evidence was downloaded and verified. These tiny runs are
optimizer/memory checks, not completed assignment trainings or proof of DPO gains.
Two further local checkpoint-selection checks passed after upload (265 total
local checks); the uploaded suite has 263, with 261 passes/2 skips on Kaggle.

For the authorized baseline/full-SFT launch, Codex added a pinned 300-prompt
baseline exporter with per-response persistence, generation interruption tests,
and notebook-friendly training progress logs. The local suite passes 268 checks.
Kaggle passes 265 with three environment/input-specific skips. The training CLI
also accepts the verified source manifest's Git identity when the uploaded
snapshot has no Git directory. These are reviewed launch changes; model quality
and full-training completion are recorded separately from code validation.

On 2026-10-10, Codex prepared and executed a development comparison notebook using the full SFT checkpoint: 15 same-setting Base/SFT prompt pairs and 7 additional matched decoding-ablation pairs, then reviewed the outputs and checked isolated generated functions. The review records improvements, regressions, safety failures and decoding constraints without claiming final benchmark gains.

For the reviewed DPO stages, LongCat implemented small scoped drafts through OpenCode: the added development probes, initial checkpoint identity verifier, diagnostic hooks and blind pairing. Codex independently rejected and repaired provenance, diagnostics and comparison defects, then completed reusable checkpoint generation, resumable judging and early CLI validation after bounded endpoint timeouts. Two further free-endpoint readiness attempts returned no output within 120 seconds. This is the explicitly authorized Codex fallback, not a claim that LongCat wrote the final implementation. Source hashes, specifications, failed drafts, tests and Kaggle versions are preserved beside the checkout. The full read-only preference retention audit and Kaggle source-mount diagnosis are Codex orchestration work. Saved pilot/full training and judge evidence remain separate from implementation validation.

Persona stage (2026-10-11): LongCat 2.5 Preview implemented the narrowly scoped conversational data conversion in src/persona_data.py. Codex found and tested a silent duplicate-filtering defect; LongCat corrected it on revision. The larger initial packet stalled and was interrupted before edits; Codex completed scripts/train_persona_lora.py and Kaggle generation/orchestration under the previously authorized fallback. Codex authored independent split/text/RNG, atomic progress, frozen-reference and protected-interface checks: 17 passed. A pinned Trainer partial-batch coverage issue was independently reviewed and corrected with explicit 13 updates plus a runtime epoch1.0 assertion. Training results and persona quality are not implied by these checks.

Submission preparation (13 October 2026): Saleh instructed Codex to reuse the old runs and write a concise report in the previous PA's structure. Codex drafted REPORT.md/PDF, assembled the local review copy, recomputed result arithmetic, verified data and pair-file identities, rendered the PDF, and reran the existing tests on the assembled copy. No new training, generation, or judge calls were performed. Original implementation assistance remains disclosed above. Saleh's report/content review and course-policy determination are still required; this disclosure does not assert sole human authorship. ZIP creation is deferred at Saleh's instruction.

Report revision (13 October 2026): At Saleh's request, Codex rewrote the report as a standalone LaTeX technical document, added response-only scoring and DPO equations, embedded vector curves from the saved metrics, tightened interpretation and references, and inspected all six rendered pages. Both the desktop editor compiler and bundled Tectonic export succeeded. The reviewed old-run results were preserved; no models or judge calls were rerun. The earlier report is archived outside the review copy.

Final preparation (13 October 2026): Saleh authorized a complete manual audit and public GitHub publication. Codex read all 24 manual pages, recovered the original saved judge archive, reproduced all 1,200 judgments' counts and confidence intervals from the raw records, and verified 250 requests/10 retries. Codex removed the requested assignment-manual bibliography entry, added the completed repository URL, prepared reproducible workflow notebooks and CPU CI, and retained the distinction between persona qualitative evaluation and external judging. No training or judge requests were rerun during this preparation.
