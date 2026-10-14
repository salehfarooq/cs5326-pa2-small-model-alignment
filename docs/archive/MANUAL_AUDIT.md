# Full manual audit

Audit date: 13 October 2026. Source: all 24 pages of `docs/assignment_manual.pdf`, including background, interface contract, Tasks 1–8, report instructions, submission checklist, integrity guidance, and references. **The five hand-in items are assembled.** Implementation acceptance is supported by offline tests and saved-run evidence; it is not a promise of private-grader marks or positive model quality.

## Sections 1–6: scope, interface, compute, and data

| Manual requirement | Finding and evidence |
| --- | --- |
| Understand pretraining/SFT/RLHF/DPO; no pretraining or PPO required (§1–2) | Report §2 explains response-only scores, reference-relative rewards, and custom DPO; the project starts from the specified pretrained models. |
| Three preference tracks and library boundary (§3) | `src/training.py` is a direct PyTorch loop shared by HH and UF. `scripts/train_persona_lora.py` alone uses TRL DPOTrainer/PEFT. No Trainer/DPOTrainer is used for general SFT or DPO. |
| Six root adapter signatures unchanged (§4.1) | `submission_adapter.py` calls the authored implementation. AST argument signatures match upstream; record in `evidence/local-test-results.json`. |
| Padding/truncation and EOS-before-truncation (§4.1) | `src/batching.py`; EOS can be removed by right truncation as specified; masks are position-based and safe when EOS equals PAD. Boundary and row-order tests pass. |
| Prompt/padding labels -100, response-only causal scores (§4.1) | `src/batching.py`, `src/losses.py`; logits/labels are causally shifted, response sums preserve input order. |
| Scalar SFT reduction and DPO mean; no reference gradients (§4.1) | SFT token mean is documented in code/report; stable `-logsigmoid` DPO batch mean, frozen reference under `no_grad`. Numerical/gradient tests pass. |
| Public smoke suite plus independent edge checks (§4.2) | Seven unchanged public smoke tests pass; full local suite has 381 checks covering numerical, tokenizer, accumulation, provenance, generation, and judging behavior. |
| Kaggle T4 x2, version logs, preserve CUDA PyTorch (§5) | Original executed notebooks and run metadata record the T4 layout and versions. Reproduction notebooks clone the completed repository and preserve preinstalled torch. |
| Minimal requirements and pyproject, no torch pin (§5.2) | `requirements.txt` is the general stack; separate pinned `requirements-qwen.txt` and `requirements-persona.txt`; none replaces CUDA torch. `pyproject.toml` supports editable import. |
| Durable stage workflow, best/final saves, incremental outputs (§5.4–5.5) | Custom loops save metrics/state and best/final; generation persists answers and judging persists attempts/results. Historical saved notebooks and current reproduction workflows are included. |
| Optional output ZIP and optimizer resume (§5.5) | Runtime ZIP is extra advice, not a hand-in requirement. No new submission ZIP was created. Optimizer-state resume is not claimed; saved generations/judging can resume. |
| Qwen2.5-0.5B, SFT single turn 30k/1k (§6.1–6.2) | `data/general/sft/manifest.json`, SFT origin in `evidence/dpo/hh-run.json`; first-turn UltraChat extraction, seed 42. |
| HH and UF independent 15k/1k (§6.3) | Manifest hashes/rows checked; normalized sources preserved; HH multi-turn records filtered. |
| Fixed evaluation sets (§6.4) | Exact seed-42 Alpaca 300 and supplied HH 300; persona 40 fixed prompts across all four requested categories. |

## Tasks 1–7: executed experiments and evaluation

| Task | Status and evidence |
| --- | --- |
| 1. GitHub/Kaggle setup (§7) | Completed project repository, requirements/pyproject, original Kaggle source/version records, and fresh-clone notebooks. |
| 2. Base generation (§8) | `evaluation/matched/base-alpaca_eval/outputs.json`: all 300 outputs, required id/instruction/output/generator/dataset fields, complete prompts and greedy matched decoding. |
| 3. Own SFT loop (§9) | Formatting, masking, padding, accumulation, FP16 autocast/FP32 trainable weights, clipping, validation, and best/final persistence verified in code/tests and run metadata. 938 updates, zero skips, best 938. |
| 4. Two own DPO runs (§10) | Same SFT 938 policy/reference origin; HH/UF each 15,000 pairs, 469 updates, zero skips, fixed LR 1e-6, beta 0.1, effective batch 32. Best HH 400 and UF 469; frozen-reference fingerprints unchanged. |
| 5. TRL/PEFT persona (§11) | Correct SmolLM2 model and full supplied 220 rows, conversational conversion, 198/22 split, justified rank/targets/LR/batch/epochs, original 13 updates with best 5. Style/usefulness/safety evaluated on 40 matched prompts with saved reviews and functional checks. Weak style transfer is an outcome, not omitted work. |
| 6. Sanity then generation and anonymous pairing (§12) | Three-prompt sanity files accompany the seven general banks. 2,100 saved answers; four 300-pair blind banks. Seed-42 pairs and hidden identity mappings reproduce byte for byte. Persona baseline/selected-adapter anonymous pairs are also included. |
| 7. Kaggle judge on saved pairs (§13) | Original utility version 357378242 with Gemini 3.6 Flash, all 1,200 judgments and attempt journals recovered. No generation inside judging. Five-pair batches, >=5s attempt pacing, <=3 attempts, persistence, failure log handling, resume and invalid/partial response validation are implemented and tested. |
| Judge results/failures/retries (§13.2) | 250 request attempts; ten retries (one HH/Alpaca, nine UF/Alpaca), no unknown/unresolved-failure/pending rows. `scripts/summarize_saved_evaluation.py` reproduces every count, rate, and bootstrap interval exactly. Missing failure files mean no unresolved rows; run summaries explicitly record zero. |

## Task 8: report sections 14.1–14.7

| Required reporting element | Report coverage |
| --- | --- |
| <=8 pages including figures/tables/references | Final PDF is six pages; LaTeX source and render/compile verification included. |
| Repository, base checkpoints, sources, subsets, seeds, filtering | §1 and §7, data table, local manifests/configs. |
| Compact four-track actual configuration table | Table 2: parameter strategy/rank/alpha/dropout/targets, optimizer/LR/schedule/warmup/WD/clip, batch/accumulation/precision/context/steps/seed, beta/reference and selected checkpoint. |
| Generation settings and prompt/chat format | §2.2: greedy/temperature/top-p inactivity, max output, repetition/no-ngram/length controls, literal Qwen prompt and native SmolLM2 template. |
| Rationale and guide deviations | §3 separates memory/layout, decoding tradeoffs, optimization, and persona choices. General training retains guide settings; no completed hyperparameter search is claimed. |
| Training plots/diagnostics/checkpoint explanation | Figures 1–2, Table 3, §4: SFT train/validation loss; DPO loss/accuracy/margin; persona loss/margin; smoothing and selected updates stated. |
| DPO vs same SFT on same prompts/decoder | §5, Table 4: each bank N=300, W/T/L, both rates, judge, confidence intervals, all original failures/retries and invalid handling. |
| Separate persona target | §5.2/Table 5: SmolLM2 baseline/adapter on same bank/decoder, preference diagnostics, style/usefulness/safety examples. The assessment is qualitative; external persona win-rate/consensus is not recorded. |
| Interpret who is better and by how much | §5 and §6: HH inconclusive/parity intervals, UF below SFT, measured negative results separated from hypotheses; explicit deltas and linked data/output risks. |
| Improvements/regressions/unchanged examples | Table 6 and §6: bead arithmetic, bat-and-ball, extraction, executable code/comments, planet-name function; coding/cooking/safety persona examples. |
| Limits and reproducible evidence paths | §7 and Table 7; original raw judge files now included, separate configurations, source revisions/checkpoint hashes, exact stack, and assistance disclosure. |

## Submission checklist and integrity (§16–18)

The five mandatory hand-in items are listed in `SUBMISSION.md`: completed GitHub URL, report PDF, training plots, evaluation summary, and root adapter/public tests. The manual does not ask for a ZIP, local multi-gigabyte weights, new training, or a guaranteed DPO improvement. The supplied manual stays in the repository for audit, but has been removed from the report bibliography at Saleh's request.

External method/API references remain in the report. Assistance is transparently recorded in `AI_USAGE.md`; technical verification does not certify sole human authorship or an unstated course AI policy. No staff/private implementation or copied public DPO trainer is presented as student-written work.

## Limits to a blanket “all boxes” claim

- The persona track has a matched qualitative assessment, without an external blinded persona judge score. This meets Task 5's explicit instruction to assess style/usefulness/safety. If §14.5's per-track judging wording is interpreted as requiring an external judge for persona too, that is the remaining methodological gap; no score has been invented.
- Full trained weights are preserved in private Kaggle saved versions. Public repository calculations do not need them; inspecting those weights requires staff access.
- Private grader outcomes and human-authorship/course-policy determinations cannot be certified by passing the public/offline tests.

These limits are stated so the project is reviewable without exaggerating its completeness or quality.
