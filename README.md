# CS5326 PA2: Small-Model Alignment

**Saleh Farooq · 27100063**

Custom response-only SFT and HH/UltraFeedback DPO for Qwen2.5-0.5B, plus a separate TRL/PEFT persona experiment for SmolLM2-360M-Instruct. This repository contains the completed implementation, original evaluated runs, full saved judgment records, and the final LaTeX report.

[Read the report](REPORT.pdf) · [LaTeX source](REPORT.tex) · [Submission items](SUBMISSION.md) · [Full manual audit](MANUAL_AUDIT.md) · [Results CSV](evaluation/summary.csv)

## Results from the original runs

| Comparison | DPO wins / ties / losses | Tie-adjusted rate |
| --- | --- | --- |
| HH DPO vs SFT on Alpaca | 101 / 98 / 101 | 50.0% |
| HH DPO vs SFT on HH | 115 / 47 / 138 | 46.2% |
| UF DPO vs SFT on Alpaca | 91 / 63 / 146 | 40.8% |
| UF DPO vs SFT on HH | 103 / 24 / 173 | 38.3% |

Each bank contains 300 judged pairs. The original Gemini 3.6 Flash journals contain 1,200 final judgments and 250 request attempts, including 10 retries; there are no unknown, unresolved-failure, or pending judgments. `scripts/summarize_saved_evaluation.py` reproduces both result counts and the saved 2,000-resample confidence intervals exactly.

The original persona adapter reaches 100% validation preference accuracy but leaves 38/40 greedy answers unchanged from its SmolLM2 baseline. Its result is a matched qualitative assessment, not an external blind win-rate estimate. Later follow-up checkpoints are not substituted into these old-run scores. Negative or inconclusive quality results are reported directly.

## Submission and implementation layout

- `REPORT.pdf` and `REPORT.tex`: six-page final report and standalone editable LaTeX source; curves are embedded vector plots.
- `submission_adapter.py`: unchanged six-function grading interface, connected to `src/batching.py` and `src/losses.py`.
- `src/training.py`: custom SFT/DPO loops; TRL and Trainer are used only for the persona task.
- `scripts/`: preparation, training, generation, anonymous pairing, resumable judging, and result recomputation.
- `data/general/`: exact prepared SFT/HH/UF train/validation splits and seed-42 Alpaca subset, with hashes and source manifests.
- `data/persona_lora/`, `eval_sets/`: unchanged supplied preferences and HH evaluation set.
- `plots/`: separate SFT, DPO, persona, and evaluation figures for the submission.
- `evaluation/matched/`: all 2,100 saved model answers and four 300-pair banks with hidden identity mappings.
- `evaluation/judged/`: original raw judgments, requests/attempts, run summaries, and comparisons from Kaggle version 357378242.
- `evaluation/persona-pairs/`: reproducibly blinded original 40-prompt baseline/adapter comparisons; external judgments are not claimed.
- `evidence/` and `configs/`: original training metrics, checkpoint states/fingerprints, actual configurations, and verification records.
- `notebooks/reproduce/`: current path-based workflows for a fresh clone; historical saved notebooks are retained in `notebooks/`.
- `MANUAL_AUDIT.md`, `SUBMISSION.md`, `AI_USAGE.md`: requirement coverage, exact hand-in items, and assistance disclosure.

## Offline checks from a fresh clone

Python 3.11 was used locally. Install a compatible PyTorch build for your platform first. On Kaggle, keep the preinstalled CUDA PyTorch. The general `requirements.txt` deliberately does not install or pin torch; the persona stack is separate.

```bash
git clone https://github.com/salehfarooq/cs5326-pa2-small-model-alignment.git
cd cs5326-pa2-small-model-alignment
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
PYTHONPATH=. python -m pytest -q
python verify_submission.py
python scripts/summarize_saved_evaluation.py
```

The local full suite passes 381 checks; the public smoke suite has seven checks. Platform-specific MPS checks may skip on Linux. The GitHub Actions workflow runs both test suites, file/evidence verification, and result recomputation on a fresh CPU checkout. No model download, GPU training, or judge calls are required for these checks.

## Reproducing training and generation on Kaggle

Use the four notebooks in `notebooks/reproduce/`, in order as appropriate:

1. **01_Qwen_Training**: clone this repository, preserve CUDA torch, print versions/T4 devices, then train SFT → HH → UF sequentially with the recorded recipe. Save the completed notebook version.
2. **02_Matched_Generation**: attach the intended saved training outputs, find completed best checkpoints by metadata, run three-prompt sanity checks and fixed-bank generation with the final 1.1/block-0 decoder, and save blinded pair files. Save that version.
3. **03_Saved_Pair_Judging**: in a Kaggle utility benchmark, attach saved generation outputs and judge only saved anonymous pairs. It uses five pairs per request, at least five seconds between attempts, three attempts, per-batch persistence, failures, and resume. The task returns `None`, avoiding the historical completion-score return warning.
4. **04_Persona_LoRA**: use the separate pinned persona stack, expose GPU0 for training, and generate baseline/best/final outputs on the fixed 40-prompt bank. The native template supplies no caller-injected persona instruction.

These notebooks are reproducibility workflows, not claims that new experiments ran during submission preparation. Original saved versions and exact checkpoint hashes are listed in `evidence/`, `MANUAL_AUDIT.md`, and the report. The original baseline exporter retains historical decoding defaults; use the matched workflow's explicit final settings for fair comparisons.

General runs used PyTorch 2.11.0+cu128, Transformers 4.57.6, datasets 4.8.5, and FP32 weights with FP16 autocast. Persona used Transformers 4.51.3, TRL 0.19.1, PEFT 0.15.2, datasets 3.6.0, and FP32. Seeds and revisions are recorded; bitwise reproducibility across hardware is not assumed. Training rejects output overwrites and does not save optimizer-state resume; generation and judging persist progress.

## Model artifacts and evidence

The manual's hand-in checklist requires the repository, PDF report, training plots, evaluation summary, and grading adapter. Large model weights remain in the original saved private Kaggle outputs rather than Git history. Their best/final paths and SHA256 fingerprints are included in run/state records. The saved originals remain useful for checkpoint inspection; staff need Kaggle access if they want those weight files. Every submitted report result can be recomputed locally from the included text evidence without that access.

The complete judge archive was recovered and independently audited during preparation. Historical notes mentioning unrecovered journals describe the earlier state; the authoritative current records are `evaluation/judged/`, `evidence/judge-journal-audit.json`, and `verification.json`.

## Provenance and assistance

Starter: [ahmad903501/small-model-alignment-starter](https://github.com/ahmad903501/small-model-alignment-starter), upstream base recorded in `source_manifest.json`. Course-supplied smoke tests, toy model, manual, persona preferences, and HH bank are preserved. `AI_USAGE.md` records implementation, review, orchestration, and report assistance. No claim of sole human authorship is made.

No submission ZIP has been created. The manual does not require one, and ZIP creation remains deferred under Saleh's instruction.
