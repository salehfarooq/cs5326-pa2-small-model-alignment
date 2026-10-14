# CS5326 PA2: Small-Model Alignment

**Saleh Farooq · 27100063**

Response-only SFT and two custom DPO runs on Qwen2.5-0.5B, plus Ramsay-style LoRA DPO on SmolLM2-360M-Instruct.

[Report](REPORT.pdf) · [Report source](REPORT.tex) · [Evaluation summary](evaluation/summary.csv) · [Submission files](SUBMISSION.md)

## Results

| DPO vs SFT | Wins | Ties | Losses | Tie-adjusted rate |
| --- | ---: | ---: | ---: | ---: |
| HH on Alpaca | 101 | 98 | 101 | 50.0% |
| HH on HH | 115 | 47 | 138 | 46.2% |
| UltraFeedback on Alpaca | 91 | 63 | 146 | 40.8% |
| UltraFeedback on HH | 103 | 24 | 173 | 38.3% |

Each row contains 300 blind judgments from Gemini 3.6 Flash. HH tied SFT on Alpaca and scored lower on HH. UltraFeedback lost on both sets despite its higher validation preference accuracy. The selected persona adapter reached 100% preference accuracy and changed 2/40 answers; the final adapter changed 6/40.

Evaluated checkpoints: **SFT 938 · HH 400 · UF 469 · persona 5**. All final comparisons use greedy decoding, 256 new tokens, repetition penalty 1.1, and no n-gram block.

## File tree

```text
cs5326-pa2-small-model-alignment/
├── README.md
├── REPORT.pdf                  # Final report
├── REPORT.tex                  # Editable report with vector plots
├── SUBMISSION.md               # Five manual hand-in items
├── submission_adapter.py       # Root grading interface
├── src/
│   ├── batching.py             # Prompt/response masking
│   ├── losses.py               # Response log probabilities, SFT and DPO loss
│   ├── training.py             # Custom SFT and DPO loops
│   ├── generation.py
│   ├── judging.py
│   ├── pairwise.py
│   └── ...
├── scripts/                    # Data, training, generation, judging, checks
├── tests/                      # Public adapter and implementation tests
├── configs/                    # Actual SFT, HH, UF, persona, decoder settings
├── data/
│   ├── general/                # Prepared splits and fixed Alpaca subset
│   ├── persona_lora/           # Supplied persona preferences
│   ├── development/            # Development prompts
│   └── exclusions/             # Evaluation prompt exclusions
├── eval_sets/                  # Supplied HH evaluation bank
├── notebooks/
│   ├── 01_Qwen_Training.ipynb
│   ├── 02_Matched_Generation.ipynb
│   ├── 03_Saved_Pair_Judging.ipynb
│   ├── 04_Persona_LoRA.ipynb
│   └── archive/                # Original executed notebooks
├── plots/                      # SFT, DPO, persona, evaluation figures
├── evaluation/
│   ├── summary.csv
│   ├── summary.json
│   ├── matched/                # 2,100 answers and anonymous comparison pairs
│   ├── judged/                 # 1,200 judgments and request logs
│   └── ...                     # Persona and development comparisons
├── evidence/                   # Training metrics and checkpoint records
├── docs/
│   ├── assignment_manual.pdf
│   ├── AI_USAGE.md
│   └── archive/                # Historical preparation notes
├── source_manifest.json        # Original source snapshot used by saved runs
├── requirements*.txt
└── pyproject.toml
```

## Run the checks

Use Python 3.11 and install PyTorch for your platform. On Kaggle, keep its existing CUDA PyTorch installation.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/verify_submission.py
PYTHONPATH=. python scripts/summarize_saved_evaluation.py
```

These checks run without training or judge calls. Result recomputation uses the saved judgments and reproduces wins, ties, losses, and bootstrap intervals.

## Reproduce the experiments

Run the four numbered notebooks on Kaggle. Qwen training uses two T4 GPUs and `requirements-qwen.txt`; persona uses a separate environment with `requirements-persona.txt`. The notebooks train SFT → HH → UF, generate matched outputs, judge saved pairs, and run persona training and comparison. The actual settings are in `configs/`.

Original weights are saved in private Kaggle outputs. Their paths, model revisions, and SHA256 values are recorded in `evidence/` and `evaluation/matched/*/run.json`. The included answers and judgments reproduce the report tables without downloading weights.

Based on the course [starter repository](https://github.com/ahmad903501/small-model-alignment-starter). [Assistance disclosure](docs/AI_USAGE.md).
