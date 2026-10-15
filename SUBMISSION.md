# CS5326 PA2 submission

Saleh Farooq · 27100063

| Manual §16 item | Final file / location |
| --- | --- |
| Completed GitHub repository | [salehfarooq/cs5326-pa2-small-model-alignment](https://github.com/salehfarooq/cs5326-pa2-small-model-alignment) |
| Final report PDF, maximum eight pages | [REPORT.pdf](REPORT.pdf) |
| SFT and DPO training plots | [sft.png](plots/sft.png), [dpo.png](plots/dpo.png); [persona.png](plots/persona.png) |
| DPO-versus-SFT evaluation summary | [summary.csv](evaluation/summary.csv), [summary.json](evaluation/summary.json) |
| Root grading adapter | [submission_adapter.py](submission_adapter.py), connected to `src/`; public tests in `tests/test_public_adapter_smoke.py` |

The local `PA2/hand-in/` folder contains the upload files:

```text
hand-in/
├── REPOSITORY.txt
├── REPORT.pdf
├── README.md
├── submission_adapter.py
├── src/
├── tests/
│   ├── test_public_adapter_smoke.py
│   ├── toy_lm.py
│   └── __init__.py
├── requirements.txt
├── pyproject.toml
├── plots/
│   ├── sft.png
│   ├── dpo.png
│   └── persona.png
└── evaluation/
    ├── summary.csv
    └── summary.json
```

The hand-in folder includes the root adapter, its implementation, and the unchanged public tests. Run `PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py` from that folder. Prepared data, training notebooks, saved answers, and judgments are in the full repository.
