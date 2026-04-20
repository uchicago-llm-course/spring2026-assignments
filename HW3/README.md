# HW3: Frontier Lab Interview Prep

All commands assume you are inside the `HW3/` directory.

---

## Setup

### 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart your shell
```

### 2 — Create environment and install dependencies

```bash
uv venv                    # creates .venv/
source .venv/bin/activate  # Linux/macOS  |  Windows: .venv\Scripts\activate
uv pip install -e .        # installs everything listed in pyproject.toml
```

### 3 — Verify

```bash
python -c "import torch, pytest, ipykernel; print('OK')"
```

No GPU required — both questions run on CPU in seconds.

### 4 — Register the venv as a Jupyter kernel

```bash
python -m ipykernel install --user --name hw3 --display-name "HW3 (uv .venv)"
```

`hw3.ipynb` is configured to use this `hw3` kernel. After installing, open the notebook and select **HW3 (uv .venv)** from the kernel picker.

---

## Questions

1. **Distributed Training** (50 pts) — `src/distributed.py`
2. **Attention Residuals** (50 pts) — `src/attn_residuals.py`

Read the problem statements in `hw3.pdf`. Each question has coding problems (look for `YOUR CODE HERE` markers in the corresponding `src/*.py` file) and written analysis questions.

---

## Autograders

We provide two test suites to validate your implementations:

```bash
pytest tests/test_distributed.py -v    # Question 1 — src/distributed.py
pytest tests/test_attn_res.py -v       # Question 2 — src/attn_residuals.py
```

Or run the full suite with `pytest tests/ -v`.

---

## Submit

Open `hw3.ipynb`, fill in the written-answer markdown cells, then export it as PDF (File → Save and Export Notebook As → PDF) and upload to Gradescope. Do **not** submit `hw3.pdf` or `src/` files — the notebook PDF is the single submission artifact.