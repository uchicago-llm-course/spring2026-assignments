# HW2: LLM Alignment

All commands assume you are inside the `HW2/` directory.

---

## Setup (once per machine)

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
python -c "import torch, transformers, trl, evaluate, datasets; print('OK')"
```

---

## Autograders

We provide three test suites to validate your implementations:

```bash
pytest tests/test_sft.py -v                    # Section 1 — src/sft.py
pytest tests/test_instruction_tuning.py -v     # Section 2 — src/instruction_tuning.py
pytest tests/test_rlhf.py -v                   # Section 3 — src/rlhf.py
```

---

## compile.py

`compile.py` runs the autograder for the relevant section(s), then executes the heavy training computations and saves results to `cache/`. Once compiled, the notebook loads from `cache/` and runs instantly on CPU — no GPU needed for the notebook itself.

```bash
python compile.py section1   # SFT training        → cache/sft_results.json
python compile.py section2   # Instruction tuning  → cache/it_results.json
python compile.py section3   # RLHF training       → cache/rlhf_results.json
python compile.py all        # all three sections at once
```

You can compile sections independently as you finish them — there is no need to complete everything before compiling.

---

## SLURM example

Compilation requires a GPU (~15–30 min per section). Example request on a SLURM cluster:

```bash
srun --partition general --gres gpu:a100:1 --time 1:00:00 --mem 32G \
     python compile.py section1
```

---

## Submit

Export `hw2.ipynb` as PDF and upload to Gradescope.
Do **not** submit `hw2.pdf` or `src/` files.
