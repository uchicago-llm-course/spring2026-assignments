# HW1: LLM Interpretability

All commands assume you are inside the `HW1/` directory.

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
python -c "import torch, transformers, nnsight, datasets; print('OK')"
```

---

## Autograders

We provide two test suites validate your implementations:

```bash
pytest tests/test_probing.py -v    # Section 1 — src/probing.py
pytest tests/test_kv_cache.py -v   # Section 4 — src/kv_cache.py
```

---

## compile.py

`compile.py` runs the autograder for the relevant section(s), then executes the heavy computations and saves results to `cache/`. Once compiled, the notebook loads from `cache/` and runs instantly on CPU — no GPU needed for the notebook itself.

```bash
python compile.py section1   # probing    → cache/probing_results.json
python compile.py section2   # logit lens → cache/logit_lens_results.pt
python compile.py section4   # KV cache   → cache/kv_cache_benchmark.json
python compile.py all        # all three sections at once
```

You can compile sections independently as you finish them — there is no need to complete everything before compiling.

---

## SLURM example

Compilation requires a GPU (~10–15 min total). Example request on a SLURM cluster:

```bash
srun --partition general --gres gpu:a100:1 --time 0:20:00 --mem 16G \
     python compile.py section1
```

---

## Submit

Export `hw1.ipynb` as PDF and upload to Gradescope.
Do **not** submit `hw1.pdf` or `src/` files.
