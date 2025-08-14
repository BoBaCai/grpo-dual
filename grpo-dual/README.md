# grpo-dual

A shared research repository for **dual-objective GRPO** (factuality + fairness).  
This skeleton includes folders for prompts, judged pairs, training code, and evaluations.

## What’s inside
```
grpo-dual/
  data/                     # raw + processed
  prompts/                 # judge templates (versioned)
  judgments/               # CSV/Parquet of judged pairs
  src/
    judges/                # inference runner for judge models
    grpo/                  # GRPO trainer + CAGrad
    models/                # LoRA/QLoRA adapters
    evals/                 # metrics, bootstraps, plots
  configs/                 # YAMLs (model, train, eval)
  scripts/                 # CLI entrypoints
  reports/                 # notebooks, figures
  README.md
  LICENSE
```

## Quickstart

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. Set environment variables (or edit `configs/*.yaml`):
   ```bash
   cp .env.example .env
   # then edit .env with your keys/paths
   ```

3. (Optional) Run a dummy judge pass on example inputs:
   ```bash
   python scripts/run_judge.py --config configs/judge_example.yaml
   ```

4. (Optional) Start a (placeholder) training loop:
   ```bash
   python -m src.grpo.trainer --config configs/train_example.yaml
   ```

## Folder conventions

- **prompts/**: versioned judge prompts—e.g., `factuality_v1.md`, `fairness_v2.md`. Keep a changelog in the file header.
- **judgments/**: store **tabular** judgments (`.csv` or `.parquet`). Recommended columns:
  `prompt_id,input,output_a,output_b,score_a,score_b,explanation,template_version,judge_model,timestamp`.
- **configs/**: reproducible YAMLs for judging, training, and evaluation.
- **src/**: Python packages for judges, GRPO training (with CAGrad), model adapters, and eval metrics.
- **scripts/**: CLI entrypoints to tie things together.
- **reports/**: Jupyter notebooks and generated figures.
- **data/**: `raw/` for immutable inputs, `processed/` for model-ready data.
  Use `.gitkeep` to keep empty folders. Consider Git LFS for files >50 MB.

## Team workflow (suggested)

- Default branch: `main` (protected). Work on feature branches:
  `feature/<area>-<short-desc>` (e.g., `feature/judges-batch-run`).
- Open Pull Requests to merge. Use brief, specific commit messages.
- Keep judgments deterministic: ensure prompt/template/version are logged.

## License

MIT (see `LICENSE`).
