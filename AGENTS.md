# Repository Guidelines

## Project Structure & Module Organization
- Core Python code lives in `nanochat/` (models, optimizers, tokenizer, checkpoints, utilities) and `scripts/` (training, evaluation, CLI/web chat entrypoints). Task definitions are in `tasks/`, and small utilities/demos in `dev/`.
- Training orchestration scripts: `speedrun.sh` ($100 d20 pipeline) and `run1000.sh` ($800 d32 pipeline). The Rust tokenizer lives in `rustbpe/` and builds via `maturin`.
- Tests reside in `tests/`; assets such as the logo live in `dev/`.

## Build, Test, and Development Commands
- Install deps with uv (CPU or GPU extras): `uv sync --extra gpu` (or `--extra cpu`); then `source .venv/bin/activate`.
- Rebuild the Rust tokenizer after changes: `uv run maturin develop --release`.
- Quick training pipelines: `bash speedrun.sh` (full $100 run) or `bash run1000.sh` (d32 ~$800). Adjust depth/batch flags inside if tuning.
- Serve/chat locally once a model exists: `uv run python -m scripts.chat_web` (web UI) or `uv run python -m scripts.chat_cli` (terminal).
- Tests: `uv run pytest tests -m "not slow"` for fast checks; include `-m slow` when touching tokenizer or engine paths.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, keep modules small and readable; favor clear, functional helpers over large classes. Use `nanochat.configurator` for CLIs instead of raw argparse.
- Naming follows snake_case for Python/Rust identifiers and kebab-case for scripts; match existing parameter names (e.g., `device_batch_size`, `checkpoints_dir`).
- Minimal dependencies are intentional—avoid adding new ones unless essential. Add concise docstrings/comments only where behavior is non-obvious.

## Testing Guidelines
- Primary framework is `pytest`; tests auto-discover `test_*.py` in `tests/`. Mark long/IO-heavy cases with `@pytest.mark.slow`.
- Add unit coverage for new model components, dataset tweaks, or tokenizer changes; prefer deterministic seeds when sampling.
- Run `uv run pytest tests/test_rustbpe.py -m "not slow"` after tokenizer edits; include slow cases before merging significant training changes.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and specific (see `git log`: e.g., “fix random.seed() footgun bug for SpellingBee data generation”).
- PRs should describe the change, expected impact on training/eval throughput, and how to reproduce (commands + logs). Attach relevant `report.md` diffs or eval metrics when altering training loops.
- Current policy: disclose any LLM-assisted sections in the PR description and ensure you fully understand submitted code. Include screenshots for UI changes and cite linked issues/discussions when applicable.
