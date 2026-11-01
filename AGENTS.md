# Development guidelines

## Scope and architecture

This repository trains agents, not the Rocket League game. Python 3.11, RLGym 2,
RocketSim and RLGym-PPO are the baseline. Source lives in `src/rocket_league_ai`:
- `environments`: environment composition, observations, actions and rewards.
- `agents`: PyTorch policies and inference; no simulator or training side effects.
- `training`: learner integration and conservative device selection.
- `self_play`: future historical checkpoint pools and opponent sampling.
- `evaluation`: bounded fixed-seed evaluation and policy diagnostics.

A minimal RocketSim environment, bounded PPO smoke runs, checkpoint resume and
evaluation are implemented. Historical-opponent self-play remains a placeholder.
Next investigate contact reliability with bounded policy diagnostics before longer
experiments. Do not build PPO from scratch, start long training, or claim
historical-opponent self-play is implemented.
An observation-only drive-to-ball baseline calibrates contact evaluation; keep it
separate from learned policies and do not describe calibration as learned skill.
Controlled learning comparisons must change one setting at a time, use fresh
matched seeds/initial weights and retain the existing baseline defaults. Record
rollout audit statistics and distinguish training activity from evaluation skill.

## Conventions

Use small functions, type hints on public interfaces, snake_case names and pathlib.
Prefer existing RLGym components before custom abstractions. Keep imports free of
training, process spawning and file writes. Guard script entry points with
`if __name__ == "__main__"` for Windows/macOS multiprocessing. Keep environment
factories module-level and picklable. Separate observations/rewards from policies.
Use TOML configs; validate inputs when the config consumer is implemented.

## Dependencies and devices

Inspect existing files and git status before editing. Preserve `check_setup.py`
and the UTF-16 `requirements-windows.txt` snapshot unless a change is requested.
Install platform PyTorch first, then editable package. Never use Windows CUDA
requirements on macOS. Shared dependencies are in pyproject.toml; PPO is pinned
to a commit. Do not silently upgrade the known GTX 1080 PyTorch/CUDA pairing.
PPO auto means CUDA or CPU. MPS detection is not PPO validation; standalone Torch
may use MPS. Require a bounded end-to-end learner test before enabling PPO MPS.

## Checks and collaboration

After editable installation run `python -m unittest discover -s tests -v`,
`python scripts/verify_setup.py --device cpu`, and `python -m pip check`.
Run the device probe with `--device cuda` or `--device mps` on matching hardware.
Use mocks for hardware selection tests; keep ordinary tests short and CPU-safe.
Future simulator/training tests must have explicit step/time bounds and clean up
workers. Report skipped hardware checks and import-only limitations honestly.
Evaluation diagnostics must preserve training RNG streams and policy weights.
Keep the frozen opponent and seed set fixed across candidate policies. Report
sampled and argmax results separately, and distinguish action frequency from
probability concentration; repeated argmax actions alone do not imply collapse.

Never commit virtual environments, credentials, logs, replays or model weights.
Use ignored output directories. Record seeds, resolved config, commit, platform,
Torch/CUDA versions and a platform-specific dependency snapshot for experiments.
Do not promise identical floating-point results across GPU backends. Keep commits
focused, describe behavior and validation, and coordinate dependency changes on
both developers' machines before replacing baseline pins.
