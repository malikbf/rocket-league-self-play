# Experiment configurations

`baseline.toml` is consumed by `python scripts/train_smoke.py --config configs/baseline.toml`.
It defaults to one small PPO iteration; `training.iterations` allows up to ten.
Each iteration saves a checkpoint, and the run ends with bounded evaluation.

Missing keys use documented defaults in `training/config.py`. Unknown sections and
keys, invalid types, nonfinite limits, PPO MPS, and incompatible batch sizes are
rejected before workers start. Smoke runs allow 32–4096 even agent transitions,
one to four epochs per iteration, and a 10–300 second deadline for the entire run.
With `--resume`, the configured iterations are additional work after restoration.
The environment settings must match the saved checkpoint.

Outputs always go to a unique ignored `logs/ppo/` directory. Each run records the
input and resolved settings, seed, Git state, platform, dependencies and metrics.
Keep credentials and machine-specific paths out of shared configurations.

`pilot.toml` is the first bounded experiment: ten iterations of 4,096 transitions,
two PPO epochs per iteration, and a five-minute overall deadline. Run with
`python scripts/train_smoke.py --config configs/pilot.toml`. It starts a fresh
policy unless `--resume` is supplied. The learner retains the latest five checkpoints.
