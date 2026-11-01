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

`live.toml` is a fresh seed-43 CPU run paced for visual inspection: 4,096 agent
transitions with a four-minute deadline. Use `--live` to open the viewer. The CLI
rejects live workloads that cannot fit the configured deadline at real-time speed.

`approach.toml` adds near-ball practice starts and the `approach_v1` reward profile.
It disables inactivity resets (`no_touch_seconds=0`), keeps a 30-second episode
clock, and caps the headless experiment at 40,960 transitions/five minutes.
Changing the environment profile requires a fresh run, not resume from a sparse
checkpoint. Existing sparse configurations retain their old behaviour.

Policy diagnostics do not require a training config or resume. Run
`python scripts/diagnose_policy.py logs/ppo/ACTUAL-RUN` on an existing run to
compare its initial policy and latest five retained checkpoints. The command
uses CPU inference, a 300-second deadline, ten fixed seeds (1001–1010) and the existing sparse
evaluation rewards for both kickoff and approach starts. It performs no updates.
Use `--seed-count 2` to reproduce the original two-seed coverage, or any count
from 1 to 20. Output is `diagnostics-N-seeds.json`; existing two-seed
`diagnostics.json` files are preserved. Reports decode the installed action table.
Full diagnostics also include the observation-only `drive-to-ball-v1` calibration
baseline. `--baseline-only` runs just this controller and writes
`calibration-N-seeds.json`, with the same seeds, opponent, start modes and bounds.
This deterministic baseline is evaluated once per start mode, not twice as sampled
and argmax; its action distribution is one-hot. It uses no training config changes.

`training.entropy_coef` now exposes the pinned learner's entropy bonus. Its
default remains 0.005; finite values from 0 to 0.1 are accepted. This changes
optimization, not environment/checkpoint architecture; resumed runs use the new
config's coefficient while restoring optimizer state. Use fresh runs for ablations.

`approach-entropy-control.toml` and `approach-no-entropy.toml` are a matched fresh
seed-42 CPU comparison: 40,960 transitions per run, at most 300 seconds each.
They differ only in experiment name and entropy coefficient (0.005 versus 0).
Run each with `scripts/train_smoke.py --config ...`, then pass their actual run
directories to `scripts/compare_entropy.py CONTROL_RUN TREATMENT_RUN`.
The comparison validates matching configs and initial policy weights, evaluates
the final policies/shared initial/driving baseline on ten seeds, and writes a
unique JSON under ignored `logs/experiments/` (280 episodes, 300-second deadline).
It requires successful fresh 40,960-transition runs and does no further training.

Every training iteration now records read-only `audit` data in `progress.json`:
reward/advantage/value-target statistics, buffer/transition sizes, terminal and
truncation counts, and maximum action log-probability error against the current
policy before updating. Error above 1e-4 or nonfinite audit data fails the run.
Advantages/targets describe the actual consumed buffer; they are not normalized
or changed by the audit. Existing configs retain their learning behavior.
