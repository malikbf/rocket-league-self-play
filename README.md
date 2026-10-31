# Rocket League Self-Play AI

Collaborative deep reinforcement learning research using Python 3.11, PyTorch,
RLGym 2.x, RocketSim and RLGym-PPO. We train agents; we do not develop the game.
Historical-opponent self-play is a later milestone.

## Current status

The package layout, device selection, setup verification and unit tests are ready.
A minimal 1v1 RocketSim environment now supports bounded rollout verification.
Policies, PPO orchestration, opponent pools and evaluation remain placeholders.
There is no training CLI yet. Next: integrate the pinned learner and run a bounded
PPO training smoke test.

## Layout

```text
src/rocket_league_ai/
  environments/       # factory.py, observations.py, actions.py, rewards.py
  agents/             # policies.py (future PyTorch networks/inference)
  training/           # devices.py, ppo.py (future learner integration)
  self_play/          # opponent_pool.py (future historical checkpoints)
  evaluation/         # runner.py (future fixed-opponent evaluation)
configs/              # TOML experiment templates
scripts/              # verify_setup.py
tests/               # standard-library unittest suite
```

`pyproject.toml` defines editable installation and shared dependencies.
`AGENTS.md` contains contributor and Codex guidance. The original `check_setup.py`
and `requirements-windows.txt` are preserved unchanged. The latter is a UTF-16
Windows environment snapshot, including `torch==2.5.1+cu121`; it is not a portable
requirements file and does not specify the CUDA wheel index.

## Installation

Install Git and Python **3.11**. Clone the repository, enter its root and use a
separate virtual environment. All following commands run from that root.
Use native arm64 Python on Apple Silicon, rather than an Intel/Rosetta interpreter.
Do not replace an existing working environment just to match these examples.

### Windows / PowerShell / GTX 1080

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
python -m pip check
python scripts/verify_setup.py --device cuda
python -m unittest discover -s tests -v
```

If activation is blocked, invoke `.\.venv\Scripts\python.exe` instead of
`python`; changing system execution policy is unnecessary. Retain the existing
2.5.1/CUDA 12.1 pairing for the GTX 1080. CUDA needs a compatible NVIDIA driver;
a successful import alone is insufficient, so the probe performs a tiny forward
and backward calculation. If CUDA is unavailable, diagnose the driver/build or
use CPU explicitly. For a fresh CPU-only Windows environment install Torch with
`--index-url https://download.pytorch.org/whl/cpu` instead of the CUDA command.
Do not install torchvision or torchaudio unless a later feature needs them.

### macOS / Apple Silicon

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1
python -m pip install -e .
python -m pip check
python scripts/verify_setup.py --device cpu
python scripts/verify_setup.py --device mps
python -m unittest discover -s tests -v
```

The 2.5.1 install is a shared starting baseline, not a claim that it was tested on
this Mac. At scaffolding time the existing Mac environment had Torch 2.15.0;
it was preserved. The package allows `torch>=2.5.1,<3` to accommodate existing
platform installs, so install your chosen exact Torch version **before** `-e .`.
Never install `requirements-windows.txt` or a CUDA wheel index on macOS.
If the MPS probe is unavailable, CPU remains the supported PPO starting point.

Platform installation commands follow the official
[PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).

## Dependency and hardware policy

Shared simulation pins match the existing environment: NumPy 1.26.4,
rlgym-api 2.0.0, rlgym-rocket-league 2.0.1 and RocketSim 2.2.1. RLGym-PPO is pinned
to commit `4ffd2e924198bf4b2d59f4bf280b29919d7c07ea` (reports 1.3.13).
Git is needed to install that source dependency. Depend on RLGym component
packages, as recommended by the [RLGym repository](https://github.com/RLGym/RLGym).
NumPy stays below 2 for this simulation stack.

| Workload | CUDA | MPS | CPU |
| --- | --- | --- | --- |
| Standalone PyTorch | Explicit probe | Explicit probe | Default fallback |
| Initial RLGym-PPO | Auto when available; smoke test still needed | Unverified; project selector rejects it | Mac/default fallback |
| RocketSim physics | CPU simulation | CPU simulation | CPU simulation |

`training.devices.select_device()` selects CUDA then CPU for PPO. With
`for_ppo=False`, auto can select MPS. Explicit unavailable accelerators fail
clearly. These choices do not establish end-to-end training compatibility.
The pinned learner's auto path selects CUDA/CPU; accepting a device string does
not prove every PPO operation works on MPS. Review the
[pinned learner](https://github.com/AechPro/rlgym-ppo/blob/4ffd2e924198bf4b2d59f4bf280b29919d7c07ea/rlgym_ppo/learner.py)
before changing this policy.

## Verification and tests

```bash
python -m pip check
python scripts/verify_setup.py --device cpu
python -m unittest discover -s tests -v
```

The verifier reports Python/platform and package versions, imports the training
stack, constructs/closes RocketSimEngine, and runs a tiny Torch backward pass.
It returns nonzero on failure. It never constructs a learner, spawns training
workers or performs a rollout. The legacy `python check_setup.py` remains usable.
Unit tests use unittest (no additional test dependencies) and mock hardware
availability, so they do not require a GPU. Run hardware probes on each machine;
Windows/CUDA cannot be validated on a Mac. Gym may emit an upstream maintenance
warning through PPO; record errors separately from that warning.

## Reproducing experiments

Shared direct pins are not a complete transitive lock. For each experiment record
its Git commit (and whether the tree is dirty), resolved TOML config, random seeds,
OS/architecture, Python, Torch/build, driver and selected device. Save a complete
platform-specific snapshot with `python -m pip freeze > logs/requirements.txt`
after creating `logs/`. PowerShell may write UTF-16; use UTF-8 when sharing snapshots.
Review snapshots before sharing and recreate them on the same OS/backend; do not
reuse a CUDA freeze on macOS. Establish reviewed per-platform locks once the
training smoke test succeeds on both machines. GPU backends need not produce
bitwise-identical learning curves even with matching seeds.

Keep checkpoints, logs, wandb output and local environments in ignored directories.
Share weights and run artifacts outside Git, together with their config and version
metadata. Use module-level environment factories and guarded script entry points
for Windows/macOS multiprocessing. Avoid network logging by default in future
smoke tests. Future evaluation should use fixed seeds and frozen opponents.

## Environment smoke check

```bash
python scripts/verify_environment.py
```

`environments.factory.build_env()` constructs a 1v1 kickoff environment using
RLGym's `DefaultObs` (92 values), `LookupTableAction` (90 choices), and
`RepeatAction` (8 physics ticks per decision). Actions are one-element integer
NumPy arrays per agent. Rewards are 0.1 for a ball touch and +1/-1 for scoring or
conceding; these are starting weights, not a tuned learning recipe. Goals terminate
an episode; 30 simulated seconds without a touch or 300 simulated seconds total
truncate it. Factory keyword arguments can override action repeat and time limits.
The caller must reset before stepping and close the environment in a `finally`
block. Seed Python's `random` before reset to control kickoff placement.

The smoke check seeds Python and NumPy with 42, performs 120 random steps,
checks observation/action spaces and finite rewards, exercises both timeout
conditions independently, and forces a goal to check termination and rewards.
It also verifies reset after episode completion and closes each environment.
A subprocess enforces a 30-second wall-clock deadline and is killed and reaped
on timeout. This runs CPU physics only; it does not construct a PPO learner or
implement historical-opponent self-play. The unittest suite includes this check.

## Next milestone

1. Wrap it for the pinned RLGym-PPO learner and add explicit config validation.
2. Run a bounded CPU PPO update, then validate Windows CUDA.
3. Save/reload a checkpoint and evaluate before adding historical-opponent pools.
