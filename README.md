# Rocket League Self-Play AI

Collaborative deep reinforcement learning research using Python 3.11, PyTorch,
RLGym 2.x, RocketSim and RLGym-PPO. We train agents; we do not develop the game.
Historical-opponent self-play is a later milestone.

## Current status

The package layout, device selection, setup verification and unit tests are ready.
A minimal 1v1 RocketSim environment now supports bounded rollout verification.
A bounded PPO smoke command now uses the pinned learner and validated TOML
settings. CPU learning updates have passed on macOS. Checkpoint save/reload, fresh-process resume and fixed-seed evaluation against a
frozen initial policy are verified on CPU. Historical-opponent pools remain future work.

## Layout

```text
src/rocket_league_ai/
  environments/       # factory.py, observations.py, actions.py, rewards.py
  agents/             # policies.py (checkpoint inference)
  training/           # Devices, config, PPO orchestration and checkpoints
  self_play/          # opponent_pool.py (future historical checkpoints)
  evaluation/         # runner.py (bounded frozen-opponent evaluation)
configs/              # Validated TOML smoke settings
scripts/              # Setup/environment verification and train_smoke.py
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

The current project-local Python 3.11 environment has passed CPU setup and PPO
smoke checks with Torch 2.5.1, plus a standalone MPS probe. This does not validate
PPO on MPS or Windows CUDA. The package allows `torch>=2.5.1,<3` to accommodate existing
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
| Initial RLGym-PPO | Auto when available; smoke test still needed | Unverified; project selector rejects it | Mac smoke test passed |
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

## PPO smoke check

```bash
python scripts/train_smoke.py --config configs/baseline.toml
```

This opt-in command defaults to one collection/update iteration with one simulator
worker, 256 agent transitions, two 32-unit hidden layers per network, and one PPO
epoch. `training.iterations` allows up to ten iterations; `timesteps` is per iteration.
It uses the pinned library's collection, advantage calculation, and PPO methods;
it does not implement PPO itself. It avoids the library's interactive training loop
so exceptions propagate to a nonzero exit status. Observation and return
standardization are disabled for this initial smoke check.

The default is CPU with a 120-second deadline covering initialization, learning and
cleanup. The supervisor stops the process tree on timeout. The learner needs local
loopback sockets; a sandbox may require permission. Online logging and automatic
checkpoint loading are disabled. Each completed iteration saves policy/critic
weights, both optimizer states, counters and a compatibility manifest. Saved tensors
are checked against live state, and an independent CPU policy verifies inference.

Success requires finite rollout data, metrics and parameters, nonzero updates to
both policy and critic, and worker cleanup. This is a training integration check,
not evidence of improved playing strength. It is separate from ordinary unit tests.

Artifacts go to a unique ignored `logs/ppo/` directory: input TOML, resolved config
and learner settings, seed, commit/dirty status, platform, Torch/CUDA information,
dependency snapshot, console output, progress, checkpoints, evaluation, and result JSON. Failed runs retain
logs but do not produce a successful result file. For Windows CUDA validation,
copy the config, change `device` to `"cuda"`, and pass that file with `--config`.
Windows process-tree cleanup and CUDA learning still require testing there.

### Resume a checkpoint

```bash
python scripts/train_smoke.py --config configs/baseline.toml --resume logs/ppo/RUN/checkpoints/256
```

Replace `RUN` and the step number with an actual saved checkpoint. Resume restores
both networks, Adam optimizer state and learning counters, then performs the
configured number of **additional** iterations in a new run directory. Environment
settings must match the checkpoint. Episodes and random streams restart, so this
is not a bitwise continuation of an uninterrupted run. Keep the whole run directory:
resume also needs its `reference_policy.pt`. Tensor files load with `weights_only=True`.
Checkpoints from other architectures or normalization settings are unsupported.

### Initial evaluation

After learning, the saved policy plays a frozen copy of the original untrained
policy using two fixed seeds and both team assignments (four episodes, at most
150 steps each). The initial policy is also evaluated against that same opponent.
Both use deterministic actions and CPU inference. Resume preserves the original
opponent, allowing like-for-like comparisons within that run's lineage.

`evaluation.json` records wins, losses, draws, touches, reward and episode lengths.
This is a small baseline, not a competitive skill benchmark. Initial checks yielded
four draws and no touches; no improvement in playing strength is claimed.
Evaluation does not train or modify either policy. It shares the run's overall
wall-clock deadline. If a later iteration/evaluation fails, completed checkpoints
remain available even though the run has no successful `result.json`.

## Next milestone

1. Validate the bounded PPO command and resume on Windows CUDA.
2. Run staged experiments and compare checkpoints using the same evaluation protocol.
3. Expand evaluation before adding historical-opponent pools or longer training.


## Watch a checkpoint on Apple Silicon macOS

RLViser 0.9.3 is installed in ignored `artifacts/rlviser/`, with its bridge in a
separate `.venv-viewer`. This isolation is necessary: the current bridge needs
NumPy 2, while the training stack requires NumPy 1.26.4. Do not install the bridge
into the training environment.

One-time setup on another Apple Silicon Mac (official binary checksum verified):

```bash
python scripts/setup_viewer.py
```

With the training virtual environment active, watch the newest saved checkpoint:

```bash
python scripts/watch_policy.py
```

The command records 30 seconds of deterministic policy play at 15 frames/second,
then loops it in RLViser for two minutes. Blue is the saved policy; orange is its
frozen initial opponent. This is checkpoint playback, not live training or a
Rocket League replay file. `--checkpoint path/to/PPO_POLICY.pt` selects another
policy; `--seconds 60 --watch-seconds 300` records a minute and watches for five.
`--record-only` exports without opening the viewer. Close the viewer or press
Ctrl+C to stop. All viewer assets, settings and exported frames stay ignored.

Press **Esc** to hide the menu, **1/2** to follow a car, **9** for director camera,
**0** for free camera, **P** to pause, and **+/-** to change playback speed.
Dragging objects does not change the recorded simulation. Repeated kickoffs are
normal episode resets. The display timer counts playback rather than training time.
Official viewer: https://github.com/VirxEC/rlviser


## Watch training live on macOS

```bash
python scripts/train_smoke.py --config configs/live.toml --live
```

This starts a **fresh** CPU run and opens RLViser alongside it. Both cars use the
current learning policy with sampled actions. Each real simulator step publishes
an atomic latest-state file to the viewer's separate environment. No replay is
used. The simulator is paced to real time, so it runs much slower than headless
training. The view can briefly stop moving during PPO updates and final evaluation.

The supplied config runs eight iterations of 512 agent transitions (4,096 total),
about 137 seconds of simulation, with a four-minute overall deadline. Checkpoints
are saved after each iteration. `--resume path/to/checkpoint` continues a compatible
checkpoint instead. Without `--live`, training keeps its original full-speed mode.

Camera keys work as in playback. Viewer pause/speed controls do not pause or alter
the training simulation. Closing the viewer does not stop training; Ctrl+C in the
training terminal stops the run. The supervisor closes the viewer when training
ends. A run's `live-viewer.log` records viewer errors separately from training.
Only one RLViser session should be open at a time (the bridge uses fixed ports).


## Enclosed arena and first approach curriculum

RocketSim already uses a closed soccar collision mesh. Bounded tests launch the
ball at the side wall, back wall away from the goal, and ceiling and verify rebound
without escaping. The viewer adds a simple wall/roof grid and open goal mouths as
visual guides. These rectangular guides do not replace the rounded collision mesh.

The upstream macOS RLViser 0.9.3 build receives custom render groups but omits
Bevy 0.18's `bevy_gizmos_render` feature. After viewer setup, build the corrected
local copy with `python scripts/build_viewer.py`. This keeps a Rust toolchain and
source under ignored `artifacts/viewer-build/`, verifies the source commit, enables
that single feature, and installs the result in the local viewer app. The first
build can take several minutes. Training dependencies are unchanged.

For a bounded headless approach/contact experiment:

```bash
python scripts/train_smoke.py --config configs/approach.toml
```

This is an easier practice task: both cars start 900–1500 units from the ball,
roughly facing it, with randomized positions and headings. `approach_v1` rewards
moving toward the ball (at most 0.25 per simulated second), touches (1), directed
touches toward the opponent goal (up to 1), and goals/concessions (+10/-10).
It gives no approach reward for standing still, moving sideways or moving away.
These weights are an experiment, not a proven reward design.

The new config uses 30-second episodes and `no_touch_seconds = 0`: only a goal or
the episode clock ends play. Older configs keep their existing limits and sparse
reward profile. Changing reward/start settings requires a fresh run; compatible
older sparse checkpoints still load. Full competitive match timing/overtime is not
implemented.

Each iteration now records actual ball touches, goals, car speed, signed speed
toward the ball and mean reward in `progress.json`. The usual kickoff evaluation
remains separate from the easier `practice-evaluation.json`. Both compare against
the frozen initial opponent. Do not compare shaped reward totals with old sparse
reward totals as if they measured the same objective.

Train headlessly for throughput and watch checkpoints periodically. The approach
pilot collected 34 touches and four goals during 40,960 sampled training
transitions, but deterministic kickoff and practice evaluation still had zero
touches. This is not evidence of learned competence. Next compare longer staged
runs and multiple seeds, then gradually reintroduce full kickoff starts and
harder opponents. Historical-opponent self-play remains unimplemented.
