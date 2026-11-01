# Rocket League Self-Play AI

Collaborative deep reinforcement learning research using Python 3.11, PyTorch,
RLGym 2.x, RocketSim and RLGym-PPO.

## Current status

The package layout, device selection, setup verification and unit tests are ready.
A minimal 1v1 RocketSim environment now supports bounded rollout verification.
A bounded PPO smoke command now uses the pinned learner and validated TOML
settings. CPU learning updates have passed on macOS. Checkpoint save/reload, fresh-process resume and fixed-seed evaluation against a
frozen initial policy are verified on CPU. Windows CPU and bounded CUDA training/
resume checks have also passed with Torch 2.5.1+cu121. Policy diagnostics compare
sampled and deterministic actions across retained checkpoints. Reliable ball
contact is not yet demonstrated. Historical-opponent pools remain future work.

## Layout

To continue development in a later session, start with the "Resume work here"
section in [implementation.md](implementation.md). Experiment artifacts remain
local under ignored `logs/`; Git contains the summarized findings and configs.

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
Windows CUDA collection/update, checkpoint verification, evaluation and normal
worker cleanup have passed for 256-transition fresh and resumed runs. Forced
Windows process-tree timeout cleanup is covered by a mock, not a real timeout test.

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

1. Investigate contact reliability with sampled/argmax checkpoint diagnostics.
2. Compare multiple seeds and expand evaluation before longer staged experiments.
3. Add historical-opponent pools only after meaningful behavior can be measured.


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

## Diagnose saved policies

Use an actual existing run directory, for example on the current Windows machine:

```powershell
python scripts/diagnose_policy.py logs/ppo/approach-v1-20261101T195138Z-03fafc8d
```

This command performs no training. It compares the frozen initial policy and up
to five latest retained checkpoints on CPU, using both kickoff and approach
starts. Each candidate plays 20 episodes per action mode by default (seeds
1001–1010, both teams), at most 150 decisions/10 simulated seconds per episode.
`--seed-count 1..20` changes coverage; `--seed-count 2` reproduces the original
seed set. Training's automatic evaluation retains its two-seed default. The
diagnostic command has a 300-second deadline. Both modes face the same frozen initial
opponent using argmax; only the evaluated policy switches between argmax and
locally seeded sampling. Sampling preserves global Python, NumPy and Torch RNG
streams. Both start modes use sparse rewards and action_repeat=8, regardless of
the training profile; this diagnostic does not reproduce training collection.

`diagnostics-10-seeds.json` (or the requested seed count) contains all results, source-run metadata
and evaluation runtime versions/Git state/dependency snapshot. Re-running replaces
that seed-count-specific file; the original `diagnostics.json` is preserved.
The report also decodes all 90 actions and each candidate's dominant action into
throttle, steer, pitch, yaw, roll, jump, boost and handbrake using the installed
RLGym lookup table. Existing training artifacts and checkpoints are preserved. Ordinary
training evaluation also includes the new per-episode diagnostic fields.

Metrics include episode touch rate, minimum car-to-ball center distance sampled
at reset and decision boundaries, 90-action counts, mean entropy in nats and mean
maximum action probability. Distance is not surface clearance or a contact
detector; use the actual touch counter for contact. Entropy is at most ln(90),
approximately 4.50. Repeated argmax choices can occur even when probabilities
are almost uniform; action frequency alone does not establish policy collapse.

The Windows approach run above collected 73 training touches and one goal in
40,960 transitions. The original two-seed diagnostics found zero contacts for all five retained trained
checkpoints in either action mode or start mode. At step 40,960, argmax selected
action 80 for every decision; approach entropy was about 4.493 and the mean
maximum probability about 1.40%, indicating a broad distribution. Sampling reduced
mean closest approach distance from about 1,192 to 729 units without contact.
The sampled initial policy made four kickoff touches across two of four episodes,
while retained trained policies made none. These few short games do not establish
regression or reliable competence; compare additional seeds before changing the
learning recipe or extending training.

Expanded ten-seed diagnostics (1001–1010, 20 episodes per row) give:

| Policy | Start | Action mode | Touches | Episodes with contact | Mean closest distance |
|---|---|---|---:|---:|---:|
| Initial | Kickoff | Sample | 4 | 2/20 | 1,505 |
| Latest (40,960) | Kickoff | Sample | 0 | 0/20 | 1,549 |
| Initial | Approach | Sample | 14 | 7/20 | 414 |
| Latest (40,960) | Approach | Sample | 12 | 6/20 | 426 |
| Latest (40,960) | Approach | Argmax | 0 | 0/20 | 1,209 |

All retained trained checkpoints still make zero argmax contacts. Sampled
approach episode contact rates range from 15% to 30%, compared with 35% initially.
The extra seeds reveal occasional sampled contact missed by the original report,
but no demonstrated improvement over the initial policy. These are paired,
short evaluations from one training seed, not independent training replicates or
evidence of statistically significant regression.

Action 80 decodes to throttle=0, steer=0, pitch=1, yaw=0, roll=1, jump=0,
boost=0, handbrake=0. It is an aerial orientation control with no driving input
from a grounded start. Repeating it explains why argmax does not drive toward the
ball. Do not remove that action or change rewards based solely on this result;
next check a simple driving/contact baseline and controlled learning comparisons.

## Contact calibration baseline

```powershell
python scripts/diagnose_policy.py logs/ppo/approach-v1-20261101T195138Z-03fafc8d --baseline-only
```

This evaluates `drive-to-ball-v1` against the same frozen initial opponent and
seeds, saving `calibration-10-seeds.json`. It does not train or change checkpoints.
Full diagnostic runs now include this baseline as well as the initial and retained
learned policies (520 episodes at the default ten seeds). Baseline-only runs use
40 episodes; both commands retain the 300-second deadline and `--seed-count`.

The controller reads ball position, own position and forward direction from the
same pinned 92-value DefaultObs vector as the learned policy. It computes signed
horizontal heading error, drives forward and steers left/right outside a 0.1-radian
deadband. It boosts only when aligned within 0.15 radians and more than 800 units
from the ball. It maps these controls to the installed discrete table. It never
jumps, predicts ball motion or reads privileged simulator state. It is a fixed
ground-contact sanity baseline, not a trained policy or competitive opponent.

Ten-seed Windows results (20 episodes per start mode):

| Start | Baseline episodes with contact | Baseline touches | Mean closest distance |
|---|---:|---:|---:|
| Kickoff | 20/20 | 46 | 185 |
| Approach | 20/20 | 62 | 182 |

The latest learned policy had zero argmax contacts and sampled contacts in 0/20
kickoff and 6/20 approach episodes on the same seeds/opponent. Calibration shows
the existing observation/action interface can support contact and the evaluation
can detect it. It does not identify the cause of weak PPO learning, prove the
reward design is sufficient, or measure competitive playing strength. Next use
controlled comparisons of the learning setup rather than changing simulation or
extending training without a hypothesis. Regression tests cover steering, team
coordinate inversion, boost gating and bounded contact on both teams.

## Controlled learning investigation

The first ablation tests whether the default entropy bonus prevents early policy
specialization. Two fresh seed-42 CPU runs use the same approach environment,
initial policy, architecture, learning rates, batch sizes and 40,960-transition
budget. Only `training.entropy_coef` changes: 0.005 in
`configs/approach-entropy-control.toml`, zero in `configs/approach-no-entropy.toml`.
The baseline default remains 0.005; this is a diagnostic experiment, not a new
recommended learning recipe. Each run retains its 300-second deadline.

`scripts/compare_entropy.py CONTROL_RUN TREATMENT_RUN` checks matching settings
and initial policy tensors, then compares final policies, the shared initial
policy and drive-to-ball baseline using seeds 1001–1010 and both teams. It has a
300-second deadline, does no training and writes a unique report under
`logs/experiments/`. See `implementation.md` for recorded run paths and findings.

Training now records an `audit` section per iteration in `progress.json` with
reward/advantage/value-target statistics and action log-probability alignment.
These checks do not change data or optimization. They fail on nonfinite signals
or an alignment error greater than 1e-4. Finite, aligned data and changed weights
still do not prove the PPO targets or learned behavior are correct.

The completed seed-42 comparison found 73 training touches/one goal in the control
and 72 touches/one goal without entropy. Both final policies made sampled approach
contact in 6/20 episodes (12 touches), versus 7/20 (14 touches) for the initial
policy. Both had zero kickoff contacts and zero argmax contacts. The driving
baseline retained 20/20 contact in each start mode. Stored action log probabilities
matched the live policy within 4.8e-7 before every update. Removing entropy did
not improve contact in this test; it is not evidence for disabling it by default.
One training seed and a small budget do not rule out entropy effects at other
scales. Next investigate reward/advantage signal scale and update strength with
another matched experiment; the underlying learning failure remains unresolved.
