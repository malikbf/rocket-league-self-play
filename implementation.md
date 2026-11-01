# Implementation status

## Resume work here

Latest development milestone: contact diagnostics, observation-only driving
calibration and the controlled entropy ablation are implemented and validated.
All 26 tests passed on Windows, along with CPU setup and pip check. Earlier small
CUDA training/resume checks passed; the newest rollout audits were exercised on
CPU only. Dependencies and baseline learning settings remain unchanged.

Read the final entropy-ablation section below before starting another experiment.
The next hypothesis is reward/advantage scale or insufficient update strength.
Choose one factor, record the hypothesis and expected measurement, and compare
fresh matched runs on the same seeds/opponent against the initial policy and
drive-to-ball calibration. Do not assume the entropy experiment identified the
cause: disabling entropy did not improve contact in its single-seed short budget.
No further training is running or scheduled.

Useful checks from the repository root with the existing virtual environment:

```powershell
python -m unittest discover -s tests -v
python scripts/verify_setup.py --device cpu
python -m pip check
```

Git includes code, configs and the findings below. Run directories, JSON reports,
weights and platform dependency snapshots under `logs/` remain ignored and local.
They will not arrive with a pull on another machine. Share whole run directories
separately if checkpoint-level comparison is needed; preserve their reference
policy, manifests, configs and metadata. Otherwise recreate the bounded runs with
the checked-in configs. Do not install the Windows dependency snapshot on macOS.

Available commands are documented in README/configs/README: `diagnose_policy.py`,
`diagnose_policy.py --baseline-only` and `compare_entropy.py` (actual run directory
arguments are required). Results and limitations are summarized here so work can
continue even without the local artifacts.

## Step 1 — Project setup: complete

Python 3.11 virtual environment, Torch 2.5.1, pinned simulation dependencies,
editable package installation, and conservative device selection are ready.
PPO uses CPU on this Mac; MPS is available for standalone Torch.

Tests passed on macOS:
- Five device-selection unit tests.
- CPU and standalone MPS forward/backward probes.
- Simulation imports and engine construction/cleanup.
- Dependency consistency check (`pip check`).

## Step 2 — Minimal environment: complete

Implemented a 1v1 RocketSim kickoff environment with 92-value observations,
90 discrete actions, touch/goal rewards, and episode/inactivity limits.

Tests passed on macOS:
- 120 random-action steps with valid observations and finite rewards.
- Episode and inactivity timeouts, including reset of timeout state.
- Forced goal termination and correct scoring/conceding rewards.
- Reset after a goal and environment cleanup.
- Invalid factory settings rejected; factory is picklable.
- Eight tests total, including the five device tests. Rollout has a 30-second deadline.

## Step 3 — PPO integration: complete on CPU

Connected the pinned learner through its RLGym adapter. Added validated TOML
settings and a one-iteration smoke command with one worker, no online logging,
and a 120-second deadline. Run settings, versions, Git state and results are
saved under ignored `logs/ppo/` directories.

Tests passed on macOS:
- 256 agent transitions and four PPO optimizer updates on CPU.
- Finite rollout data, metrics and model parameters.
- Both policy and critic parameters changed; worker cleanup completed.
- Invalid configs rejected; adapter rollout/reset and factory pickling passed.
- Supervisor timeout handling checked with mocks.
- Fourteen ordinary tests total; the PPO run is a separate opt-in check.

## Step 4 — Checkpoints and resume: complete on CPU

Each iteration saves networks, optimizer state and counters. Resume continues in
a new run directory. Runs support up to ten iterations under the same deadline.

Tests passed:
- Saved network/optimizer tensors match live state; reloaded inference matches.
- Fresh-process resume advanced from 256 to 768 transitions and four to twelve updates.
- Invalid checkpoint metadata and nonfinite policy weights are rejected.
- Inference loading preserves the training random-number state.

## Step 5 — Initial evaluation: complete

Saved policies play a frozen initial opponent using two seeds and both sides.
Four short episodes report wins/losses/draws, touches and rewards. No learning
occurs during evaluation; the original opponent is preserved across resumes.

Tests passed:
- Repeated evaluation gives identical results without changing policy weights.
- Each episode is capped at 150 steps; environments close after use.
- Eighteen ordinary tests total, plus separate CPU training/resume checks.

Initial result: four draws, no touches. Better playing strength is not yet demonstrated.

## First training experiment — completed on CPU

`configs/pilot.toml`: ten iterations, 40,960 agent transitions, 160 optimizer
updates, with a five-minute deadline. Checkpoint/inference verification and worker
cleanup passed. The latest five checkpoints were retained.

Evaluation: four draws, zero touches, zero reward, matching the initial policy.
No playing improvement demonstrated by this small evaluation.
Run artifacts: `logs/ppo/pilot-20261031T010851Z-90931eb6/`.

## macOS viewer — connected

RLViser 0.9.3 runs on Apple Silicon. `python scripts/watch_policy.py` plays the
latest checkpoint (blue) against its frozen initial opponent (orange).
The viewer bridge is isolated in `.venv-viewer`; training pins are unchanged.

Verified: exported simulator frames, viewer connection, and rendered field/ball/cars
on screen. Playback is recorded policy behaviour, not live training.

## Real-time training viewer — verified

`python scripts/train_smoke.py --config configs/live.toml --live` opens RLViser
while the simulator collects actual training experience. Both cars use the current
learning policy. Runs are paced to real time; no recorded playback is used.

Verified fresh seed-43 CPU run: 4,096 transitions, eight iterations, 64 PPO updates,
checkpoint/inference checks, evaluation, worker and viewer shutdown. Nineteen tests
pass, including bounded live-frame publishing and pacing. CPU setup and pip checks pass.
Run: `logs/ppo/live-20261031T123923Z-70a170e1/`.

## Arena checks and approach curriculum

Verified physical rebounds at both side walls, both back walls outside the goals,
and the ceiling. RocketSim already has a closed arena; no collision rules changed.
Added simple wall/ceiling grid guides and goal outlines to the macOS viewer.
Rebuilt RLViser with its missing gizmo-rendering feature and visually verified
the enclosure during checkpoint playback. Guides approximate the collision mesh.

Added `configs/approach.toml`: randomized near-ball starts, small approach rewards,
stronger touch/goal rewards, 30-second episodes and no inactivity resets. Old sparse
profiles/checkpoints remain supported. Per-iteration gameplay metrics now count
actual touches/goals and measure car speed and speed toward the ball.

Bounded CPU pilot: 40,960 transitions, 34 touches and four goals during exploration.
Kickoff and easier-practice deterministic evaluation both still had zero touches;
no learned skill improvement is claimed. All 23 tests, CPU setup and dependency
checks pass; an old checkpoint also resumed successfully.
Run: `logs/ppo/approach-v1-20261031T125114Z-2898b4df/`.

## Next / not yet verified

- Compare staged runs across seeds using touches, approaches and goal attempts; expand evaluation.
- Investigate weak contact behavior using bounded checkpoint diagnostics.
- Expand evaluation, then add historical-opponent self-play.

## Run checks

From the repository root, with the virtual environment active:

```bash
python -m unittest discover -s tests -v
python scripts/verify_environment.py
python scripts/train_smoke.py --config configs/baseline.toml
python scripts/verify_setup.py --device cpu
python -m pip check
```

The standalone environment check also runs as part of the unit tests.
CPU training, saving, resuming and initial evaluation are verified. Windows/CUDA
bounded training/resume has now also passed (see below). Playing strength and
historical self-play are unverified. Resume restarts episodes and RNG
streams; it is not an exact continuation of an uninterrupted run.

## Windows validation and policy diagnostics — 2026-11-01

Windows Python 3.11.7, Torch 2.5.1+cu121: all 23 tests, CPU/CUDA forward/backward
probes and pip check passed. CPU baseline training passed. CUDA fresh training
and resume passed at 256 and 512 cumulative transitions, including checkpoint,
inference, evaluation and normal worker cleanup. An earlier CUDA process exited
silently without checkpoint/result; the cause remains unexplained. Real forced
timeout cleanup on Windows has not been exercised; its test uses mocks.

The CPU approach run `approach-v1-20261101T195138Z-03fafc8d` passed all ten
iterations (40,960 transitions), with 73 touches and one goal across both cars.
Kickoff and practice argmax evaluation each had four draws and zero touches.

Added `scripts/diagnose_policy.py`: a 120-second supervised, CPU-only comparison
of the initial policy plus latest five retained checkpoints, two start modes and
argmax/sampled candidate actions against the same deterministic frozen opponent.
Maximum workload is 96 episodes of 150 decisions. No training occurs. Reports
include episode touch rate, sampled minimum center distance, action histograms,
entropy and maximum probabilities, plus evaluation/source-run provenance.
Evaluation uses local sampling RNG and preserves global RNG streams/weights.
Existing training evaluation gains diagnostic fields without changing its default
argmax protocol. Practice evaluation continues to use sparse rewards.

Observed retained steps: 24,576; 28,672; 32,768; 36,864; 40,960. All trained
candidates had zero touches in both modes and starts. Initial sampled kickoff
produced four touches in two of four episodes. Latest approach argmax repeated
action 80 for all steps, but entropy was 4.493 nats (uniform maximum 4.500) and
mean maximum probability approximately 0.014. This is repeated argmax behavior
with a broad distribution, not evidence of probability collapse. Latest sampled
approach reduced mean minimum center distance from 1,192 to 729 units, still
without contact. These small evaluations establish neither improvement nor
statistically reliable regression. Next expand seeds and inspect control behavior
before extending training or tuning rewards.

Validation: 23 tests passed with added sampled-repeatability, RNG/weight
preservation, invalid-mode and metric-invariant assertions. CPU/CUDA probes and
dependency consistency passed. No dependencies or baseline pins changed.

### Expanded seed coverage and decoded controls

The diagnostic CLI now defaults to ten seeds (1001–1010), accepts `--seed-count`
from 1 to 20 and has a 300-second wall-clock deadline. Automatic post-training
evaluation retains its original two seeds. Seed inputs reject duplicates, empty
sets, booleans, out-of-range values and oversized sets. Reports include the
installed 90-action table plus named controls for the dominant action. Output
uses `diagnostics-N-seeds.json`, preserving the earlier `diagnostics.json`.

Completed the ten-seed comparison: 480 episodes, each capped at 150 decisions,
with no training or checkpoint modifications. Latest sampled approach: 12 touches
in 6/20 episodes (30%), mean closest distance 426 units. Initial sampled approach:
14 touches in 7/20 episodes (35%), mean closest distance 414 units. Retained
sampled approach episode contact rates were 20%, 15%, 30%, 30%, 30% in checkpoint
order. Initial sampled kickoff: four touches in 2/20 episodes; all retained
trained sampled kickoff policies: zero. All candidates had zero argmax contacts.
Broader coverage therefore finds sampled contacts missed by the two-seed report,
without demonstrating improvement or statistically reliable regression. Both
team assignments share scenario seeds; these are not independent training runs.

Decoded action 80: throttle=0, steer=0, pitch=1, yaw=0, roll=1, jump=0, boost=0,
handbrake=0. This provides aerial orientation inputs with no driving input from
a grounded start. Near-uniform probabilities combined with this repeated argmax
choice explain ineffective deterministic control, not a concentrated probability
distribution. Next calibrate evaluation with a simple driving/contact baseline
and compare controlled learning experiments before changing action space/rewards.

Validation: all 23 tests passed after extending seed validation and custom-seed
episode-count checks. The complete ten-seed report succeeded within its deadline;
`git diff --check` passed. Dependencies and device selection were unchanged.

### Observation-only drive-to-ball calibration

Added `agents/baselines.py` with a fixed ground-driving controller using only the
pinned DefaultObs ball position, own position and forward vector. It computes a
signed heading error, drives forward, steers discretely and boosts when aligned
and farther than 800 units. It uses the same installed 90-action table, has no
simulator imports/state access, performs no training and returns one-hot action
probabilities. A structural ActionPolicy interface lets the evaluator accept
learned and fixed policies through the same path.

The diagnostic CLI includes this controller in full reports and supports
`--baseline-only`, saving `calibration-N-seeds.json`. Default full diagnostics
now cap at 520 episodes; baseline-only uses 40, each at most 150 steps, under the
same 300-second wall-clock deadline. Existing reports/checkpoints are preserved.

Completed ten-seed CPU calibration against the run's original deterministic
opponent: kickoff 20/20 episodes with contact, 46 touches, mean minimum center
distance 184.9 units; approach 20/20 with contact, 62 touches, distance 182.1.
Latest learned results on the same seeds/opponent: no argmax contacts; sampled
kickoff 0/20, approach 6/20 episodes with contact. This validates achievable and
detectable contact through the current interface; it neither diagnoses PPO's
weak learning nor establishes competitive skill. No learning settings changed.

Added tests for signed steering, team inversion, one-hot discrete controls,
distance-based boost gating, nonfinite observations and bounded simulator contact
for both teams in both starts. Next compare controlled learning hypotheses with
this calibration retained as an evaluation sanity check.

Validation after calibration: all 25 tests passed, including the new bounded
contact and steering tests; CPU forward/backward/setup and pip check passed.
`git diff --check` passed. CUDA was not retested for this CPU-only controller
change; the earlier bounded CUDA learner/resume validation remains recorded above.

## Controlled entropy ablation — 2026-11-01

Hypothesis: the default entropy bonus is limiting early policy specialization.
Added validated `training.entropy_coef` (default 0.005, finite range 0–0.1) passed
directly to the pinned learner. Baseline configs retain their previous behavior.
Added fresh paired CPU configs with seed 42, identical approach environment and
40,960-transition budget (ten iterations, two epochs, 300-second deadline each).
Only the output name and entropy coefficient differ (0.005 vs zero). No optimizer,
reward, action space, model architecture or dependency changes were made.

Runs completed successfully with checkpoint/inference verification, automatic
evaluation and worker cleanup:

- Control: `logs/ppo/approach-entropy-control-20261101T201825Z-afe60cf0/`.
- Treatment: `logs/ppo/approach-no-entropy-20261101T201926Z-f1df774e/`.

`training/audits.py` now verifies collected action/log-probability alignment before
each update and records reward, advantage, target and buffer statistics without
changing tensors, RNG streams or policy parameters. Maximum discrepancy was
4.768e-7 in both runs. Buffer sizes and finite data checks passed. This checks
alignment, not GAE mathematical correctness or adequacy of the learning signal.
The control reproduced the earlier approach run's 73 touches and one goal.
The no-entropy run collected 72 touches and one goal across both training cars.

`scripts/compare_entropy.py` validates identical configs except name/coefficient,
matching initial policy weights, successful fresh 40,960-transition results,
then compares final policies/shared initial/driving baseline with ten fixed seeds
and both teams. It runs no training, caps each episode at 150 steps and the whole
comparison at 300 seconds. The 280-episode comparison passed and saved metadata,
source training reports and all per-episode results in
`logs/experiments/entropy-7bbd57f5.json`.

| Candidate | Sampled approach contact | Touches | Mean closest distance |
|---|---:|---:|---:|
| Initial | 7/20 | 14 | 414.4 |
| Control, entropy 0.005 | 6/20 | 12 | 426.1 |
| Treatment, entropy 0 | 6/20 | 12 | 424.5 |

Both trained policies had zero contacts in kickoff (either mode) and approach
argmax. The fixed driving baseline made contact in 20/20 episodes of each start.
Final training entropy was 4.49214 vs 4.49187 nats; final mean KL was 6.12e-7 vs
7.78e-7. Removing entropy did not produce useful specialization in this bounded
test. Retain the default coefficient. These are one-seed short-budget results,
not a general rejection of entropy regularization or a significant regression.

Next hypothesis: reward/advantage signal scale or update strength limits learning;
test one factor at a time with matched fresh seeds and the same evaluation. No
cause of the weak learning has been established. Do not extend training or change
several learning settings based on the contact totals alone.

Validation: all 26 tests passed, including corrupt log-probability rejection,
audit RNG/weight/gradient preservation and entropy configuration validation. CPU
setup/forward/backward and pip check passed; diff whitespace checks passed.
CUDA was not rerun for this CPU-only experiment. No baseline pins changed.
