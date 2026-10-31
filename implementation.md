# Implementation status

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
- Verify on Windows/CUDA.
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
CPU training, saving, resuming and initial evaluation are verified. Windows/CUDA,
playing strength and historical self-play are not. Resume restarts episodes and RNG
streams; it is not an exact continuation of an uninterrupted run.
