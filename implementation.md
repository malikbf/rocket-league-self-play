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

## Next / not yet verified

- Connect PPO and validate training configuration.
- Complete a bounded learning update and checkpoint save/reload.
- Verify on Windows/CUDA.
- Add evaluation, then historical-opponent self-play.

## Run checks

From the repository root, with the virtual environment active:

```bash
python -m unittest discover -s tests -v
python scripts/verify_environment.py
python scripts/verify_setup.py --device cpu
python -m pip check
```

The standalone environment check also runs as part of the unit tests.
No PPO training or playing strength has been validated yet.
