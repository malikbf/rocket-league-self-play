"""Bounded iterations, checkpoints and evaluation using the pinned PPO learner."""
from dataclasses import asdict
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import TYPE_CHECKING

from rocket_league_ai.training.config import EnvironmentConfig, SmokeConfig
from rocket_league_ai.training.devices import select_device

if TYPE_CHECKING:
    from rlgym_ppo.util import RLGymV2GymWrapper


def build_ppo_env(settings: EnvironmentConfig, seed: int, live_path: Path | None = None) -> "RLGymV2GymWrapper":
    """Picklable factory for the single smoke worker, including kickoff seeding."""
    import numpy as np
    from rlgym_ppo.util import RLGymV2GymWrapper
    from rocket_league_ai.environments.factory import build_env

    random.seed(seed)
    np.random.seed(seed)
    env = build_env(**asdict(settings))
    try:
        if live_path is not None:
            from rocket_league_ai.training.live import LiveEnv
            return LiveEnv(env, live_path, settings.action_repeat)
        return RLGymV2GymWrapper(env)
    except BaseException:
        env.close()
        raise


def _write_metadata(config: SmokeConfig, output: Path, root: Path, device: str) -> None:
    import torch

    def capture(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=root, text=True, timeout=15).strip()

    metadata = {
        "config": asdict(config), "device": device,
        "commit": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "workers": 1, "policy_layers": [32, 32], "critic_layers": [32, 32],
        "standardize_obs": False, "standardize_returns": False,
        "online_logging": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "requirements.txt").write_text(
        capture([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8",
    )
    if device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(0)
        try:
            metadata["driver"] = capture([
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
            ])
        except (OSError, subprocess.SubprocessError):
            metadata["driver"] = "unavailable"
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_smoke(config: SmokeConfig, output: Path, root: Path, resume: Path | None = None, live: bool = False) -> None:
    """Collect/update bounded batches, verify checkpoints, evaluate, and stop workers.

    Called only in the supervised subprocess. Avoid Learner.learn(): its terminal
    polling is interactive and its exception handler suppresses training failures.
    """
    import numpy as np
    import torch
    from rlgym_ppo import Learner
    from rocket_league_ai.agents.policies import load_policy
    from rocket_league_ai.evaluation.runner import evaluate
    from rocket_league_ai.training.checkpoints import (
        restore_checkpoint, save_checkpoint, verify_checkpoint,
    )

    from rocket_league_ai.training.metrics import TrainingMetrics, summarize

    settings = config.training
    device = select_device(settings.device)
    torch.set_num_threads(1)
    _write_metadata(config, output, root, device)
    learner = None
    result = None
    try:
        learner = Learner(
            partial(build_ppo_env, config.environment, config.experiment.seed,
                    output / "live.json" if live else None),
            n_proc=1, min_inference_size=1, render=False, metrics_logger=TrainingMetrics(),
            timestep_limit=settings.timesteps, ts_per_iteration=settings.timesteps,
            exp_buffer_size=settings.timesteps,
            policy_layer_sizes=(32, 32), critic_layer_sizes=(32, 32),
            ppo_batch_size=settings.batch_size, ppo_minibatch_size=settings.minibatch_size,
            ppo_epochs=settings.epochs, standardize_obs=False, standardize_returns=False,
            random_seed=config.experiment.seed, device=device,
            log_to_wandb=False, load_wandb=False, checkpoint_load_folder=None,
            checkpoints_save_folder=str(output / "checkpoints"), add_unix_timestamp=False,
        )
        (output / "learner_config.json").write_text(
            json.dumps(learner.config, indent=2), encoding="utf-8",
        )
        # Preserve the exact original opponent across all resumed runs.
        reference_path = output / "reference_policy.pt"
        if resume is not None:
            import shutil
            shutil.copy2(resume.parent.parent / "reference_policy.pt", reference_path)
            restore_checkpoint(learner, resume, config)
            verify_checkpoint(learner, resume)
        else:
            torch.save(learner.ppo_learner.policy.state_dict(), reference_path)
        start_steps = learner.agent.cumulative_timesteps
        start_updates = learner.ppo_learner.cumulative_model_updates
        (output / "resume.json").write_text(json.dumps({
            "checkpoint": str(resume.resolve()) if resume else None,
            "starting_steps": start_steps, "starting_updates": start_updates,
        }, indent=2), encoding="utf-8")
        reports = []
        for iteration in range(settings.iterations):
            experience, collected_metrics, steps, _ = learner.agent.collect_timesteps(settings.timesteps)
            if not settings.timesteps <= steps <= settings.timesteps + 2:
                raise RuntimeError(f"Unexpected transition count: {steps}")
            if not all(np.isfinite(array).all() for array in experience):
                raise RuntimeError("Nonfinite rollout data")
            gameplay = summarize(collected_metrics)
            gameplay["mean_reward_per_transition"] = float(np.asarray(experience[3]).mean())
            learner.add_new_experience(experience)
            report = learner.ppo_learner.learn(learner.experience_buffer)
            if not all(np.isfinite(value) for value in report.values()):
                raise RuntimeError("Nonfinite PPO metrics")
            for key in ("Policy Update Magnitude", "Value Function Update Magnitude"):
                if report[key] <= 0:
                    raise RuntimeError(f"No parameter change: {key}")
            for model in (learner.ppo_learner.policy, learner.ppo_learner.value_net):
                if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                    raise RuntimeError("Nonfinite model parameters")
            learner.epoch += 1
            checkpoint = save_checkpoint(learner, config)
            verify_checkpoint(learner, checkpoint)
            # Independent CPU model must reproduce probabilities, not just actions.
            restored_policy = load_policy(checkpoint / "PPO_POLICY.pt")
            with torch.no_grad():
                probe = experience[0][:8]
                expected = learner.ppo_learner.policy.get_output(probe).cpu()
                actual = restored_policy.get_output(probe)
                if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-5):
                    raise RuntimeError("Reloaded policy inference differs")
            reports.append({"iteration": iteration + 1, "steps": steps, "metrics": report,
                            "checkpoint": str(checkpoint), "gameplay": gameplay})
            (output / "progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        opponent = load_policy(reference_path)
        evaluation = evaluate(restored_policy, opponent)
        baseline = evaluate(opponent, opponent)
        (output / "evaluation.json").write_text(
            json.dumps({"trained": evaluation, "initial": baseline}, indent=2), encoding="utf-8",
        )
        if config.environment.start_mode == "approach":
            (output / "practice-evaluation.json").write_text(json.dumps({
                "trained": evaluate(restored_policy, opponent, start_mode="approach"),
                "initial": evaluate(opponent, opponent, start_mode="approach"),
            }, indent=2), encoding="utf-8")
        result = {"steps": learner.agent.cumulative_timesteps,
                  "new_steps": learner.agent.cumulative_timesteps - start_steps,
                  "starting_updates": start_updates, "iterations": reports,
                  "device": device, "checkpoint_verified": True, "live_viewer": live,
                  "evaluation": evaluation, "initial_evaluation": baseline}

    finally:
        try:
            if learner is not None:
                learner.cleanup()
        finally:
            # Also covers partial learner initialization failures.
            for child in mp.active_children():
                child.terminate()
                child.join(timeout=2)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=2)
            if mp.active_children():
                raise RuntimeError("Training workers did not exit")
    result["workers_cleaned_up"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: {result['new_steps']} new transitions, checkpoint/inference verified, "
          "evaluation completed; workers closed")
