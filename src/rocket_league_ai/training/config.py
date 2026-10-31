"""Validated TOML settings for a deliberately small PPO smoke run."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 42


@dataclass(frozen=True)
class EnvironmentConfig:
    action_repeat: int = 8
    episode_seconds: float = 10.0
    no_touch_seconds: float = 5.0
    reward_profile: str = "sparse"
    start_mode: str = "kickoff"


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cpu"
    timesteps: int = 256
    batch_size: int = 64
    minibatch_size: int = 32
    epochs: int = 1
    iterations: int = 1
    timeout_seconds: int = 120


@dataclass(frozen=True)
class SmokeConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def load_config(path: Path) -> SmokeConfig:
    """Load settings, rejecting unknown keys and oversized smoke workloads."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    section_types = {
        "experiment": ExperimentConfig, "environment": EnvironmentConfig,
        "training": TrainingConfig,
    }
    if data.keys() - section_types.keys():
        raise ValueError("Unknown configuration section")
    sections = {}
    for name, cls in section_types.items():
        values = data.get(name, {})
        if not isinstance(values, dict) or values.keys() - asdict(cls()).keys():
            raise ValueError(f"Unknown or invalid settings in {name}")
        sections[name] = cls(**values)
    config = SmokeConfig(**sections)
    experiment, env, training = config.experiment, config.environment, config.training
    if not isinstance(experiment.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", experiment.name):
        raise ValueError("experiment.name must contain 1–64 letters, digits, underscores or hyphens")
    _integer(experiment.seed, "seed", 0, 2**32 - 1)
    _integer(env.action_repeat, "action_repeat", 1, 120)
    for name in ("episode_seconds", "no_touch_seconds"):
        value = getattr(env, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 300 or (name == "episode_seconds" and value == 0):
            raise ValueError(f"{name} must be finite, <=300 and positive (no_touch_seconds may be zero)")
    if env.reward_profile not in ("sparse", "approach_v1"):
        raise ValueError("Unknown reward profile")
    if env.start_mode not in ("kickoff", "approach"):
        raise ValueError("Unknown start mode")
    if training.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda; PPO MPS is unverified")
    for name, low, high in (
        ("timesteps", 32, 4096), ("batch_size", 2, 4096),
        ("minibatch_size", 2, 4096), ("epochs", 1, 4), ("iterations", 1, 10), ("timeout_seconds", 10, 300),
    ):
        _integer(getattr(training, name), name, low, high)
    if training.timesteps % 2:
        raise ValueError("timesteps must be even for two agents")
    if training.batch_size > training.timesteps or training.batch_size % training.minibatch_size:
        raise ValueError("batch_size must fit timesteps and be divisible by minibatch_size")
    return config
