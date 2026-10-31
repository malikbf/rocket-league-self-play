"""Device selection with an explicit boundary around unverified PPO MPS support."""


def select_device(requested: str = "auto", *, for_ppo: bool = True) -> str:
    """Select hardware; MPS is available only for standalone PyTorch work."""
    import torch

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "mps" and for_ppo:
        raise ValueError("RLGym-PPO MPS support is unverified; use cpu for PPO.")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return "cuda"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if not for_ppo and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
