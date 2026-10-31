"""Upstream observations, padded for one car per team."""
from rlgym.rocket_league.obs_builders import DefaultObs


def build_obs_builder() -> DefaultObs:
    """Return the fixed 92-value observation builder for 1v1."""
    return DefaultObs(zero_padding=1)
