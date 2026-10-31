"""Discrete controls using the upstream 90-action lookup table."""
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction


def build_action_parser(action_repeat: int = 8) -> RepeatAction:
    """Repeat each chosen action for action_repeat physics ticks."""
    return RepeatAction(LookupTableAction(), repeats=action_repeat)
