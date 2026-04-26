"""
Single-campaign stochastic environment for Requirement 1.

Owner:
    Foad

Purpose:
    Implement the single-campaign environment used in Requirement 1.

Do not:
    - Add multi-campaign logic here.
    - Add conflict-graph logic here.
    - Add adversarial or non-stationary environment logic here.
"""

from typing import Any, Dict

from ola.environments.base import BaseEnvironment


class SingleCampaignEnvironment(BaseEnvironment):
    """Environment skeleton for Requirement 1."""

    def __init__(self) -> None:
        # TODO(Foad): initialize campaign value, competing-bid distribution, budget, and horizon.
        pass

    def reset(self) -> Dict[str, Any]:
        # TODO(Foad): reset budget, round counter, and stored history.
        pass

    def step(self, action: Any) -> Dict[str, Any]:
        # TODO(Foad): execute one auction round and return the standard outcome dictionary.
        pass