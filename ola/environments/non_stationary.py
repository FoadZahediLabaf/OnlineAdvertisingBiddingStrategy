

"""
Slightly non-stationary multi-campaign environment for Requirement 4.

Owner:
    Mengyao + Simone

Purpose:
    Implement the non-stationary environment used to compare adaptive algorithms.

What to implement here:
    - Time intervals with different stochastic distributions.
    - Joint distribution over highest competing bids for each campaign in each interval.
    - Multi-campaign reward, cost, win/loss, and budget updates.
    - Compatibility with campaign conflict constraints if needed.

Do not:
    - Add Requirement 1 single-campaign logic here.
    - Add Requirement 2 stationary-only environment logic here.
    - Add Requirement 3 adversarial full-feedback logic here.
    - Add sliding-window or change-detector algorithm logic here.
    - Change the standard step output format.
"""

from typing import Any, Dict

from ola.environments.base import BaseEnvironment


class NonStationaryMultiCampaignEnvironment(BaseEnvironment):
    """Environment skeleton for Requirement 4."""

    def __init__(self) -> None:
        # TODO(Mengyao + Simone): initialize intervals, distributions, campaign values, budget, and horizon.
        pass

    def reset(self) -> Dict[str, Any]:
        # TODO(Mengyao + Simone): reset budget, round counter, interval state, and stored history.
        pass

    def step(self, action: Any) -> Dict[str, Any]:
        # TODO(Mengyao + Simone): execute one multi-campaign non-stationary auction round.
        pass