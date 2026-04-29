

"""
Adversarial / highly non-stationary multi-campaign environment for Requirement 3.

Owner:
    Mengyao

Support:
    Simone may edit this file when actively helping Mengyao.

Purpose:
    Implement the environment used to test the best-of-both-worlds primal-dual method.

What to implement here:
    - Non-stochastic sequence of highest competing bids for each campaign.
    - Full-feedback support: the algorithm may observe the highest competing bid for each auction.
    - Multi-campaign reward, cost, win/loss, and budget updates.
    - Compatibility with campaign conflict constraints if needed.

Do not:
    - Add Requirement 1 single-campaign logic here.
    - Add Requirement 2 stochastic-only environment logic here.
    - Add Requirement 4 sliding-window or change-detector logic here.
    - Change the standard step output format.
"""

from typing import Any, Dict

from ola.environments.base import BaseEnvironment


class AdversarialMultiCampaignEnvironment(BaseEnvironment):
    """Environment skeleton for Requirement 3."""

    def __init__(self) -> None:
        # TODO(Mengyao): initialize campaign values, bid set, adversarial sequence, budget, and horizon.
        # Simone may also update this when helping Mengyao.
        pass

    def reset(self) -> Dict[str, Any]:
        # TODO(Mengyao): reset budget, round counter, and stored history.
        # Simone may also update this when helping Mengyao.
        pass

    def step(self, action: Any) -> Dict[str, Any]:
        # TODO(Mengyao): execute one multi-campaign adversarial auction round.
        # Simone may also update this when helping Mengyao.
        pass