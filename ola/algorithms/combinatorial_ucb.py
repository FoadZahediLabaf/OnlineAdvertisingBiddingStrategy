"""
Combinatorial UCB algorithm for Requirement 2.

Owner:
    Luigi

Purpose:
    Implement a Combinatorial-UCB bidding strategy with budget constraints for
    multiple campaigns.

What to implement here:
    - UCB estimates for campaign/bid choices.
    - Selection of a feasible set of campaigns and bids.
    - Budget-aware action selection.
    - Compatibility with the conflict graph.

Do not:
    - Add single-campaign UCB1 logic here.
    - Add primal-dual logic here.
    - Add sliding-window or change-detector logic here.
    - Add experiment-running code here.
"""

from typing import Any


class CombinatorialUCB:
    """Combinatorial-UCB learner skeleton for Requirement 2."""

    def __init__(self) -> None:
        # TODO(Luigi): initialize campaign arms, confidence estimates, budget, and graph constraints.
        pass

    def select_action(self) -> Any:
        # TODO(Luigi): select feasible campaign/bid combination using optimism and budget constraints.
        pass

    def update(self, action: Any, reward: float, cost: float) -> None:
        # TODO(Luigi): update estimates after observing reward/cost feedback.
        pass
