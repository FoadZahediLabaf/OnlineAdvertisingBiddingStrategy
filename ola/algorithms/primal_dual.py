

"""
Primal-dual bidding algorithm for Requirement 3.

Owner:
    Mengyao

Support:
    Simone may edit this file when actively helping Mengyao.

Purpose:
    Implement the best-of-both-worlds primal-dual method with budget constraints.

What to implement here:
    - Primal regret minimizer for bidding decisions.
    - Dual variable / pacing multiplier updates for the budget constraint.
    - Full-feedback update logic using observed competing bids.
    - Budget-aware action selection for multiple campaigns.

Do not:
    - Add standard UCB1 logic here.
    - Add Combinatorial-UCB logic here.
    - Add sliding-window or change-detector logic here.
    - Add experiment-running code here.
"""

from typing import Any


class PrimalDualBidder:
    """Primal-dual learner skeleton for Requirement 3."""

    def __init__(self) -> None:
        # TODO(Mengyao): initialize primal learner, dual variable, budget, and action space.
        # Simone may also update this when helping Mengyao.
        pass

    def select_action(self) -> Any:
        # TODO(Mengyao): select bids using the current primal strategy and dual variable.
        # Simone may also update this when helping Mengyao.
        pass

    def update(self, action: Any, reward: float, cost: float, feedback: Any) -> None:
        # TODO(Mengyao): update primal learner and dual variable using full feedback.
        # Simone may also update this when helping Mengyao.
        pass