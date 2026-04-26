"""
Standard UCB1 algorithm for Requirement 1.

Owner:
    Foad

Purpose:
    Implement the budget-free UCB1 learner for a finite set of bids.

Do not:
    - Add budgeted UCB logic here.
    - Add multi-campaign logic here.
    - Add experiment-running code here.
"""

from typing import Any


class UCB1:
    """Standard UCB1 learner skeleton for a finite action set."""

    def __init__(self) -> None:
        # TODO(Foad): initialize arms, counts, empirical means, and horizon if needed.
        pass

    def select_action(self) -> Any:
        # TODO(Foad): select the bid/action with the highest upper confidence bound.
        pass

    def update(self, action: Any, reward: float) -> None:
        # TODO(Foad): update counts and empirical reward estimate for the selected action.
        pass
