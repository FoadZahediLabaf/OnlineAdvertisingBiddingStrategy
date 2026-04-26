"""
Budgeted UCB algorithm for Requirement 1.

Owner:
    Foad

Purpose:
    Implement a UCB-style learner that handles the budget constraint for the
    single-campaign setting.

Do not:
    - Add standard budget-free UCB1 logic here unless it is reused cleanly.
    - Add multi-campaign logic here.
    - Add experiment-running code here.
    - Change result/output formats here.
"""

from typing import Any


class BudgetedUCB:
    """Budget-aware UCB learner skeleton for Requirement 1."""

    def __init__(self) -> None:
        # TODO(Foad): initialize bids, counts, reward estimates, cost estimates, and budget.
        pass

    def select_action(self) -> Any:
        # TODO(Foad): select a feasible bid/action using optimistic reward and budget information.
        pass

    def update(self, action: Any, reward: float, cost: float) -> None:
        # TODO(Foad): update reward estimates, cost estimates, and remaining budget.
        pass
