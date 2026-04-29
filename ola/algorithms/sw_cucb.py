"""
Sliding-window Combinatorial UCB algorithm for Requirement 4.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Implement the sliding-window version of Combinatorial-UCB for slightly
    non-stationary multi-campaign environments.

What to implement here:
    - Combinatorial-UCB action selection using only recent observations.
    - Sliding-window reward and cost estimates.
    - Budget-aware action selection.
    - Compatibility with campaign conflict constraints.

Do not:
    - Add standard stationary Combinatorial-UCB logic here unless reused cleanly.
    - Add change-detector logic here.
    - Add primal-dual logic here.
    - Add experiment-running code here.
"""

from typing import Any


class SlidingWindowCUCB:
    """Sliding-window CUCB learner skeleton for Requirement 4."""

    def __init__(self) -> None:
        # TODO(Alessandro): initialize window size, estimates, budget, and action space.
        # Simone may also update this when helping Alessandro.
        pass

    def select_action(self) -> Any:
        # TODO(Alessandro): select feasible campaign/bid combination using recent observations.
        # Simone may also update this when helping Alessandro.
        pass

    def update(self, action: Any, reward: float, cost: float) -> None:
        # TODO(Alessandro): update sliding-window statistics after observing feedback.
        # Simone may also update this when helping Alessandro.
        pass
