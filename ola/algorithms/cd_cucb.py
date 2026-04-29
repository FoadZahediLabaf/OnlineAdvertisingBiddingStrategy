"""
Change-detector Combinatorial UCB algorithm for Requirement 4.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Implement the change-detector version of Combinatorial-UCB for slightly
    non-stationary multi-campaign environments.

What to implement here:
    - Combinatorial-UCB action selection.
    - Change detection on reward/cost observations.
    - Reset or adaptation logic after detected changes.
    - Budget-aware action selection.
    - Compatibility with campaign conflict constraints.

Do not:
    - Add sliding-window logic here.
    - Add primal-dual logic here.
    - Add experiment-running code here.
    - Change the standard CSV result format here.
"""

from typing import Any


class ChangeDetectorCUCB:
    """Change-detector CUCB learner skeleton for Requirement 4."""

    def __init__(self) -> None:
        # TODO(Alessandro): initialize detector parameters, estimates, budget, and action space.
        # Simone may also update this when helping Alessandro.
        pass

    def select_action(self) -> Any:
        # TODO(Alessandro): select feasible campaign/bid combination using current estimates.
        # Simone may also update this when helping Alessandro.
        pass

    def update(self, action: Any, reward: float, cost: float) -> None:
        # TODO(Alessandro): update estimates and run change detection.
        # Simone may also update this when helping Alessandro.
        pass
