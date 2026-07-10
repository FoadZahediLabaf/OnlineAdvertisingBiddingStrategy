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

from typing import Any, Sequence
import numpy as np
class UCB1:
    """Standard UCB1 learner for a finite action set.

    In Requirement 1, each action is one possible bid.
    This learner ignores the budget constraint.
    """

    def __init__(self, actions: Sequence[float], horizon: int) -> None:
        if len(actions) == 0:
            raise ValueError("Action set must be non-empty.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        
        self.actions = np.array(actions, dtype=float)
        self.horizon = int(horizon)
        self.n_actions = len(self.actions)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.total_rewards = np.zeros(self.n_actions, dtype=float)
     


    def select_action(self) -> Any:
        """Select an action using the UCB1 algorithm rule."""
        for index in range(self.n_actions):
            if self.counts[index] == 0:
                return float(self.actions[index])

        t = int(np.sum(self.counts)) + 1  # current round t, not fixed horizon T
        empirical_means = self.total_rewards / self.counts
        confidence_widths = np.sqrt(2.0 * np.log(t) / self.counts)
        upper_confidence_values = empirical_means + confidence_widths
        best_index = int(np.argmax(upper_confidence_values))
        return float(self.actions[best_index])  


    def update(self, action: Any, reward: float) -> None:
        """Update the learner after observing the reward of selected action."""
        action_value = float(action)
        matching_indices = np.where(np.isclose(self.actions, action_value))[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Unknown action: {action}")

        index = int(matching_indices[0])
        self.counts[index] += 1
        self.total_rewards[index] += float(reward)
