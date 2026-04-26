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

from typing import Any, Sequence
import numpy as np



class BudgetedUCB:
    """Budget-aware UCB learner skeleton for Requirement 1.
    In Requirement 1, each action is one possible bid.

    This learner estimates both reward and cost for each bid."""

    def __init__(
            self,
            actions: Sequence[float],
            horizon: int,
            initial_budget: float,) -> None:

        if len(actions)==0:
            raise ValueError("Must have at least one action.")  
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if initial_budget < 0:
            raise ValueError("Initial budget must be non-negative.")
        self.actions = np.array(actions, dtype=float)
        self.horizon = int(horizon)
        self.initial_budget = float(initial_budget)
        self.budget_left = float(self.initial_budget)
        self.n_actions = len(self.actions)
        self.counts = np.zeros(self.n_actions, dtype=int)  
        self.total_rewards = np.zeros(self.n_actions, dtype=float)
        self.total_costs = np.zeros(self.n_actions, dtype=float)
        self.t = 0

    def select_action(self) -> Any:
        if self.budget_left <=0:
            return 0.0
        for index in range(self.n_actions):
            if self.counts[index] == 0 and self.actions[index] <= self.budget_left:
                return float(self.actions[index])
        
            
        remaining_rounds = max(1, self.horizon - self.t)
        budget_per_remaining_round = self.budget_left / remaining_rounds
        empirical_rewards = self.total_rewards / np.maximum(1, self.counts)
        empirical_costs = self.total_costs / np.maximum(1, self.counts)
        confidence_widths = np.sqrt(
            2 * np.log(self.horizon) / np.maximum(1, self.counts)
        )
        reward_ucbs = empirical_rewards + confidence_widths

        cost_lcbs = np.maximum(0.0, empirical_costs - confidence_widths)

        feasible_indices = np.where(

            (self.actions <= self.budget_left)

            & (cost_lcbs <= budget_per_remaining_round)

        )[0]

        if len(feasible_indices) == 0:

            return 0.0

        best_feasible_position = int(np.argmax(reward_ucbs[feasible_indices]))

        best_index = int(feasible_indices[best_feasible_position])

        return float(self.actions[best_index])
    def update(self, action: Any, reward: float, cost: float) -> None:
        """Update reward, cost, and remaining budget after one round."""

        action_value = float(action)

        matching_indices = np.where(np.isclose(self.actions, action_value))[0]

        if len(matching_indices) == 0:

            raise ValueError(f"Unknown action: {action}")

        index = int(matching_indices[0])

        self.counts[index] += 1

        self.total_rewards[index] += float(reward)

        self.total_costs[index] += float(cost)

        self.budget_left = max(0.0, self.budget_left - float(cost))

        self.t += 1