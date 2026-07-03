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
    """LP-based budget-aware UCB learner for Requirement 1.

    In Requirement 1, each action is one possible bid. The learner first tries
    each affordable bid once, then estimates both reward and cost for each bid,
    builds reward UCBs, solves the budgeted LP using empirical cost estimates,
    and samples one bid from the resulting distribution.
    """

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
        self.rng = np.random.default_rng(0)

    def select_action(self) -> Any:
        """Select one bid by solving the optimistic budgeted LP and sampling."""
        if self.budget_left <= 0:
            return 0.0

        remaining_rounds = max(1, self.horizon - self.t)
        budget_per_remaining_round = self.budget_left / remaining_rounds

        for index, bid in enumerate(self.actions):
            if self.counts[index] == 0 and bid <= self.budget_left:
                return float(bid)

        empirical_rewards = self.total_rewards / np.maximum(1, self.counts)
        empirical_costs = self.total_costs / np.maximum(1, self.counts)
        current_round = self.t + 1
        confidence_widths = np.sqrt(
            0.1 * np.log(current_round) / np.maximum(1, self.counts)
        )

        reward_ucbs = empirical_rewards + confidence_widths
        cost_estimates = empirical_costs

        distribution = self._solve_optimistic_budgeted_lp(
            reward_ucbs=reward_ucbs,
            cost_estimates=cost_estimates,
            budget_per_remaining_round=budget_per_remaining_round,
        )

        if np.sum(distribution) <= 0.0:
            return 0.0

        selected_index = int(self.rng.choice(self.n_actions, p=distribution))
        return float(self.actions[selected_index])

    def _solve_optimistic_budgeted_lp(
        self,
        reward_ucbs: np.ndarray,
        cost_estimates: np.ndarray,
        budget_per_remaining_round: float,
    ) -> np.ndarray:
        """Solve the optimistic LP over bid distributions for Req 1.

        The LP is:
            maximize    sum_b gamma_b reward_ucb(b)
            subject to  sum_b gamma_b cost_estimate(b) <= budget_per_remaining_round
                        sum_b gamma_b = 1
                        gamma_b >= 0

        Since Req 1 has only one budget constraint, an optimal extreme point has
        support on at most two bids. We enumerate feasible single bids and all
        two-bid mixtures.
        """
        affordable_indices = np.where(self.actions <= self.budget_left)[0]
        if len(affordable_indices) == 0:
            return np.zeros(self.n_actions, dtype=float)

        best_distribution = np.zeros(self.n_actions, dtype=float)
        best_objective = -float("inf")
        tie_tolerance = 1e-12

        def register_candidate(candidate_distribution: np.ndarray, objective: float) -> None:
            nonlocal best_distribution, best_objective
            if objective > best_objective + tie_tolerance:
                best_objective = objective
                best_distribution = candidate_distribution.copy()
            elif abs(objective - best_objective) <= tie_tolerance:
                best_distribution = best_distribution + candidate_distribution

        for index in affordable_indices:
            cost = float(cost_estimates[index])
            if cost <= budget_per_remaining_round:
                objective = float(reward_ucbs[index])
                candidate_distribution = np.zeros(self.n_actions, dtype=float)
                candidate_distribution[index] = 1.0
                register_candidate(candidate_distribution, objective)

        for left_position, left_index in enumerate(affordable_indices):
            for right_index in affordable_indices[left_position + 1:]:
                cost_left = float(cost_estimates[left_index])
                cost_right = float(cost_estimates[right_index])

                if cost_left == cost_right:
                    continue

                probability_left = (
                    budget_per_remaining_round - cost_right
                ) / (cost_left - cost_right)
                probability_right = 1.0 - probability_left

                if 0.0 <= probability_left <= 1.0 and 0.0 <= probability_right <= 1.0:
                    objective = (
                        probability_left * float(reward_ucbs[left_index])
                        + probability_right * float(reward_ucbs[right_index])
                    )
                    candidate_distribution = np.zeros(self.n_actions, dtype=float)
                    candidate_distribution[left_index] = probability_left
                    candidate_distribution[right_index] = probability_right
                    register_candidate(candidate_distribution, objective)

        if np.sum(best_distribution) <= 0.0:
            zero_bid_indices = np.where(np.isclose(self.actions, 0.0))[0]
            if len(zero_bid_indices) > 0:
                best_distribution[int(zero_bid_indices[0])] = 1.0

        return self._normalize_distribution(best_distribution)

    def _normalize_distribution(self, distribution: np.ndarray) -> np.ndarray:
        """Return a valid probability distribution."""
        total_probability = float(np.sum(distribution))
        if total_probability <= 0.0:
            return distribution
        return distribution / total_probability

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