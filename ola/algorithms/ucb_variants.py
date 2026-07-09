"""
Exploration-constant variants of UCB1 for Requirement 1.

Owner:
    Foad

Purpose:
    Implement two supplementary bidding strategies, evaluated alongside the
    canonical UCB1 (ola/algorithms/ucb1.py), for the "how should we tune
    exploration" discussion:

      - UCB1Tuned: Auer, Cesa-Bianchi & Fischer (2002)'s variance-adaptive
        refinement of UCB1. Per-arm, data-driven, no exploration constant to
        pick by hand.

      - ExpectedRewardScaledUCB: the exploration-constant rescaled. Standard UCB1 structure, but the "2" inside the
        confidence bound is rescaled based on the best achievable EXPECTED
        reward under the environment.



"""

from typing import Any, Sequence
import numpy as np


class UCB1Tuned:
    """UCB1-Tuned (Auer, Cesa-Bianchi & Fischer, 2002).

    Same structure as UCB1, but the fixed 1/4 worst-case variance proxy is
    replaced with a per-arm empirical variance estimate (still capped at
    1/4, the maximum possible variance of any reward bounded in [0, 1]):

        index_i(t) = mean_i + sqrt( (ln t / n_i) * min(1/4, V_i(n_i)) )
        V_i(n_i)   = empirical_variance_i(n_i) + sqrt(2 ln t / n_i)

    Exploration shrinks specifically for arms whose observed rewards are
    consistently close together, rather than shrinking uniformly for every
    arm the way a single hand-picked constant would. Requires rewards
    bounded in [0, 1] (true here since reward = value - bid with value=1).

    No formal finite-time regret bound is proven for this variant (unlike
    UCB1), but it is well documented to perform very well empirically --
    that tradeoff is worth one sentence in the writeup.
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
        self.total_squared_rewards = np.zeros(self.n_actions, dtype=float)

    def select_action(self) -> Any:
        for index in range(self.n_actions):
            if self.counts[index] == 0:
                return float(self.actions[index])
        t = int(np.sum(self.counts)) + 1
        empirical_means = self.total_rewards / self.counts
        empirical_second_moments = self.total_squared_rewards / self.counts
        empirical_variances = empirical_second_moments - empirical_means ** 2
        empirical_variances = np.maximum(0.0, empirical_variances)
        variance_ucb = empirical_variances + np.sqrt(2.0 * np.log(t) / self.counts)
        capped_variance = np.minimum(0.25, variance_ucb)
        confidence_widths = np.sqrt((np.log(t) / self.counts) * capped_variance)
        upper_confidence_values = empirical_means + confidence_widths
        best_index = int(np.argmax(upper_confidence_values))
        return float(self.actions[best_index])

    def update(self, action: Any, reward: float) -> None:
        action_value = float(action)
        matching_indices = np.where(np.isclose(self.actions, action_value))[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Unknown action: {action}")
        index = int(matching_indices[0])
        self.counts[index] += 1
        self.total_rewards[index] += float(reward)
        self.total_squared_rewards[index] += float(reward) ** 2


class ExpectedRewardScaledUCB:
    """UCB1 with a manually rescaled exploration constant (our team's proposal).

    Standard UCB1 selection rule, but the inner constant of the confidence
    bound is rescaled from the canonical 2 down to:

        inner_constant = 2 * (max_expected_reward / reference_scale) ** 2

    which for max_expected_reward=0.25 and reference_scale=1.0 gives
    inner_constant = 2 * 0.25**2 = 0.125, matching the derivation on the
    whiteboard. `max_expected_reward` is NOT hardcoded here -- pass in
    whatever max(expected_reward(b) for b in bids) actually is for the
    environment you're running (it's 0.25 under Uniform(0,1); it will be a
    different number under Uniform(0.1, 0.7) or any other config, so it has
    to be recomputed per environment rather than reused).

    See the module docstring for the caveat on why this doesn't carry
    UCB1's formal regret guarantee.
    """
     
    def __init__(
        self,
        actions: Sequence[float],
        horizon: int,
        max_expected_reward: float,
        reference_scale: float = 1.0,
    ) -> None:
        if len(actions) == 0:
            raise ValueError("Action set must be non-empty.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if max_expected_reward <= 0:
            raise ValueError("max_expected_reward must be positive.")
        if reference_scale <= 0:
            raise ValueError("reference_scale must be positive.")
        self.actions = np.array(actions, dtype=float)
        self.horizon = int(horizon)
        self.n_actions = len(self.actions)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.total_rewards = np.zeros(self.n_actions, dtype=float)
        scale_factor = float(max_expected_reward) / float(reference_scale)
        self.inner_constant = 2.0 * (scale_factor ** 2)

    def select_action(self) -> Any:
        for index in range(self.n_actions):
            if self.counts[index] == 0:
                return float(self.actions[index])
        t = int(np.sum(self.counts)) + 1
        empirical_means = self.total_rewards / self.counts
        confidence_widths = np.sqrt(self.inner_constant * np.log(t) / self.counts)
        upper_confidence_values = empirical_means + confidence_widths
        best_index = int(np.argmax(upper_confidence_values))
        return float(self.actions[best_index])

    def update(self, action: Any, reward: float) -> None:
        action_value = float(action)
        matching_indices = np.where(np.isclose(self.actions, action_value))[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Unknown action: {action}")
        index = int(matching_indices[0])
        self.counts[index] += 1
        self.total_rewards[index] += float(reward)


def _solve_optimistic_budgeted_lp(
    actions: np.ndarray,
    budget_left: float,
    reward_ucbs: np.ndarray,
    cost_estimates: np.ndarray,
    budget_per_remaining_round: float,
) -> np.ndarray:
    n_actions = len(actions)
    affordable_indices = np.where(actions <= budget_left)[0]
    if len(affordable_indices) == 0:
        return np.zeros(n_actions, dtype=float)

    best_distribution = np.zeros(n_actions, dtype=float)
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
            candidate_distribution = np.zeros(n_actions, dtype=float)
            candidate_distribution[index] = 1.0
            register_candidate(candidate_distribution, objective)

    for left_position, left_index in enumerate(affordable_indices):
        for right_index in affordable_indices[left_position + 1:]:
            cost_left = float(cost_estimates[left_index])
            cost_right = float(cost_estimates[right_index])
            if cost_left == cost_right:
                continue
            probability_left = (budget_per_remaining_round - cost_right) / (cost_left - cost_right)
            probability_right = 1.0 - probability_left
            if 0.0 <= probability_left <= 1.0 and 0.0 <= probability_right <= 1.0:
                objective = (
                    probability_left * float(reward_ucbs[left_index])
                    + probability_right * float(reward_ucbs[right_index])
                )
                candidate_distribution = np.zeros(n_actions, dtype=float)
                candidate_distribution[left_index] = probability_left
                candidate_distribution[right_index] = probability_right
                register_candidate(candidate_distribution, objective)

    if np.sum(best_distribution) <= 0.0:
        zero_bid_indices = np.where(np.isclose(actions, 0.0))[0]
        if len(zero_bid_indices) > 0:
            best_distribution[int(zero_bid_indices[0])] = 1.0

    total_probability = float(np.sum(best_distribution))
    if total_probability <= 0.0:
        return best_distribution
    return best_distribution / total_probability


class BudgetedUCB1Tuned:
    """UCB1-Tuned's variance-adaptive bonus + BudgetedUCB's LP pacing."""

    def __init__(
        self,
        actions: Sequence[float],
        horizon: int,
        initial_budget: float,
        random_seed: int = None,
    ) -> None:
        if len(actions) == 0:
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
        self.total_squared_rewards = np.zeros(self.n_actions, dtype=float)
        self.total_costs = np.zeros(self.n_actions, dtype=float)
        self.t = 0
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def select_action(self) -> Any:
        if self.budget_left <= 0.0:
            return 0.0
        remaining_rounds = max(1, self.horizon - self.t)
        budget_per_remaining_round = self.budget_left / remaining_rounds
        for index, bid in enumerate(self.actions):
            if self.counts[index] == 0 and bid <= self.budget_left:
                return float(bid)
        counts_safe = np.maximum(1, self.counts)
        empirical_rewards = self.total_rewards / counts_safe
        empirical_second_moments = self.total_squared_rewards / counts_safe
        empirical_variances = np.maximum(0.0, empirical_second_moments - empirical_rewards ** 2)
        empirical_costs = self.total_costs / counts_safe
        current_round = self.t + 1
        variance_ucb = empirical_variances + np.sqrt(2.0 * np.log(current_round) / counts_safe)
        capped_variance = np.minimum(0.25, variance_ucb)
        confidence_widths = np.sqrt((np.log(current_round) / counts_safe) * capped_variance)
        reward_ucbs = empirical_rewards + confidence_widths
        distribution = _solve_optimistic_budgeted_lp(
            actions=self.actions,
            budget_left=self.budget_left,
            reward_ucbs=reward_ucbs,
            cost_estimates=empirical_costs,
            budget_per_remaining_round=budget_per_remaining_round,
        )
        if np.sum(distribution) <= 0.0:
            return 0.0
        selected_index = int(self.rng.choice(self.n_actions, p=distribution))
        return float(self.actions[selected_index])

    def update(self, action: Any, reward: float, cost: float) -> None:
        action_value = float(action)
        matching_indices = np.where(np.isclose(self.actions, action_value))[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Unknown action: {action}")
        index = int(matching_indices[0])
        self.counts[index] += 1
        self.total_rewards[index] += float(reward)
        self.total_squared_rewards[index] += float(reward) ** 2
        self.total_costs[index] += float(cost)
        self.budget_left = max(0.0, self.budget_left - float(cost))
        self.t += 1


class BudgetedExpectedRewardScaledUCB:
    """Teammate's rescaled constant + BudgetedUCB's LP pacing."""

    def __init__(
        self,
        actions: Sequence[float],
        horizon: int,
        initial_budget: float,
        max_expected_reward: float,
        reference_scale: float = 1.0,
        random_seed: int = None,
    ) -> None:
        if len(actions) == 0:
            raise ValueError("Must have at least one action.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if initial_budget < 0:
            raise ValueError("Initial budget must be non-negative.")
        if max_expected_reward <= 0:
            raise ValueError("max_expected_reward must be positive.")
        if reference_scale <= 0:
            raise ValueError("reference_scale must be positive.")
        self.actions = np.array(actions, dtype=float)
        self.horizon = int(horizon)
        self.initial_budget = float(initial_budget)
        self.budget_left = float(self.initial_budget)
        self.n_actions = len(self.actions)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.total_rewards = np.zeros(self.n_actions, dtype=float)
        self.total_costs = np.zeros(self.n_actions, dtype=float)
        self.t = 0
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        scale_factor = float(max_expected_reward) / float(reference_scale)
        self.inner_constant = 2.0 * (scale_factor ** 2)

    def select_action(self) -> Any:
        if self.budget_left <= 0.0:
            return 0.0
        remaining_rounds = max(1, self.horizon - self.t)
        budget_per_remaining_round = self.budget_left / remaining_rounds
        for index, bid in enumerate(self.actions):
            if self.counts[index] == 0 and bid <= self.budget_left:
                return float(bid)
        counts_safe = np.maximum(1, self.counts)
        empirical_rewards = self.total_rewards / counts_safe
        empirical_costs = self.total_costs / counts_safe
        current_round = self.t + 1
        confidence_widths = np.sqrt(self.inner_constant * np.log(current_round) / counts_safe)
        reward_ucbs = empirical_rewards + confidence_widths
        distribution = _solve_optimistic_budgeted_lp(
            actions=self.actions,
            budget_left=self.budget_left,
            reward_ucbs=reward_ucbs,
            cost_estimates=empirical_costs,
            budget_per_remaining_round=budget_per_remaining_round,
        )
        if np.sum(distribution) <= 0.0:
            return 0.0
        selected_index = int(self.rng.choice(self.n_actions, p=distribution))
        return float(self.actions[selected_index])

    def update(self, action: Any, reward: float, cost: float) -> None:
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