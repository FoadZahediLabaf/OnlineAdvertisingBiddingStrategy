"""
Primal-dual bidding algorithm for Requirement 3.

Owner:
    Mengyao

Support:
    Simone may edit this file when actively helping Mengyao.

Purpose:
    Implement the best-of-both-worlds primal-dual method with budget constraints.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, List, Tuple

import numpy as np


def is_feasible_action(action: np.ndarray, conflict_edges: List[Tuple[int, int]]) -> bool:
    """Check whether an action respects campaign conflict constraints."""

    active_campaigns = action > 0.0
    return all(
        not (active_campaigns[first] and active_campaigns[second])
        for first, second in conflict_edges
    )


def generate_feasible_actions(
    bid_set: Iterable[float],
    number_of_campaigns: int,
    conflict_edges: List[Tuple[int, int]],
) -> np.ndarray:
    """Generate all feasible bid vectors from a shared discrete bid set."""

    bid_array = np.asarray(list(bid_set), dtype=float)
    actions = []
    for action_tuple in product(bid_array, repeat=number_of_campaigns):
        action = np.array(action_tuple, dtype=float)
        if is_feasible_action(action, conflict_edges):
            actions.append(action)
    return np.array(actions)


def compute_utility_and_cost(
    action: np.ndarray,
    highest_competing_bids: np.ndarray,
    values: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Compute first-price auction utility, cost, and win indicators."""

    wins = action >= highest_competing_bids
    reward = float(np.sum((values - action) * wins))
    cost = float(np.sum(action * wins))
    return reward, cost, wins.astype(int)


class PrimalDualBidder:
    """Budget-aware primal-dual learner with factorized Hedge updates."""

    def __init__(
        self,
        actions: np.ndarray,
        values: Iterable[float],
        initial_budget: float,
        horizon: int,
        learning_rate: float | None = None,
        dual_learning_rate: float | None = None,
        random_seed: int = 0,
    ) -> None:
        self.actions = np.asarray(actions, dtype=float)
        self.values = np.asarray(list(values), dtype=float)
        self.initial_budget = float(initial_budget)
        self.budget_left = float(initial_budget)
        self.horizon = int(horizon)
        self.rho = self.initial_budget / self.horizon
        self.learning_rate = (
            float(learning_rate) if learning_rate is not None else 1.0 / np.sqrt(horizon)
        )
        self.dual_learning_rate = (
            float(dual_learning_rate)
            if dual_learning_rate is not None
            else 1.0 / np.sqrt(horizon)
        )
        self.rng = np.random.default_rng(random_seed)

        if self.actions.ndim != 2:
            raise ValueError("Actions must be a 2D array")
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")
        if self.initial_budget < 0.0:
            raise ValueError("Initial budget must be non-negative")
        if self.values.size != self.actions.shape[1]:
            raise ValueError("Values must match the number of campaigns")

        self.number_of_actions = self.actions.shape[0]
        self.number_of_campaigns = self.actions.shape[1]
        self.bid_set = np.unique(self.actions)
        self.number_of_bids = self.bid_set.size
        self.bid_to_index = {bid: index for index, bid in enumerate(self.bid_set)}
        self.action_bid_indices = np.array(
            [[self.bid_to_index[bid] for bid in action] for action in self.actions]
        )
        self.action_worst_case_costs = np.sum(self.actions, axis=1)
        zero_action_candidates = np.where(self.action_worst_case_costs == 0.0)[0]
        if zero_action_candidates.size == 0:
            raise ValueError("The action set must include the all-zero action")
        self.zero_action_index = int(zero_action_candidates[0])

        self.arm_weights = np.ones((self.number_of_campaigns, self.number_of_bids))
        self.dual_variable = 0.0
        self.dual_variable_max = 1.0 / self.rho if self.rho > 0.0 else 1.0

    def get_distribution(self, budget_feasible_only: bool = True) -> np.ndarray:
        """Return the current distribution over feasible actions."""

        if budget_feasible_only:
            feasible_mask = self.action_worst_case_costs <= self.budget_left + 1e-12
        else:
            feasible_mask = np.ones(self.number_of_actions, dtype=bool)

        if not np.any(feasible_mask):
            feasible_mask = np.zeros(self.number_of_actions, dtype=bool)
            feasible_mask[self.zero_action_index] = True

        clipped_weights = np.maximum(self.arm_weights, 1e-300)
        log_weights = np.log(clipped_weights)
        action_log_scores = np.sum(
            log_weights[np.arange(self.number_of_campaigns), self.action_bid_indices],
            axis=1,
        )

        masked_log_scores = np.full(self.number_of_actions, -np.inf)
        masked_log_scores[feasible_mask] = action_log_scores[feasible_mask]
        max_log_score = np.max(masked_log_scores[feasible_mask])
        action_scores = np.zeros(self.number_of_actions)
        action_scores[feasible_mask] = np.exp(
            masked_log_scores[feasible_mask] - max_log_score
        )

        total_score = np.sum(action_scores)
        if total_score == 0.0 or not np.isfinite(total_score):
            action_scores[feasible_mask] = 1.0
            total_score = np.sum(action_scores)

        return action_scores / total_score

    def select_action(self) -> Tuple[int, np.ndarray]:
        probabilities = self.get_distribution(budget_feasible_only=True)
        action_index = int(self.rng.choice(self.number_of_actions, p=probabilities))
        return action_index, self.actions[action_index]

    def update(self, action: Any, reward: float, cost: float, feedback: Any) -> None:
        """Update primal Hedge weights and the dual budget multiplier."""

        highest_competing_bids = np.asarray(feedback["highest_competing_bid"], dtype=float)
        self.budget_left = max(0.0, self.budget_left - float(cost))

        arm_rewards = np.zeros((self.number_of_campaigns, self.number_of_bids))
        for campaign_index in range(self.number_of_campaigns):
            for bid_index, bid in enumerate(self.bid_set):
                win = bid >= highest_competing_bids[campaign_index]
                arm_reward = (self.values[campaign_index] - bid) * win
                arm_cost = bid * win
                arm_rewards[campaign_index, bid_index] = (
                    arm_reward - self.dual_variable * arm_cost
                )

        for campaign_index in range(self.number_of_campaigns):
            rewards = arm_rewards[campaign_index]
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            if max_reward > min_reward:
                normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
            else:
                normalized_rewards = np.zeros_like(rewards)
            self.arm_weights[campaign_index] *= np.exp(
                self.learning_rate * normalized_rewards
            )

        self.dual_variable += self.dual_learning_rate * (float(cost) - self.rho)
        self.dual_variable = float(
            np.clip(self.dual_variable, 0.0, self.dual_variable_max)
        )


PrimalDualHedgeLearner = PrimalDualBidder
