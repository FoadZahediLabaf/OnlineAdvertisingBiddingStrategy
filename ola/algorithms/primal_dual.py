"""Factorized primal-dual Hedge learner for Requirement 3."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def compute_utility_and_cost(
    action: np.ndarray,
    highest_competing_bids: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return first-price utility, cost, and win indicators."""

    wins = action >= highest_competing_bids
    utility = float(np.sum((values - action) * wins))
    cost = float(np.sum(action * wins))
    return utility, cost, wins.astype(int)


def _campaign_set_is_feasible(campaign_set: tuple[int, ...], conflict_edges) -> bool:
    selected = set(campaign_set)
    return all(first not in selected or second not in selected for first, second in conflict_edges)


class FactorizedPrimalDualHedgeLearner:
    """Primal-dual Hedge without enumerating the ``11^N`` joint actions.

    The learner keeps one Hedge distribution over feasible campaign sets and
    one Hedge distribution per campaign over the shared bid grid. This keeps
    the action model compact while still learning which super-arm to play.
    """

    def __init__(
        self,
        bid_set: Iterable[float],
        values: Iterable[float],
        initial_budget: float,
        horizon: int,
        learning_rate: float = 0.2,
        dual_learning_rate: float | None = None,
        allowed_campaign_sets: Iterable[Iterable[int]] | None = None,
        conflict_edges: Iterable[Iterable[int]] | None = None,
        random_seed: int = 0,
    ) -> None:
        self.bid_set = np.asarray(list(bid_set), dtype=float)
        self.values = np.asarray(list(values), dtype=float)
        self.number_of_campaigns = self.values.size
        self.number_of_bids = self.bid_set.size
        self.conflict_edges = [tuple(edge) for edge in (conflict_edges or [])]
        self.allowed_campaign_sets = self._normalise_allowed_campaign_sets(
            allowed_campaign_sets
        )
        self.budget_left = float(initial_budget)
        self.rho = float(initial_budget) / int(horizon)
        self.learning_rate = float(learning_rate)
        self.dual_learning_rate = (
            float(dual_learning_rate)
            if dual_learning_rate is not None
            else 1.0 / np.sqrt(horizon)
        )
        self.rng = np.random.default_rng(random_seed)
        self.arm_weights = np.ones((self.number_of_campaigns, self.number_of_bids))
        self.superarm_weights = np.ones(len(self.allowed_campaign_sets), dtype=float)
        self.dual_variable = 0.0
        self.dual_variable_max = 1.0 / self.rho

    def _normalise_allowed_campaign_sets(
        self,
        allowed_campaign_sets: Iterable[Iterable[int]] | None,
    ) -> list[tuple[int, ...]]:
        if allowed_campaign_sets is None:
            if not self.conflict_edges:
                return [tuple(range(self.number_of_campaigns))]
            generated = []
            for size in (1, 2):
                for campaign_set in combinations(range(self.number_of_campaigns), size):
                    if _campaign_set_is_feasible(campaign_set, self.conflict_edges):
                        generated.append(campaign_set)
            return generated

        normalised = []
        for campaign_set in allowed_campaign_sets:
            candidate = tuple(sorted(int(campaign) for campaign in campaign_set))
            if not candidate:
                continue
            if any(campaign < 0 or campaign >= self.number_of_campaigns for campaign in candidate):
                raise ValueError(f"Invalid campaign set {campaign_set}.")
            if len(set(candidate)) != len(candidate):
                raise ValueError(f"Campaign set {campaign_set} contains duplicates.")
            if not _campaign_set_is_feasible(candidate, self.conflict_edges):
                raise ValueError(f"Campaign set {campaign_set} violates conflict_edges.")
            normalised.append(candidate)

        if not normalised:
            raise ValueError("allowed_campaign_sets must contain at least one non-empty set.")
        return normalised

    def _superarm_distribution(self) -> np.ndarray:
        weights = np.maximum(self.superarm_weights, 1e-300)
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            weights = np.ones_like(weights)
            total = weights.sum()
        return weights / total

    def _campaign_distribution(self, campaign_index: int, budget_left: float) -> np.ndarray:
        feasible = self.bid_set <= budget_left + 1e-12
        if not np.any(feasible):
            feasible = np.isclose(self.bid_set, 0.0)
        weights = np.maximum(self.arm_weights[campaign_index], 1e-300)
        scores = np.where(feasible, weights, 0.0)
        total = scores.sum()
        if total <= 0.0 or not np.isfinite(total):
            scores = feasible.astype(float)
            total = scores.sum()
        return scores / total

    def select_action(self) -> np.ndarray:
        """Sample one bid per campaign under worst-case budget feasibility."""

        action = np.zeros(self.number_of_campaigns)
        worst_case_budget_left = self.budget_left
        superarm_probabilities = self._superarm_distribution()
        superarm_index = int(
            self.rng.choice(len(self.allowed_campaign_sets), p=superarm_probabilities)
        )
        campaign_set = self.allowed_campaign_sets[superarm_index]
        for campaign_index in self.rng.permutation(campaign_set):
            probabilities = self._campaign_distribution(campaign_index, worst_case_budget_left)
            bid = float(
                self.bid_set[self.rng.choice(self.number_of_bids, p=probabilities)]
            )
            action[campaign_index] = bid
            worst_case_budget_left = max(0.0, worst_case_budget_left - bid)
        return action

    def update(
        self,
        action: np.ndarray,
        highest_competing_bids: np.ndarray,
    ) -> tuple[float, float, float]:
        """Apply full-feedback Hedge and dual budget updates."""

        utility, cost, _ = compute_utility_and_cost(
            np.asarray(action, dtype=float),
            np.asarray(highest_competing_bids, dtype=float),
            self.values,
        )
        self.budget_left = max(0.0, self.budget_left - cost)

        arm_rewards = np.zeros((self.number_of_campaigns, self.number_of_bids), dtype=float)
        for campaign_index in range(self.number_of_campaigns):
            wins = self.bid_set >= highest_competing_bids[campaign_index]
            scores = (self.values[campaign_index] - self.bid_set) * wins
            scores -= self.dual_variable * self.bid_set * wins
            arm_rewards[campaign_index] = scores
            span = scores.max() - scores.min()
            normalized = (
                (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)
            )
            self.arm_weights[campaign_index] *= np.exp(
                self.learning_rate * normalized
            )

        superarm_rewards = np.array(
            [
                sum(float(np.max(arm_rewards[campaign_index])) for campaign_index in campaign_set)
                for campaign_set in self.allowed_campaign_sets
            ],
            dtype=float,
        )
        min_superarm_reward = np.min(superarm_rewards)
        max_superarm_reward = np.max(superarm_rewards)
        normalized_superarm_rewards = (
            (superarm_rewards - min_superarm_reward)
            / (max_superarm_reward - min_superarm_reward)
            if max_superarm_reward > min_superarm_reward
            else np.zeros_like(superarm_rewards)
        )
        self.superarm_weights *= np.exp(self.learning_rate * normalized_superarm_rewards)

        self.dual_variable += self.dual_learning_rate * (cost - self.rho)
        self.dual_variable = float(
            np.clip(self.dual_variable, 0.0, self.dual_variable_max)
        )
        return utility, cost, self.dual_variable


PrimalDualHedgeLearner = FactorizedPrimalDualHedgeLearner
