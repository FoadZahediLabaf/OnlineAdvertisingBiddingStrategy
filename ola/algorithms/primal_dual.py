"""Factorized primal-dual Hedge learner for Requirement 3."""

from __future__ import annotations

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


class FactorizedPrimalDualHedgeLearner:
    """Primal-dual Hedge without enumerating the ``11^N`` joint actions."""

    def __init__(
        self,
        bid_set: Iterable[float],
        values: Iterable[float],
        initial_budget: float,
        horizon: int,
        learning_rate: float = 0.2,
        dual_learning_rate: float | None = None,
        random_seed: int = 0,
    ) -> None:
        self.bid_set = np.asarray(list(bid_set), dtype=float)
        self.values = np.asarray(list(values), dtype=float)
        self.number_of_campaigns = self.values.size
        self.number_of_bids = self.bid_set.size
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
        self.dual_variable = 0.0
        self.dual_variable_max = 1.0 / self.rho

    def select_action(self) -> np.ndarray:
        """Sample one bid per campaign under worst-case budget feasibility."""

        action = np.zeros(self.number_of_campaigns)
        worst_case_budget_left = self.budget_left
        for campaign_index in self.rng.permutation(self.number_of_campaigns):
            feasible = self.bid_set <= worst_case_budget_left + 1e-12
            scores = np.where(
                feasible,
                np.maximum(self.arm_weights[campaign_index], 1e-300),
                0.0,
            )
            if scores.sum() <= 0.0:
                scores = feasible.astype(float)
            probabilities = scores / scores.sum()
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

        for campaign_index in range(self.number_of_campaigns):
            wins = self.bid_set >= highest_competing_bids[campaign_index]
            scores = (self.values[campaign_index] - self.bid_set) * wins
            scores -= self.dual_variable * self.bid_set * wins
            span = scores.max() - scores.min()
            normalized = (
                (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)
            )
            self.arm_weights[campaign_index] *= np.exp(
                self.learning_rate * normalized
            )

        self.dual_variable += self.dual_learning_rate * (cost - self.rho)
        self.dual_variable = float(
            np.clip(self.dual_variable, 0.0, self.dual_variable_max)
        )
        return utility, cost, self.dual_variable


PrimalDualHedgeLearner = FactorizedPrimalDualHedgeLearner
