"""
Adversarial / highly non-stationary multi-campaign environment for Requirement 3.

Owner:
    Mengyao

Support:
    Simone may edit this file when actively helping Mengyao.

Purpose:
    Implement the environment used to test the best-of-both-worlds primal-dual method.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from ola.environments.base import BaseEnvironment


class AdversarialMultiCampaignEnvironment(BaseEnvironment):
    """Highly non-stationary multi-campaign first-price auction environment.

    The highest competing bids are generated from a multivariate Gaussian whose
    parameters change every ``regime_length`` rounds. The full vector of highest
    competing bids is returned in ``info`` to support the full-feedback setting
    required by Requirement 3.
    """

    def __init__(
        self,
        values: Iterable[float],
        initial_budget: float,
        horizon: int,
        regime_length: int,
        sigma_scale: float = 0.05,
        random_seed: int = 0,
    ) -> None:
        self.values = np.asarray(list(values), dtype=float)
        self.number_of_campaigns = int(self.values.size)
        self.initial_budget = float(initial_budget)
        self.horizon = int(horizon)
        self.regime_length = int(regime_length)
        self.sigma_scale = float(sigma_scale)
        self.random_seed = int(random_seed)

        if self.number_of_campaigns <= 0:
            raise ValueError("At least one campaign is required")
        if np.any(self.values < 0.0):
            raise ValueError("Campaign values must be non-negative")
        if self.initial_budget < 0.0:
            raise ValueError("Initial budget must be non-negative")
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")
        if self.regime_length <= 0:
            raise ValueError("Regime length must be positive")
        if self.sigma_scale < 0.0:
            raise ValueError("Sigma scale must be non-negative")

        self.rng = np.random.default_rng(self.random_seed)
        self.highest_competing_bids = np.zeros((self.horizon, self.number_of_campaigns))
        self.regime_ids = np.zeros(self.horizon, dtype=int)
        self.regime_means: List[np.ndarray] = []
        self.regime_covariances: List[np.ndarray] = []
        self._generate_sequence()
        self.reset()

    def _sample_gaussian_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.rng.uniform(0.0, 1.0, size=self.number_of_campaigns)
        random_matrix = self.rng.normal(
            0.0,
            self.sigma_scale,
            size=(self.number_of_campaigns, self.number_of_campaigns),
        )
        covariance = random_matrix @ random_matrix.T + 1e-4 * np.eye(
            self.number_of_campaigns
        )
        return mean, covariance

    def _generate_sequence(self) -> None:
        regime_id = -1
        mean = None
        covariance = None

        for round_index in range(self.horizon):
            if round_index % self.regime_length == 0:
                mean, covariance = self._sample_gaussian_parameters()
                self.regime_means.append(mean)
                self.regime_covariances.append(covariance)
                regime_id += 1

            sample = self.rng.multivariate_normal(mean, covariance)
            self.highest_competing_bids[round_index] = np.clip(sample, 0.0, 1.0)
            self.regime_ids[round_index] = regime_id

    def reset(self) -> Dict[str, Any]:
        self.current_round = 0
        self.budget_left = self.initial_budget
        self.done = False
        return {
            "round": self.current_round,
            "budget_left": self.budget_left,
            "done": self.done,
            "info": {
                "values": self.values.tolist(),
                "number_of_campaigns": self.number_of_campaigns,
            },
        }

    def get_competing_bids(self, round_index: int) -> np.ndarray:
        """Return the full-feedback highest competing bid vector."""

        return self.highest_competing_bids[round_index]

    def evaluate_action(
        self,
        action: Any,
        highest_competing_bids: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        bids = np.asarray(action, dtype=float)
        if bids.shape != (self.number_of_campaigns,):
            raise ValueError("Action must contain one bid per campaign")
        if np.any(bids < 0.0):
            raise ValueError("Bids must be non-negative")

        wins = bids >= highest_competing_bids
        reward = float(np.sum((self.values - bids) * wins))
        cost = float(np.sum(bids * wins))
        return reward, cost, wins.astype(int)

    def step(self, action: Any) -> Dict[str, Any]:
        if self.done:
            raise RuntimeError("Cannot call step after the environment is done")

        bids = np.asarray(action, dtype=float)
        if bids.sum() > self.budget_left:
            if bids.sum() > 0.0:
                bids = bids * (self.budget_left / bids.sum())

        competing_bids = self.get_competing_bids(self.current_round)
        reward, cost, wins = self.evaluate_action(bids, competing_bids)

        self.budget_left = max(0.0, self.budget_left - cost)
        self.current_round += 1
        self.done = self.current_round >= self.horizon or self.budget_left <= 1e-12

        return {
            "round": self.current_round,
            "action": bids.tolist(),
            "bids": bids.tolist(),
            "wins": wins.tolist(),
            "reward": reward,
            "cost": cost,
            "budget_left": self.budget_left,
            "done": self.done,
            "info": {
                "highest_competing_bid": competing_bids.tolist(),
                "regime_id": int(self.regime_ids[self.current_round - 1]),
            },
        }


RandomMultivariateGaussianEnvironment = AdversarialMultiCampaignEnvironment
