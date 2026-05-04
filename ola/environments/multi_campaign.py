"""
Multi-campaign stochastic environment for Requirement 2.

Owner:
    Luigi

Purpose:
    Implement the multi-campaign environment used in Requirement 2.

Do not:
    - Add Requirement 1 single-campaign logic here.
    - Add adversarial or non-stationary environment logic here.
    - Add learner-side logic here.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from ola.environments.base import BaseEnvironment
from ola.utils.bidding import first_price_cost, first_price_reward, wins_auction
from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


BID_SET_TOLERANCE = 1e-9


class MultiCampaignEnvironment(BaseEnvironment):
    """Multi-campaign stochastic first-price auction environment."""

    def __init__(
        self,
        number_of_campaigns: int = 3,
        campaign_values: Optional[List[float]] = None,
        bid_values: Optional[List[float]] = None,
        budget: float = 300.0,
        horizon: int = 1000,
        conflict_edges: Optional[List[List[int]]] = None,
        mean_competing_bids: Optional[List[float]] = None,
        std_competing_bids: float = 0.1,
        correlation_on_edge: float = 0.8,
        random_seed: Optional[int] = None,
    ) -> None:
        if campaign_values is None:
            campaign_values = [1.0] * number_of_campaigns
        if bid_values is None:
            bid_values = [round(b, 1) for b in np.linspace(0.0, 1.0, 11)]
        if mean_competing_bids is None:
            mean_competing_bids = [0.5] * number_of_campaigns
        if len(campaign_values) != number_of_campaigns:
            raise ValueError("campaign_values length must match number_of_campaigns.")
        if len(mean_competing_bids) != number_of_campaigns:
            raise ValueError("mean_competing_bids length must match number_of_campaigns.")
        if budget < 0:
            raise ValueError("Budget must be non-negative.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if std_competing_bids <= 0:
            raise ValueError("std_competing_bids must be positive.")
        if not 0 <= correlation_on_edge < 1:
            raise ValueError("correlation_on_edge must be in [0, 1).")

        self.number_of_campaigns = int(number_of_campaigns)
        self.campaign_values = np.array(campaign_values, dtype=float)
        self.bid_values = np.array(sorted(bid_values), dtype=float)
        self.initial_budget = float(budget)
        self.horizon = int(horizon)
        self.conflict_edges: List[List[int]] = conflict_edges or []
        self.conflict_graph = build_conflict_graph(
            number_of_campaigns=self.number_of_campaigns,
            edges=[tuple(e) for e in self.conflict_edges],
        )
        self.mean_competing_bids = np.array(mean_competing_bids, dtype=float)
        self.std_competing_bids = float(std_competing_bids)
        self.correlation_on_edge = float(correlation_on_edge)

        self.adjacency_matrix = self._build_adjacency_matrix()
        self.covariance_matrix = self._build_covariance_matrix()

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.current_round = 0
        self.budget_left = self.initial_budget

    def reset(self) -> Dict[str, Any]:
        """Reset the environment to it's initial state"""
        self.current_round = 0
        self.budget_left = self.initial_budget
        self.rng = np.random.default_rng(self.random_seed)
        return{
            "round": self.current_round,
            "budget_left": self.budget_left,
            "done": False,
            "info": {
                "number_of_campaigns": self.number_of_campaigns,
                "campaign_values": self.campaign_values.tolist(),
                "bid_values": self.bid_values.tolist(),
                "horizon": self.horizon,
                "conflict_edges": self.conflict_edges,
                "adjacency_matrix": self.adjacency_matrix.tolist(),
                "covariance_matrix": self.covariance_matrix.tolist(),
                "mean_competing_bids": self.mean_competing_bids.tolist(),
                "std_competing_bids": self.std_competing_bids,
            },

        }

    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one multi-campaign first-price auction round."""
        if self.current_round >= self.horizon:
            raise RuntimeError("Cannot call step() after the horizon is reached. call reset() first")

        active_campaigns: List[int] = action["active_campaigns"]
        bids_for_active: List[float] = action["bids"]

        if len(active_campaigns) != len(bids_for_active):
            raise ValueError("active_campaigns and bids must have the same length.")
        if len(set(active_campaigns)) != len(active_campaigns):
            raise ValueError("active_campaigns must not contain duplicates.")
        for c in active_campaigns:
            if not 0 <= c < self.number_of_campaigns:
                raise ValueError("active_campaigns contains an out-of-range campaign index.")
        for b_i in bids_for_active:
            if not 0 <= b_i <= 1:
                raise ValueError("Bid must be in [0, 1].")
            if not self._is_in_bid_set(b_i):
                raise ValueError("Bid must belong to the discrete bid set.")
        if not is_feasible_campaign_set(active_campaigns, self.conflict_graph):
            raise ValueError("Campaign set is infeasible: two or more selected campaigns are non-compatible.")

        # Scale bids proportionally if their sum exceeds budget_left.
        effective_bids = np.zeros(self.number_of_campaigns)
        total_requested = float(sum(bids_for_active))
        if total_requested <= self.budget_left or total_requested == 0.0:
            scale = 1.0
        else:
            scale = self.budget_left / total_requested
        for i, b_i in zip(active_campaigns, bids_for_active):
            effective_bids[i] = float(b_i) * scale

        highest_competing_bids = np.clip(
            self.rng.multivariate_normal(self.mean_competing_bids, self.covariance_matrix),
            0.0,
            1.0,
        )

        wins = np.zeros(self.number_of_campaigns, dtype=int)
        rewards = np.zeros(self.number_of_campaigns)
        costs = np.zeros(self.number_of_campaigns)
        for i in active_campaigns:
            wins[i] = int(wins_auction(effective_bids[i], highest_competing_bids[i]))
            rewards[i] = first_price_reward(
                value=self.campaign_values[i],
                bid=effective_bids[i],
                competing_bid=highest_competing_bids[i],
            )
            costs[i] = first_price_cost(
                bid=effective_bids[i],
                competing_bid=highest_competing_bids[i],
            )
        reward = float(rewards.sum())
        cost = float(costs.sum())

        self.budget_left = max(self.budget_left - cost, 0.0)
        self.current_round += 1
        done = self.current_round >= self.horizon or self.budget_left <= 0.0

        # Bandit feedback: only reward and cost are exposed for learning.
        # Realized competing bids go in info["debug"] for visualization only.
        return {
            "round": self.current_round,
            "action": action,
            "bids": effective_bids.tolist(),
            "wins": wins.tolist(),
            "reward": reward,
            "cost": cost,
            "budget_left": self.budget_left,
            "done": done,
            "info": {
                "requested_bids": bids_for_active,
                "active_campaigns": active_campaigns,
                "campaign_values": self.campaign_values.tolist(),
                "debug": {
                    "highest_competing_bids": highest_competing_bids.tolist(),
                },
            },
        }

    def _is_in_bid_set(self, bid: float) -> bool:
        """Return True if the bid belongs to the discrete bid set within tolerance."""
        return bool(np.any(np.abs(self.bid_values - bid) <= BID_SET_TOLERANCE))

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build the binary adjacency matrix of the conflict graph."""
        adjacency_matrix = np.zeros(
            (self.number_of_campaigns, self.number_of_campaigns), dtype=int
        )
        for edge in self.conflict_edges:
            i, j = edge[0], edge[1]
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
        return adjacency_matrix

    def _build_covariance_matrix(self) -> np.ndarray:
        """Build covariance matrix from adjacency: edge -> correlation_on_edge * variance.

        Regularized to be positive semidefinite via eigenvalue clipping.
        """
        variance = self.std_competing_bids ** 2
        covariance_matrix = np.eye(self.number_of_campaigns) * variance
        covariance_matrix += (
            self.adjacency_matrix * self.correlation_on_edge * variance
        )
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        covariance_matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
        return covariance_matrix