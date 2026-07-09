"""
Single-campaign stochastic environment for Requirement 1.

Owner:
    Foad

Purpose:
    Implement the single-campaign environment used in Requirement 1.

Do not:
    - Add multi-campaign logic here.
    - Add conflict-graph logic here.
    - Add adversarial or non-stationary environment logic here.
"""

from typing import Any, Dict, Optional
import numpy as np
from ola.environments.base import BaseEnvironment


class SingleCampaignEnvironment(BaseEnvironment):
    """Single-campaign stochastic first-price auction environment."""

    def __init__(
        self,
        value: float=1.00,
        initial_budget: float=100.0,
        horizon: int=100,
        min_competing_bid:  float = 0,
        max_competing_bid: float = 1.0,
        competing_bid_distribution: str = "uniform",
        n_competitors: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        if value < 0:
            raise ValueError("Value must be non-negative.")
        if initial_budget < 0:
            raise ValueError("Initial budget must be non-negative.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if min_competing_bid < 0:
            raise ValueError("Minimum competing bid must be non-negative.")
        if max_competing_bid < min_competing_bid:
            raise ValueError("Maximum competing bid must be greater than minimum competing bid.")
        allowed_distributions = {"uniform", "max_uniform"}
        if str(competing_bid_distribution) not in allowed_distributions:
            raise ValueError(
                "competing_bid_distribution must be one of "
                f"{sorted(allowed_distributions)}."
            )
        if int(n_competitors) <= 0:
            raise ValueError("n_competitors must be positive.")
        self.value = float(value)
        self.initial_budget = float(initial_budget)
        self.horizon = int(horizon)
        self.min_competing_bid = float(min_competing_bid)
        self.max_competing_bid = float(max_competing_bid)
        self.competing_bid_distribution = str(competing_bid_distribution)
        self.n_competitors = int(n_competitors)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.current_round = 0
        self.budget_left = self.initial_budget
        self.minimum_budget_to_continue = 0.0


    def reset(self) -> Dict[str, Any]:
        """Reset the environment to it's initial state"""
        self.current_round = 0
        self.budget_left = self.initial_budget  
        self.rng = np.random.default_rng(self.random_seed)
        return {
            "round": self.current_round,
            "budget_left": self.budget_left,
            "done": self.budget_left <= self.minimum_budget_to_continue,
            "info": {
                "value": self.value,
                "horizon": self.horizon,
                "min_competing_bid": self.min_competing_bid,
                "max_competing_bid": self.max_competing_bid,
                "competing_bid_distribution": self.competing_bid_distribution,
                "n_competitors": self.n_competitors,
                "minimum_budget_to_continue": self.minimum_budget_to_continue,
            },
        }
        
    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one first-price auction round."""
        if self.current_round >= self.horizon:
            raise RuntimeError("Cannot call step() after the horizon is reached. call reset() first")
        if self.budget_left <= self.minimum_budget_to_continue:
            raise RuntimeError("Cannot call step() after the campaign has stopped because the budget is depleted. call reset() first")
        bid = float(action)

        if bid < 0:
            raise ValueError("Bid must be non-negative.")
        if bid > self.budget_left:
            raise ValueError("Bid cannot exceed the remaining budget.")
        if bid > 1.0:
            raise ValueError("Bid cannot exceed 1.0 in the normalized auction setting.")

        if self.competing_bid_distribution == "uniform":
            highest_competing_bid = float(self.rng.uniform(self.min_competing_bid, self.max_competing_bid))
        elif self.competing_bid_distribution == "max_uniform":
            competitor_bids = self.rng.uniform(
                self.min_competing_bid,
                self.max_competing_bid,
                size=self.n_competitors,
            )
            highest_competing_bid = float(np.max(competitor_bids))
        else:
            raise RuntimeError("Unsupported competing bid distribution.")
        win = int(bid >= highest_competing_bid)
        reward = float(self.value - bid) if win else 0.0
        cost = float(bid) if win else 0.0

        self.budget_left = max(self.budget_left - cost, 0.0)
        self.current_round += 1
        done = self.current_round >= self.horizon or self.budget_left <= self.minimum_budget_to_continue

        # In Requirement 1, bids and wins are scalar values because there is only
        # one campaign. Future multi-campaign environments may return lists or
        # arrays for these fields while keeping the same output structure.
        return {
            "round": self.current_round,
            "action": bid,
            "bids": bid,
            "wins": win,
            "reward": reward,
            "cost": cost,
            "budget_left": self.budget_left,
            "done": done,
            "info": {
                "requested_bid": bid,
                "highest_competing_bid": highest_competing_bid,
                "value": self.value,
                "competing_bid_distribution": self.competing_bid_distribution,
                "n_competitors": self.n_competitors,
            },
        }
