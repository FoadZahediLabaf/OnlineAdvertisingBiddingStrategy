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
        self.value = float(value)
        self.initial_budget = float(initial_budget)
        self.horizon = int(horizon)
        self.min_competing_bid = float(min_competing_bid)
        self.max_competing_bid = float(max_competing_bid)
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
                "value": self.value,
                "horizon": self.horizon,
                "min_competing_bid": self.min_competing_bid,
                "max_competing_bid": self.max_competing_bid,
            },
            
        }
        
    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one first-price auction round."""
        if self.current_round >= self.horizon:
            raise RuntimeError("Cannot call step() after the horizon is reached. call reset() first")
        bid = float(action)

        if bid <0:
            raise ValueError("Bid must be non-negative.")
        
        effective_bid = min(bid, self.budget_left)
        highest_competing_bid = float(self.rng.uniform(self.min_competing_bid,self.max_competing_bid))
        win = int(effective_bid >= highest_competing_bid)
        reward = float(self.value - effective_bid )if win else 0.0
        cost = float(effective_bid) if win else 0.0

        self.budget_left = max(self.budget_left - cost, 0.0)
        self.current_round += 1
        done = self.current_round >= self.horizon or self.budget_left <= 0.0

        # In Requirement 1, bids and wins are scalar values because there is only
        # one campaign. Future multi-campaign environments may return lists or
        # arrays for these fields while keeping the same output structure.
        return {
            "round": self.current_round,
            "action": effective_bid,
            "bids": effective_bid,
            "wins": win,
            "reward": reward,
            "cost": cost,
            "budget_left": self.budget_left,
            "done": done,
            "info": {
                "requested_bid": bid,
                "highest_competing_bid": highest_competing_bid,     
                "value": self.value,
            },
        }
