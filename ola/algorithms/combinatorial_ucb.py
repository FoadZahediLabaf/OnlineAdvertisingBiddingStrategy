"""
Combinatorial UCB algorithm for Requirement 2.

Owner:
    Luigi

Purpose:
    Implement a Combinatorial-UCB bidding strategy with budget constraints for
    multiple campaigns.

What to implement here:
    - UCB estimates for campaign/bid choices.
    - Selection of a feasible set of campaigns and bids.
    - Budget-aware action selection.
    - Compatibility with the conflict graph.

Do not:
    - Add single-campaign UCB1 logic here.
    - Add primal-dual logic here.
    - Add sliding-window or change-detector logic here.
    - Add experiment-running code here.
"""

from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np

from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


BID_SET_TOLERANCE = 1e-9


class CombinatorialUCB:
    """Combinatorial-UCB learner skeleton for Requirement 2.

    The learner maintains per-(campaign, bid) reward and cost estimates, builds
    optimistic UCB scores and pessimistic LCB costs, and selects a feasible
    super-arm (a set of non-conflicting campaigns plus a bid per campaign) whose
    optimistic cost respects the per-round budget share.
    """

    # TODO(Luigi): initialize campaign arms, confidence estimates, budget, and graph constraints.
    def __init__(
        self,
        number_of_campaigns: int,
        campaign_values: List[float],
        bid_values: List[float],
        budget: float,
        horizon: int,
        conflict_edges: Optional[List[List[int]]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        if number_of_campaigns <= 0:
            raise ValueError("number_of_campaigns must be positive.")
        if len(campaign_values) != number_of_campaigns:
            raise ValueError("campaign_values length must match number_of_campaigns.")
        if len(bid_values) == 0:
            raise ValueError("Must have at least one bid value.")
        if budget < 0:
            raise ValueError("Budget must be non-negative.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")

        self.number_of_campaigns = int(number_of_campaigns)
        self.campaign_values = np.array(campaign_values, dtype=float)
        self.bid_values = np.array(sorted(bid_values), dtype=float)
        self.budget = float(budget)
        self.horizon = int(horizon)
        self.conflict_edges: List[List[int]] = list(conflict_edges) if conflict_edges else []
        self.conflict_graph = build_conflict_graph(
            number_of_campaigns=self.number_of_campaigns,
            edges=[tuple(e) for e in self.conflict_edges],
        )

        # Enumerate all non-empty feasible campaign sets via the conflict graph.
        self.feasible_sets: List[FrozenSet[int]] = []
        all_campaigns = list(range(self.number_of_campaigns))
        for size in range(1, self.number_of_campaigns + 1):
            for combo in combinations(all_campaigns, size):
                if is_feasible_campaign_set(combo, self.conflict_graph):
                    self.feasible_sets.append(frozenset(combo))

        self.n_pulls = np.zeros((self.number_of_campaigns, len(self.bid_values)))
        self.mean_reward = np.zeros((self.number_of_campaigns, len(self.bid_values)))
        self.mean_cost = np.zeros((self.number_of_campaigns, len(self.bid_values)))
        self.t = 0
        self.budget_left = float(budget)
        self.cumulative_reward = 0.0
        self.cumulative_cost = 0.0
        self.rng = np.random.default_rng(random_seed)

    # TODO(Luigi): select feasible campaign/bid combination using optimism and budget constraints.
    def select_action(self) -> Any:
        """Select a feasible super-arm using optimistic reward and pessimistic cost."""
        positive_bids = self.bid_values[self.bid_values > 0.0]
        min_positive_bid = float(positive_bids.min()) if len(positive_bids) > 0 else 0.0

        if self.budget_left < min_positive_bid:
            return {"active_campaigns": [], "bids": []}

        # Slide 24: confidence bonus uses log(T), where T is the horizon.
        bonus = np.sqrt(
            2.0 * np.log(self.horizon) / np.maximum(self.n_pulls, 1.0)
        )
        ucb_reward = self.mean_reward + bonus
        lcb_cost = np.maximum(self.mean_cost - bonus, 0.0)

        budget_per_round = self.budget_left / max(self.horizon - self.t, 1)

        best_set: Optional[FrozenSet[int]] = None
        best_score = -float("inf")
        best_bids: Dict[int, int] = {}

        fallback_set: Optional[FrozenSet[int]] = None
        fallback_cost = float("inf")
        fallback_bids: Dict[int, int] = {}

        for feasible_set in self.feasible_sets:
            best_bid_idx: Dict[int, int] = {}
            superarm_ucb = 0.0
            superarm_lcb_cost = 0.0
            for i in feasible_set:
                b_idx = int(np.argmax(ucb_reward[i]))
                best_bid_idx[i] = b_idx
                superarm_ucb += float(ucb_reward[i, b_idx])
                superarm_lcb_cost += float(lcb_cost[i, b_idx])

            if superarm_lcb_cost <= budget_per_round:
                if superarm_ucb > best_score:
                    best_score = superarm_ucb
                    best_set = feasible_set
                    best_bids = best_bid_idx

            if superarm_lcb_cost < fallback_cost:
                fallback_cost = superarm_lcb_cost
                fallback_set = feasible_set
                fallback_bids = best_bid_idx

        if best_set is None:
            # No feasible super-arm fits the budget: abstain on the cheapest set.
            chosen_set = fallback_set if fallback_set is not None else frozenset()
            sorted_campaigns = sorted(chosen_set)
            zero_bid = float(self.bid_values[0])
            return {
                "active_campaigns": sorted_campaigns,
                "bids": [zero_bid for _ in sorted_campaigns],
            }

        sorted_campaigns = sorted(best_set)
        bids = [float(self.bid_values[best_bids[i]]) for i in sorted_campaigns]
        return {"active_campaigns": sorted_campaigns, "bids": bids}

    # TODO(Luigi): update estimates after observing reward/cost feedback.
    def update(self, action: Any, step_result: Any) -> None:
        """Update per-campaign reward and cost estimates from one step outcome."""
        active_campaigns: List[int] = action["active_campaigns"]
        wins: List[int] = step_result["wins"]
        effective_bids: List[float] = step_result["bids"]
        reward = float(step_result["reward"])
        cost = float(step_result["cost"])

        for i in active_campaigns:
            bid_played = float(effective_bids[i])
            matching_indices = np.where(
                np.abs(self.bid_values - bid_played) <= BID_SET_TOLERANCE
            )[0]
            if len(matching_indices) == 0:
                # Scaled bids from the environment may not lie in the discrete set.
                continue
            b_idx = int(matching_indices[0])

            r_i = (
                float(self.campaign_values[i]) - bid_played
                if wins[i]
                else 0.0
            )
            c_i = bid_played if wins[i] else 0.0

            self.n_pulls[i, b_idx] += 1
            self.mean_reward[i, b_idx] += (
                r_i - self.mean_reward[i, b_idx]
            ) / self.n_pulls[i, b_idx]
            self.mean_cost[i, b_idx] += (
                c_i - self.mean_cost[i, b_idx]
            ) / self.n_pulls[i, b_idx]

        self.t += 1
        self.budget_left -= cost
        self.cumulative_reward += reward
        self.cumulative_cost += cost
