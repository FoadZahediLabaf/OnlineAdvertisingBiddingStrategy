"""
Combinatorial UCB algorithm for Requirement 2.

Owner:
    Luigi

Purpose:
    Implement a Combinatorial-UCB bidding strategy with budget constraints for
    multiple campaigns.

Do not:
    - Add single-campaign UCB1 logic here.
    - Add primal-dual logic here.
    - Add sliding-window or change-detector logic here.
    - Add experiment-running code here.
"""

from itertools import combinations, product
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np

from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


BID_SET_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12


class CombinatorialUCB:
    """Combinatorial-UCB learner for Requirement 2.

    The learner first tries every campaign-bid arm once. Then, at each round,
    it builds reward UCBs and cost LCBs and picks the feasible super-arm with
    largest optimistic reward under the lecture budget rate rho = B / T.
    """

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
        self.conflict_edges: List[List[int]] = (
            list(conflict_edges) if conflict_edges else []
        )
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

        self._sorted_feasible_sets: List[List[int]] = [
            sorted(s) for s in self.feasible_sets
        ]

        self.n_pulls = np.zeros((self.number_of_campaigns, len(self.bid_values)))
        self.mean_reward = np.zeros((self.number_of_campaigns, len(self.bid_values)))
        self.mean_cost = np.zeros((self.number_of_campaigns, len(self.bid_values)))
        self.t = 0
        self.budget_left = float(budget)
        self.cumulative_reward = 0.0
        self.cumulative_cost = 0.0
        self.rng = np.random.default_rng(random_seed)

    def select_action(self) -> Any:
        """Select a feasible super-arm using CUCB optimism and rho = B / T."""
        positive_bids = self.bid_values[self.bid_values > 0.0]
        min_positive_bid = (
            float(positive_bids.min()) if len(positive_bids) > 0 else 0.0
        )

        if self.budget_left < min_positive_bid:
            return {"active_campaigns": [], "bids": []}

        unplayed_action = self._select_unplayed_arm()
        if unplayed_action is not None:
            return unplayed_action

        denominator = np.maximum(self.n_pulls, 1.0)
        confidence_widths = np.sqrt(2.0 * np.log(self.horizon) / denominator)
        ucb_reward = self.mean_reward + confidence_widths
        lcb_cost = np.maximum(self.mean_cost - confidence_widths, 0.0)

        # Lecture budget rate rho = B / T.
        rho = self.budget / self.horizon
        pacing_budget = self.budget_left / max(self.horizon - self.t, 1)

        best_score = -float("inf")
        best_candidates: List[Dict[str, Any]] = []
        for sorted_campaigns in self._sorted_feasible_sets:
            for bid_combo in product(
                range(len(self.bid_values)),
                repeat=len(sorted_campaigns),
            ):
                candidate = self._score_candidate(
                    sorted_campaigns=sorted_campaigns,
                    bid_combo=list(bid_combo),
                    ucb_reward=ucb_reward,
                    lcb_cost=lcb_cost,
                    rho=rho,
                    pacing_budget=pacing_budget,
                )
                if candidate is None:
                    continue
                score = float(candidate["score"])
                if score > best_score + TIE_TOLERANCE:
                    best_score = score
                    best_candidates = [candidate]
                elif abs(score - best_score) <= TIE_TOLERANCE:
                    best_candidates.append(candidate)

        if not best_candidates:
            return {"active_campaigns": [], "bids": []}

        chosen = best_candidates[int(self.rng.integers(len(best_candidates)))]
        return {
            "active_campaigns": chosen["active_campaigns"],
            "bids": chosen["bids"],
        }

    def _select_unplayed_arm(self) -> Optional[Dict[str, Any]]:
        """Play each campaign-bid arm once, following the UCB1 initialisation."""
        for campaign in range(self.number_of_campaigns):
            for bid_idx, bid in enumerate(self.bid_values):
                if (
                    self.n_pulls[campaign, bid_idx] == 0
                    and bid <= self.budget_left
                ):
                    return {
                        "active_campaigns": [campaign],
                        "bids": [float(bid)],
                    }
        return None

    def _score_candidate(
        self,
        sorted_campaigns: List[int],
        bid_combo: List[int],
        ucb_reward: np.ndarray,
        lcb_cost: np.ndarray,
        rho: float,
        pacing_budget: float,
    ) -> Optional[Dict[str, Any]]:
        """Return a scored candidate if it satisfies the budget constraints."""
        sum_bids = float(sum(self.bid_values[b] for b in bid_combo))
        if sum_bids > self.budget_left + BID_SET_TOLERANCE:
            return None

        score = 0.0
        optimistic_cost = 0.0
        empirical_cost = 0.0
        for pos, campaign in enumerate(sorted_campaigns):
            bid_idx = bid_combo[pos]
            score += float(ucb_reward[campaign, bid_idx])
            optimistic_cost += float(lcb_cost[campaign, bid_idx])
            empirical_cost += float(self.mean_cost[campaign, bid_idx])

        if optimistic_cost > rho + TIE_TOLERANCE:
            return None
        # Simple pacing guard while LCB costs are still optimistic.
        if empirical_cost > pacing_budget + TIE_TOLERANCE:
            return None

        bids = [float(self.bid_values[b]) for b in bid_combo]
        return {
            "score": score,
            "active_campaigns": list(sorted_campaigns),
            "bids": bids,
        }

    def update(self, action: Any, step_result: Any) -> None:
        """Update per-campaign reward and cost estimates from one step outcome."""
        active_campaigns: List[int] = action["active_campaigns"]
        chosen_bids: List[float] = action["bids"]
        wins: List[int] = step_result["wins"]
        effective_bids: List[float] = step_result["bids"]
        reward = float(step_result["reward"])
        cost = float(step_result["cost"])

        for pos, i in enumerate(active_campaigns):
            chosen_bid = float(chosen_bids[pos])
            matching_indices = np.where(
                np.abs(self.bid_values - chosen_bid) <= BID_SET_TOLERANCE
            )[0]
            if len(matching_indices) == 0:
                # Defensive: chosen bids come from self.bid_values.
                continue
            b_idx = int(matching_indices[0])

            # With the no-scaling guard in selection, effective_bid == chosen_bid
            # within tolerance. Use effective_bid for r/c to stay robust against
            # any residual numerical drift on the env side.
            effective_bid_i = float(effective_bids[i])
            r_i = (
                float(self.campaign_values[i]) - effective_bid_i
                if wins[i]
                else 0.0
            )
            c_i = effective_bid_i if wins[i] else 0.0

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
