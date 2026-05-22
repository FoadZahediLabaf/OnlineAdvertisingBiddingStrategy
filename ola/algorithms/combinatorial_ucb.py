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
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


BID_SET_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12


class CombinatorialUCB:
    """Combinatorial-UCB learner for Requirement 2.

    The learner maintains per-(campaign, bid) reward and cost estimates, builds
    optimistic UCB scores and pessimistic LCB costs, and selects a feasible
    super-arm jointly over both the non-conflicting campaign set and the bid
    vector. Two constraints are respected at selection time:
        - sum of LCB costs <= budget_per_round (pacing target);
        - sum of chosen bids <= budget_left (no env-side scaling), so the arm
          updated in update() matches the bid actually executed.

    Exploration is handled by an explicit warm-up phase. While any (campaign,
    bid) pair has never been played, the learner picks the feasible (set, bid
    vector) covering the most unplayed pairs under the same no-scaling guard.
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

        # Pre-sort each feasible set once: index access in the oracle uses lists
        # so the order is stable across enumeration and the returned action.
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
        """Select a feasible super-arm using optimistic reward and pessimistic cost."""
        positive_bids = self.bid_values[self.bid_values > 0.0]
        min_positive_bid = float(positive_bids.min()) if len(positive_bids) > 0 else 0.0

        # Abstain when budget cannot cover even the cheapest positive bid.
        if self.budget_left < min_positive_bid:
            return {"active_campaigns": [], "bids": []}

        unplayed = self.n_pulls == 0

        # Warm-up: every (campaign, bid) pair must be tried at least once before
        # the LCB-based budget feasibility check becomes meaningful. The warm-up
        # selector also enforces the no-scaling guard so updates remain unbiased.
        if unplayed.any():
            return self._warmup_select(unplayed)

        # Slide 24: confidence width uses log(T), where T is the horizon.
        confidence_widths = np.sqrt(2.0 * np.log(self.horizon) / self.n_pulls)
        ucb_reward = self.mean_reward + confidence_widths
        lcb_cost = np.maximum(self.mean_cost - confidence_widths, 0.0)

        budget_per_round = self.budget_left / max(self.horizon - self.t, 1)

        return self._joint_oracle(ucb_reward, lcb_cost, budget_per_round)

    def _joint_oracle(
        self,
        ucb_reward: np.ndarray,
        lcb_cost: np.ndarray,
        budget_per_round: float,
    ) -> Dict[str, Any]:
        """Jointly maximize sum-UCB over (campaign set, bid vector).

        For each feasible campaign set S, enumerate every bid vector b over
        the bid grid and keep the (S, b) with maximum sum_{i in S} UCB(i, b_i)
        subject to:
            - sum_{i in S} LCB_cost(i, b_i) <= budget_per_round,
            - sum_{i in S} b_i <= budget_left (no env scaling).
        Ties are broken uniformly at random across candidates that match the
        best score within TIE_TOLERANCE. If no candidate satisfies the pacing
        constraint, fall back to the (S, b) with minimum LCB cost.
        """
        n_bids = len(self.bid_values)
        best_score = -np.inf
        tied: List[Tuple[List[int], List[int]]] = []
        fallback: Optional[Tuple[List[int], List[int]]] = None
        fallback_cost = float("inf")

        for sorted_campaigns in self._sorted_feasible_sets:
            k = len(sorted_campaigns)
            for bid_combo in product(range(n_bids), repeat=k):
                sum_bids = float(sum(self.bid_values[b] for b in bid_combo))
                if sum_bids > self.budget_left + BID_SET_TOLERANCE:
                    continue

                sum_ucb = 0.0
                sum_lcb = 0.0
                for pos in range(k):
                    i = sorted_campaigns[pos]
                    b = bid_combo[pos]
                    sum_ucb += ucb_reward[i, b]
                    sum_lcb += lcb_cost[i, b]

                if sum_lcb <= budget_per_round:
                    if sum_ucb > best_score + TIE_TOLERANCE:
                        best_score = sum_ucb
                        tied = [(sorted_campaigns, list(bid_combo))]
                    elif abs(sum_ucb - best_score) <= TIE_TOLERANCE:
                        tied.append((sorted_campaigns, list(bid_combo)))

                if sum_lcb < fallback_cost:
                    fallback_cost = sum_lcb
                    fallback = (sorted_campaigns, list(bid_combo))

        if tied:
            chosen_campaigns, chosen_bid_idx = tied[int(self.rng.integers(len(tied)))]
            bids = [float(self.bid_values[b]) for b in chosen_bid_idx]
            return {"active_campaigns": list(chosen_campaigns), "bids": bids}

        if fallback is not None:
            chosen_campaigns, chosen_bid_idx = fallback
            bids = [float(self.bid_values[b]) for b in chosen_bid_idx]
            return {"active_campaigns": list(chosen_campaigns), "bids": bids}

        return {"active_campaigns": [], "bids": []}

    def _warmup_select(self, unplayed: np.ndarray) -> Dict[str, Any]:
        """Pick a feasible (set, bid vector) covering the most unplayed pairs.

        Enforces sum of chosen bids <= budget_left so the env never scales
        during warm-up. Without this guard the env would scale the bids and
        the (campaign, chosen_bid) arm would be updated with a realized
        reward/cost matching a different (smaller) effective bid, biasing the
        very estimates warm-up is meant to seed.
        """
        n_bids = len(self.bid_values)
        best_count = -1
        candidates: List[Tuple[List[int], List[int]]] = []

        for sorted_campaigns in self._sorted_feasible_sets:
            k = len(sorted_campaigns)
            for bid_combo in product(range(n_bids), repeat=k):
                sum_bids = float(sum(self.bid_values[b] for b in bid_combo))
                if sum_bids > self.budget_left + BID_SET_TOLERANCE:
                    continue
                covered = 0
                for pos in range(k):
                    if unplayed[sorted_campaigns[pos], bid_combo[pos]]:
                        covered += 1
                if covered > best_count:
                    best_count = covered
                    candidates = [(sorted_campaigns, list(bid_combo))]
                elif covered == best_count:
                    candidates.append((sorted_campaigns, list(bid_combo)))

        if not candidates:
            return {"active_campaigns": [], "bids": []}

        chosen_campaigns, chosen_bid_idx = candidates[
            int(self.rng.integers(len(candidates)))
        ]
        bids = [float(self.bid_values[b]) for b in chosen_bid_idx]
        return {"active_campaigns": list(chosen_campaigns), "bids": bids}

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
                # Defensive: chosen bids come from self.bid_values, this shouldn't happen.
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
