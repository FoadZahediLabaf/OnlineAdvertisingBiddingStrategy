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

from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


BID_SET_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12


class CombinatorialUCB:
    """Combinatorial-UCB learner for Requirement 2.

    The learner maintains per-(campaign, bid) reward and cost estimates, builds
    optimistic UCB scores and pessimistic LCB costs, and selects a feasible
    super-arm (a set of non-conflicting campaigns plus a bid per campaign) whose
    optimistic cost respects the per-round budget share.

    Exploration is handled by an explicit warm-up phase: while any (campaign,
    bid) pair has never been played, the learner picks the feasible super-arm
    that covers the most unplayed pairs. The budget feasibility check is
    skipped during warm-up; otherwise unplayed arms (cost estimate = 0) would
    silently dominate the feasibility set without ever being validated.
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

        # Warm-up: each (campaign, bid) pair must be tried at least once before
        # the budget feasibility check becomes meaningful. Otherwise unplayed
        # arms appear free (LCB cost = 0) and an "expensive" arm could be picked
        # as feasible without ever being validated.
        if unplayed.any():
            return self._warmup_select(unplayed)

        # Slide 24: confidence width uses log(T), where T is the horizon.
        confidence_widths = np.sqrt(2.0 * np.log(self.horizon) / self.n_pulls)
        ucb_reward = self.mean_reward + confidence_widths
        lcb_cost = np.maximum(self.mean_cost - confidence_widths, 0.0)

        budget_per_round = self.budget_left / max(self.horizon - self.t, 1)

        # Per-campaign best bid by UCB, with random tie-break across equal UCBs.
        best_bid_idx_per_campaign = self._argmax_with_random_tiebreak(ucb_reward)

        best_score = -np.inf
        tied_candidates: List[Tuple[FrozenSet[int], Dict[int, int]]] = []
        fallback_set: Optional[FrozenSet[int]] = None
        fallback_cost = float("inf")

        for feasible_set in self.feasible_sets:
            bid_choice = {i: int(best_bid_idx_per_campaign[i]) for i in feasible_set}
            superarm_ucb = float(sum(ucb_reward[i, bid_choice[i]] for i in feasible_set))
            superarm_lcb_cost = float(sum(lcb_cost[i, bid_choice[i]] for i in feasible_set))

            if superarm_lcb_cost <= budget_per_round:
                if superarm_ucb > best_score + TIE_TOLERANCE:
                    best_score = superarm_ucb
                    tied_candidates = [(feasible_set, bid_choice)]
                elif abs(superarm_ucb - best_score) <= TIE_TOLERANCE:
                    tied_candidates.append((feasible_set, bid_choice))

            if superarm_lcb_cost < fallback_cost:
                fallback_cost = superarm_lcb_cost
                fallback_set = feasible_set

        if tied_candidates:
            # Break ties between equal-score feasible super-arms uniformly at random.
            chosen_set, chosen_bids = tied_candidates[
                int(self.rng.integers(len(tied_candidates)))
            ]
            sorted_campaigns = sorted(chosen_set)
            bids = [float(self.bid_values[chosen_bids[i]]) for i in sorted_campaigns]
            return {"active_campaigns": sorted_campaigns, "bids": bids}

        # No feasible super-arm fits the budget: abstain on the cheapest set.
        chosen_set = fallback_set if fallback_set is not None else frozenset()
        sorted_campaigns = sorted(chosen_set)
        zero_bid = float(self.bid_values[0])
        return {
            "active_campaigns": sorted_campaigns,
            "bids": [zero_bid for _ in sorted_campaigns],
        }

    def _warmup_select(self, unplayed: np.ndarray) -> Dict[str, Any]:
        """Pick the feasible super-arm covering the most unplayed (campaign, bid) pairs."""
        best_count = -1
        candidates: List[Tuple[FrozenSet[int], Dict[int, int]]] = []
        for feasible_set in self.feasible_sets:
            bid_choice: Dict[int, int] = {}
            covered = 0
            for i in feasible_set:
                unplayed_bid_idx = np.where(unplayed[i])[0]
                if len(unplayed_bid_idx) > 0:
                    bid_choice[i] = int(self.rng.choice(unplayed_bid_idx))
                    covered += 1
                else:
                    # All bids for this campaign already played: pick at random.
                    bid_choice[i] = int(self.rng.integers(len(self.bid_values)))
            if covered > best_count:
                best_count = covered
                candidates = [(feasible_set, bid_choice)]
            elif covered == best_count:
                candidates.append((feasible_set, bid_choice))

        chosen_set, chosen_bids = candidates[int(self.rng.integers(len(candidates)))]
        sorted_campaigns = sorted(chosen_set)
        bids = [float(self.bid_values[chosen_bids[i]]) for i in sorted_campaigns]
        return {"active_campaigns": sorted_campaigns, "bids": bids}

    def _argmax_with_random_tiebreak(self, scores: np.ndarray) -> np.ndarray:
        """Per-row argmax with uniform random tie-break across equal entries."""
        result = np.zeros(scores.shape[0], dtype=int)
        for i in range(scores.shape[0]):
            row = scores[i]
            max_val = row.max()
            tied = np.where(np.abs(row - max_val) <= TIE_TOLERANCE)[0]
            result[i] = int(self.rng.choice(tied))
        return result

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

            # Per-campaign realized outcomes use the effective bid run by the env.
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
