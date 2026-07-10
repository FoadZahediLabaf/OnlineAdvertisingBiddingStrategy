"""
Sliding-window Combinatorial UCB algorithm for Requirement 4.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Implement the sliding-window version of Combinatorial-UCB for slightly
    non-stationary multi-campaign environments.

What to implement here:
    - Combinatorial-UCB action selection using only recent observations.
    - Sliding-window reward and cost estimates.
    - Budget-aware action selection.
    - Compatibility with campaign conflict constraints.

Do not:
    - Add standard stationary Combinatorial-UCB logic here unless reused cleanly.
    - Add change-detector logic here.
    - Add primal-dual logic here.
    - Add experiment-running code here.
"""

# The class is created starting from the, previously implemented, combinatorial UCB algorithm (combinatorial_ucb.py).
# The main features added, in order to implement the sliding-window approach, are:
# 1. The estimates of the rewards and costs, as well as the pull counts, are computed based only on the last window_size rounds.
# 2. Based on theoretical results, instead of using log(horizon) in the exploration bonus, it is used the log(min(t, window_size)).
# 3. It is added an ever_pulled mask (which won't ever be reset) in order to track which arms have been pulled. 
#       By doing this, the arms that have not been pulled in the last window_size rounds are explored again, 
#       thanks to the optimistic confidence bounds.
# Note that based on theoretical results, without knowing a priori the number of times how many times a distribution will change, a good value
# to use as window size is equal to sqrt(T).
# In the case of knowing a priori the number of changes, another optimal value for the window size is
# W = int(2*np.sqrt(T*np.log(T)/U_T))

from collections import deque
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np

# Import from CUCB version and keep as the same some auxiliiary parameters.
from ola.algorithms.combinatorial_ucb import (
    BID_SET_TOLERANCE,
    TIE_TOLERANCE,
    CombinatorialUCB,
)


class SlidingWindowCUCB(CombinatorialUCB):
    def __init__(
        self,
        number_of_campaigns: int,
        campaign_values: List[float],
        bid_values: List[float],
        budget: float,
        horizon: int,
        # New parameter
        window_size: int,
        conflict_edges: Optional[List[List[int]]] = None,
        random_seed: Optional[int] = None,
        # New parameter. 
        # IT IS SET TO 2.0 FROM THEORY SLIDES. In the lectures slide it was directly the value 2, and not seen as a tunable parameter at all.
        exploration_coefficient: float = 2.0,
    ) -> None:
        # After setting up all the needed parameters, the construct of the parent class is called.
        # So the baseline, starting point, is effectively a complete combinatorial UCB.
        super().__init__(
            number_of_campaigns=number_of_campaigns,
            campaign_values=campaign_values,
            bid_values=bid_values,
            budget=budget,
            horizon=horizon,
            conflict_edges=conflict_edges,
            random_seed=random_seed,
        )

        # The new parameters needs to be validated in this new class.
        if window_size <= 0:
            raise ValueError("Window_size must be positive.")
        if exploration_coefficient <= 0:
            raise ValueError("Exploration_coefficient must be positive.")

        self.window_size = int(window_size)
        self.exploration_coefficient = float(exploration_coefficient)

        # The window is defined as a double-ended queue.
        # Buffer of recent observations, in particular, 
        # each observation will have the following structure: (round_index, campaign, bid_idx, r, c).
        self.window: deque = deque()

        # Define the shape of the matrix: number of campaigns and possible bids.
        shape = (self.number_of_campaigns, len(self.bid_values))
        # Keep the count of how many times each couple (campaign, bid) has been played inside the window.
        self.window_count = np.zeros(shape, dtype=int)
        # Keeps track of the current sum of rewards for each couple (campaign, bid) inside the window.
        self.window_sum_reward = np.zeros(shape, dtype=float)
        # Keeps track of the current sum of costs for each couple (campaign, bid) inside the window.
        self.window_sum_cost = np.zeros(shape, dtype=float)
        # Persistent mask used for initialisation sweep only. After that, arms without observations in the window
        # will be explored through the optimistic confidence bound, as normally happens.
        self.ever_pulled = np.zeros(shape, dtype=bool)

    def select_action(self) -> Any:
        """ Select an arm to be played. Check also firstly if there is the need to perform an arm due to the initial
            forced exploration phase. Reasoning similar to the classical CUCB but considering the information
            contained in the window, instead of the whole history."""

        # Get the value of the minimum possible bid to choose. 
        positive_bids = self.bid_values[self.bid_values > 0.0]
        min_positive_bid = (
            float(positive_bids.min()) if len(positive_bids) > 0 else 0.0
        )
        
        # If the budget left is smaller than the minimum possible bid, no action can be selected.
        if self.budget_left < min_positive_bid:
            return {"active_campaigns": [], "bids": []}

        # In the first phase, every couple (campaign, bid) must be played at least one. 
        unplayed_action = self.select_unplayed_arm()
        if unplayed_action is not None:
            return unplayed_action

        # We get the effective length of the window, or directly t if the round is smaller than the window size.
        # We use this value in sliding window instead of the whole time horizon. 
        # It is taken the maximum within the inner min() and 2 to avoid having error computing log(0) and log(1)
        # in the following computation.
        effective_length = max(min(self.t, self.window_size), 2)
        # Keep 1 as lower bound in order to avoid division by 0, this also boosts the exploration term.
        # n_pulls is the same as window_count but it is kept as a float matrix. 
        # So, it is used to know how many times each arm has been pulled in the last window_size rounds.
        recent_pulls = np.maximum(self.n_pulls, 1.0)
        confidence_widths = np.sqrt(
            self.exploration_coefficient * np.log(effective_length) / recent_pulls
        )
        # For optimistic estimation in UCB, take an upper confidence bound for rewards and a lower confidence
        # bound for costs. 
        # mean_reward and mean_cost are the mean values for every couple (campaign, bid) computed considering only the 
        # observations in the current window.
        ucb_reward = self.mean_reward + confidence_widths
        lcb_cost = np.maximum(self.mean_cost - confidence_widths, 0.0)

        # Budget per round ratio, rho = B / T.
        rho = self.budget / self.horizon
        # pacing_budget is the adjusted version of rho, considers the actual budget left and the remaining number of rounds.
        pacing_budget = self.budget_left / max(self.horizon - self.t, 1)

        # Initialize the search for the best super-arm. Start from minus infinite and then compare with the many candidates.
        best_score = -float("inf")
        best_candidates: List[Dict[str, Any]] = []

        # Consider all the possible campaign subsets based on the conflict graph and ...
        for sorted_campaigns in self._sorted_feasible_sets:
            # ... try every possible combination of bids.
            for bid_combo in product(
                range(len(self.bid_values)),
                repeat=len(sorted_campaigns),
            ):
                # Delegate the scoring of the possible candidate to the _score_candidate method.
                candidate = self._score_candidate(
                    sorted_campaigns=sorted_campaigns,
                    bid_combo=list(bid_combo),
                    ucb_reward=ucb_reward,
                    lcb_cost=lcb_cost,
                    rho=rho,
                    pacing_budget=pacing_budget,
                )
                # In the case the candidate is not valid we continue with the next one.
                if candidate is None:
                    continue
                
                # If the score of the new candidate is sensibly better than the best score, 
                # update the value related to it with the new one.
                # If there is a sort of a tie, append the new candidate to the list.
                score = float(candidate["score"])
                if score > best_score + TIE_TOLERANCE:
                    best_score = score
                    best_candidates = [candidate]
                elif abs(score - best_score) <= TIE_TOLERANCE:
                    best_candidates.append(candidate)

        # If there is no valid candidate, don't choose any action.
        if not best_candidates:
            return {"active_campaigns": [], "bids": []}

        # In the case of ties among different candidates, select randomly.
        chosen = best_candidates[int(self.rng.integers(len(best_candidates)))]
        return {
            "active_campaigns": chosen["active_campaigns"],
            "bids": chosen["bids"],
        }

    def select_unplayed_arm(self) -> Optional[Dict[str, Any]]:
        """ Manage the first forced exploration phase, where every couple (campaign, bid) needs to be played at least once.
            If every couple has been played, this method returns None. """
        # Check for every possible combination of campaign and bid values ...
        for campaign in range(self.number_of_campaigns):
            for bid_idx, bid in enumerate(self.bid_values):
                # ... if a couple is not found in ever_pulled, and there is enough budget left, the found couple is returned.
                # Note: ever_pulled keeps track whether a couple has been played in the past. Its value can't go to 0, as it happens for n_pulls or window_count (they consider only the last window_size rounds).
                if (
                    not self.ever_pulled[campaign, bid_idx]
                    and bid <= self.budget_left
                ):
                    return {
                        "active_campaigns": [campaign],
                        "bids": [float(bid)],
                    }
        # If there are not any unplayed couple, the method return None.
        return None

    def update(self, action: Any, step_result: Any) -> None:
        """ After the action has been selected by the agent and the environment returned an outcome for the action,
            update the information of the agent based on the feedback received. Update also the information regarding
            the window. """
        # Get the various information based on the action played and the result obtained by the environment.
        active_campaigns: List[int] = action["active_campaigns"]    # Selected campaigns for the current round.
        chosen_bids: List[float] = action["bids"]                   # Selected bids values for the current round.
        wins: List[int] = step_result["wins"]                       # Indicator of whether the many bids for the campaigns resulted in a win or not.
        effective_bids: List[float] = step_result["bids"]           # Bid vector returned by the environment. If a bid was not succesful it set to 0.
        reward = float(step_result["reward"])                       # Reward obtained in the current round.
        cost = float(step_result["cost"])                           # Cost sustained in the current round.

        # Update the index of the round.
        self.t += 1
        round_index = self.t

        # For every active campaign in the current round ...
        for pos, i in enumerate(active_campaigns):
            chosen_bid = float(chosen_bids[pos])
            # Find the index of the chosen bid. Use a numerical tolerance for small discrepancies that can happen.
            matching_indices = np.where(
                np.abs(self.bid_values - chosen_bid) <= BID_SET_TOLERANCE
            )[0]
            if len(matching_indices) == 0:
                # Defensive: chosen bids come from self.bid_values, otherwise skip it.
                continue
            b_idx = int(matching_indices[0])

            # Get the effective bid for the campaign i, as returned by the environment. Then compute the reward,
            # if the bid is not a winning one the reward is 0.
            effective_bid_i = float(effective_bids[i])
            r_i = (
                float(self.campaign_values[i]) - effective_bid_i
                if wins[i]
                else 0.0
            )
            # Compute the cost (the effective bid value), if the bid is not a winning one the cost is 0.
            c_i = effective_bid_i if wins[i] else 0.0

            # Update the window information with the new feedback. Add to it the round index, campaign index, bid index, reward and cost observed.
            self.window.append((round_index, i, b_idx, r_i, c_i))
            # Update the number of times the couple (campaign, bid) has been played in the last window_size;
            # and add to the couple (campaign, bid) information the reward obtained and the cost sustained.
            self.window_count[i, b_idx] += 1
            self.window_sum_reward[i, b_idx] += r_i
            self.window_sum_cost[i, b_idx] += c_i
            # Lastly set to true the flag in ever_pulled, regarding the initial exploration phase.
            self.ever_pulled[i, b_idx] = True

        # Evict observations that fell outside the last window_size rounds.
        cutoff = round_index - self.window_size
        # self.window[0][0] is the round index of that oldest observation, 
        # if it fell before the cutoff it must be removed from the window.
        while self.window and self.window[0][0] <= cutoff:
            # Remove the oldest observation from the dequeu window. (round_index is ignored with _)
            _, old_c, old_b, old_r, old_cost = self.window.popleft()
            # By removing an observation, remove also all the information regarding that couple (campaign, bid) 
            # from the window_count, window_sum_reward and window_sum_cost.
            self.window_count[old_c, old_b] -= 1
            self.window_sum_reward[old_c, old_b] -= old_r
            self.window_sum_cost[old_c, old_b] -= old_cost

        # Copy the window_count to n_pulls, used in the select_action method.
        self.n_pulls = self.window_count.astype(float)
        # Take a window_count of at least 1 in order to avoid division by 0.
        counts_safe = np.maximum(self.window_count, 1)
        # Define a mask matrix which identifies all the (campaign, bid) couples that have been played at least once in the current window.
        inwindow = self.window_count > 0
        # If a couple (campaign, bid) has been observed at least once in the current window compute its actual
        # mean reward and mean cost, otherwise set both those values to 0.
        self.mean_reward = np.where(
            inwindow, self.window_sum_reward / counts_safe, 0.0
        )
        self.mean_cost = np.where(
            inwindow, self.window_sum_cost / counts_safe, 0.0
        )

        # Lastly, update the global trackers. Diminish the budget by the cost sustained, and sum the reward and cost observed
        # to the cumulative reward and cost.
        self.budget_left -= cost
        self.cumulative_reward += reward
        self.cumulative_cost += cost
