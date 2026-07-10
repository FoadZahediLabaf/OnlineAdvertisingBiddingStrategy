"""
Change-detector Combinatorial UCB algorithm for Requirement 4.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Implement the change-detector version of Combinatorial-UCB for slightly
    non-stationary multi-campaign environments.

What to implement here:
    - Combinatorial-UCB action selection.
    - Change detection on reward/cost observations.
    - Reset or adaptation logic after detected changes.
    - Budget-aware action selection.
    - Compatibility with campaign conflict constraints.

Do not:
    - Add sliding-window logic here.
    - Add primal-dual logic here.
    - Add experiment-running code here.
    - Change the standard CSV result format here.
"""

# The class is built starting from the, previously implemented, Combinatorial-UCB
# algorithm (combinatorial_ucb.py).
# The change-detector mechanism added on top of CUCB is a CUSUM detector which
# works for each arm separately. Later, it was added also the possibility to select
# a Normal Discrepancy detetctor, which lies outside the scope of the requirement 4 assignment,
# and uses a full-feedback approach, both on the detector and the UCB side.

# The three keypoints of CUSUM-CUCB:
# 1. Per-arm CUSUM detector on the WIN INDICATOR.
#       For each (campaign, bid) arm we monitor the Bernoulli win indicator
#       (1 if the bid won that round, else 0). We don't monitor the raw reward:
#       for a fixed arm the reward is win * (value - bid).
#       By checking the reward, we are more prone to detect changes where the bidding
#       value is low rather than high, because of the way the reward is computed. 
#       Since the value of each campaign is 1, a high value of the bid shrinks the 
#       computed reward and a change in the distribution is harder to detect, rather than
#       the case where the bidding value is low (so the computed reward is a higher value).
#       For this reason we monitor the win value, which is either 1 or 0 and describe the
#       probability of having a bid value higher than the highest competing bid.
#       If the distribution changes, it is less probable to win that auction and
#       so the win variable associated to the played arm will reflect that change. 
#       The metrics is therefore fair with all the arms, leaving aside the bid values (and reward) values.

# 2. Reset on detection.
#       A) When the CUSUM detector of an arm fires, only that arm's statistics
#       (pull count, mean reward, mean cost) and its detector state are reset.
#       All the other arms keep their already gathered data. After a reset the arm is
#       re-explored naturally through the optimistic confidence bound; since the information
#       about that arma are erased from the learner.
#       B) The CUSUM detector reset all the arms when a change is detected, for any possible arm.
#       In this case all the data gathered up to the detection round are erased from the algorithm,
#       and UCB restarts from scratch.

# 3. alpha forced exploration.
#       With probability alpha each round, the agent plays a random super-arm. 
#       This keeps sampling arms that are not in the current optimum so their 
#       detectors keep receiving observations and can still notice a change.

# The logic for dealing with both "chosen_bids" and "effective_bids" (in the update method, for example)
# is quite meaningless since the environment can't re-scale anymore the bid chosen by the learner.
# For this reason the fragment of code dealing with this feature in the environment is dead code, and the
# chosen bids are equal to the effective bids.
# For now, the code is left as it is because still run correctly and is not harmed. If there is spare-time,
# update the environment by removing the discussed lines of code and modify the related logic in CD-UCB and
# SW-UCB.

from collections import deque
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np

# Import from the CUCB version and keep the same auxiliary tolerances.
from ola.algorithms.combinatorial_ucb import (
    BID_SET_TOLERANCE,
    TIE_TOLERANCE,
    CombinatorialUCB,
)


# Normal Discrepancy (ND) core, used ONLY by the optional full-feedback
# detector (detector_type == "nd_full_feedback"). None of the CUSUM logic uses any of this.
def log_det_cov(samples: np.ndarray) -> float:
    """Return log|Sigma| for a (n_samples, n_features) data matrix."""
    cov = np.atleast_2d(np.cov(samples, rowvar=False))
    # Adds a tiny ridge (1e-10 * I) so the covariance is strictly positive-definite even when
    # n_samples ~ n_features.
    cov = cov + np.eye(cov.shape[0]) * 1e-10
    _sign, logdet = np.linalg.slogdet(cov)
    return float(logdet)


def normal_discrepancy(window: np.ndarray) -> float:
    """Normal Discrepancy for one window of shape (W, d)."""
    # Classic Normal Discrepancy algorithm, which take the shape of the window,
    # divide it in a left and right part and return the value computed on the 
    # log det of the parts. 
    W = window.shape[0]
    split = W // 2
    left = window[:split]
    right = window[split:]
    return (
        log_det_cov(window)
        - 0.5 * log_det_cov(left)
        - 0.5 * log_det_cov(right)
    )


class ChangeDetectorCUCB(CombinatorialUCB):
    """Change-detector (CUSUM) Combinatorial-UCB learner for Requirement 4.
    Subclasses the stationary CombinatorialUCB. It selects feasible super-arms
    with the same optimistic reward-UCB / cost-LCB reasoning as CUCB, but each
    (campaign, bid) arm carries a CUSUM detector on its win indicator.
    When a detector fires, that arm's statistics (or all the arms' statistics, in global reset mode)
    are reset so the learner re-learns it (them) from scratch."""

    def __init__(
        self,
        number_of_campaigns: int,
        campaign_values: List[float],
        bid_values: List[float],
        budget: float,
        horizon: int,
        conflict_edges: Optional[List[List[int]]] = None,
        random_seed: Optional[int] = None,
        cusum_M: int = 30,                      # Samples used to estimate the mean to be taken as baseline for the change detection.
        cusum_epsilon: float = 0.1,             # Threshold used to ignore differences from the baseline which are too small to be taken into account.
        cusum_threshold: float = 5.0,           # Actual threshold used to identify a change, if the differences accumulated exceed the threshold, then the change is detected.
        alpha: float = 0.05,                    # Probability to perform an arm picking it from a uniform probability distribution.
        exploration_coefficient: float = 2.0,   # Exploration coefficient as seen also in the sliding window approach. Probabily need to be lowered below 1.0 from experiments results.
        reset_mode: str = "per_arm",            # What to reset on a detection: "per_arm" (only the firing arm) or "global" (every arm).
        detector_type: str = "cusum",           # Which change detector to use: "cusum" (per-arm, bandit) or "nd_full_feedback" (joint Normal Discrepancy on the competing bids -> FULL FEEDBACK).
        nd_window_size: int = 50,               # ND sliding-window length, in rounds.
        nd_threshold: float = 3.0,              # ND alarm threshold: ND ~ 0 under no change, and spikes.
    ) -> None:
        # Build a complete Combinatorial-UCB baseline first.
        super().__init__(
            number_of_campaigns=number_of_campaigns,
            campaign_values=campaign_values,
            bid_values=bid_values,
            budget=budget,
            horizon=horizon,
            conflict_edges=conflict_edges,
            random_seed=random_seed,
        )

        # Validate the new parameters in this subclass.
        if cusum_M <= 0:
            raise ValueError("cusum_M must be positive.")
        if not (0.0 < cusum_epsilon < 1.0):
            raise ValueError("cusum_epsilon must be in (0, 1).")
        if cusum_threshold <= 0:
            raise ValueError("cusum_threshold must be positive.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if exploration_coefficient <= 0:
            raise ValueError("exploration_coefficient must be positive.")
        if reset_mode not in ("per_arm", "global"):
            raise ValueError("reset_mode must be 'per_arm' or 'global'.")
        if detector_type not in ("cusum", "nd_full_feedback"):
            raise ValueError("detector_type must be 'cusum' or 'nd_full_feedback'.")
        if nd_window_size < 4:
            raise ValueError("nd_window_size must be at least 4 (>=2 samples per half-window).")
        if not np.isfinite(nd_threshold):
            raise ValueError("nd_threshold must be a finite number.")

        # Assign the values obtained as parameters.
        self.cusum_M = int(cusum_M)
        self.cusum_epsilon = float(cusum_epsilon)
        self.cusum_threshold = float(cusum_threshold)
        self.alpha = float(alpha)
        self.exploration_coefficient = float(exploration_coefficient)
        self.reset_mode = str(reset_mode)
        self.detector_type = str(detector_type)
        self.nd_window_size = int(nd_window_size)
        self.nd_threshold = float(nd_threshold)

        # Define the shape (number_of_campaigns, number_of_bids) to be used for the matrices.
        shape = (self.number_of_campaigns, len(self.bid_values))
        # Keep tracks of which arm has been played at least M times, so which arms have a meaningful computed mean.
        self.cusum_initialized = np.zeros(shape, dtype=bool)
        # Keep track, for every arm, of the number of times we played it and the sum of the reward obtained,
        # but based on the first M samples.
        self.cusum_M_count = np.zeros(shape, dtype=int)
        self.cusum_M_sum = np.zeros(shape, dtype=float)
        # Reference mean estimated from the warmup samples.
        self.cusum_M_reference = np.zeros(shape, dtype=float)
        # The two CUSUM statistics (positive / negative drift).
        self.cusum_g_plus = np.zeros(shape, dtype=float)
        self.cusum_g_minus = np.zeros(shape, dtype=float)
        # Persistent mask for the initial exploration sweep (never reset,
        # so a detector reset does not force the arms back into the initial sweep).
        self.ever_pulled = np.zeros(shape, dtype=bool)
        # Alarms counters: how many alarms in total and per arm.
        # (detections_per_arm is meaningful for the per-arm CUSUM detector; the
        # joint ND detector only increments the global detected_changes counter.)
        self.detected_changes = 0
        self.detections_per_arm = np.zeros(shape, dtype=int)

        # Sliding window of recent competing-bid vectors m(t).
        # Used only by the "nd_full_feedback" detector; stays empty for CUSUM.
        # maxlen makes it a FIFO sliding window of the last nd_window_size rounds.
        self.nd_window: deque = deque(maxlen=self.nd_window_size)

    def select_action(self) -> Any:
        """Select a feasible super-arm with CUCB optimism (current-regime
        estimates) plus alpha forced exploration."""
        # Find the smallest positive bid; if the budget cannot cover it, pass and don't return any action.
        positive_bids = self.bid_values[self.bid_values > 0.0]
        min_positive_bid = (
            float(positive_bids.min()) if len(positive_bids) > 0 else 0.0
        )
        if self.budget_left < min_positive_bid:
            return {"active_campaigns": [], "bids": []}

        # Initial forced exploration: play every arm once at the start.
        unplayed_action = self.select_unplayed_arm()
        if unplayed_action is not None:
            return unplayed_action

        # Confidence width (exploration bonus).
        # Under the FULL-FEEDBACK ND workflow the estimates are (near-)exact and
        # every arm is observed every round, so there is no need to explore:
        # the width is dropped and the scores use the plain means. 
        # The bandit CUSUM path keeps the standard optimistic width unchanged.
        if self.detector_type == "nd_full_feedback":
            confidence_widths = 0.0
        else:
            # Numerator: the total number of currently-valid samples, i.e. the sum
            # across all arms of the observations kept since each arm's last reset.
            # Denominator: the number of times the learner pulled the considered arm
            # since the last detected change.
            total_valid_samples = max(float(self.n_pulls.sum()), 2.0)
            recent_pulls = np.maximum(self.n_pulls, 1.0)
            confidence_widths = np.sqrt(self.exploration_coefficient * np.log(total_valid_samples) / recent_pulls)
        ucb_reward = self.mean_reward + confidence_widths
        lcb_cost = np.maximum(self.mean_cost - confidence_widths, 0.0)

        # Compute the per-round budget rho = B / T, plus adaptive pacing on the remainder.
        rho = self.budget / self.horizon
        pacing_budget = self.budget_left / max(self.horizon - self.t, 1)

        # Find the best optimistic candidate and also collect all the
        # budget-feasible candidates.
        best_score = -float("inf")
        best_candidates: List[Dict[str, Any]] = []
        feasible_candidates: List[Dict[str, Any]] = []

        # For all the possible campaign combinaition (not forbidden by the conflict graph) 
        # and bid values...
        for sorted_campaigns in self._sorted_feasible_sets:
            for bid_combo in product(
                range(len(self.bid_values)),
                repeat=len(sorted_campaigns),
            ):
                # Compute the score of the candidate by invoking the related method.
                candidate = self._score_candidate(
                    sorted_campaigns=sorted_campaigns,
                    bid_combo=list(bid_combo),
                    ucb_reward=ucb_reward,
                    lcb_cost=lcb_cost,
                    rho=rho,
                    pacing_budget=pacing_budget,
                )
                # If the considered candidate is not feasible, skip it.
                if candidate is None:
                    continue
                # If a candidate is found, append it to the list of feasible candidate.
                feasible_candidates.append(candidate)

                score = float(candidate["score"])
                # If the newly found candidate has a score which is sensibly higher than the previous best candidate,
                # change the best candidate. 
                if score > best_score + TIE_TOLERANCE:
                    best_score = score
                    best_candidates = [candidate]
                # If the newly found score is near the best score found so far, append the new candidate to the list
                # of best candidates.
                elif abs(score - best_score) <= TIE_TOLERANCE:
                    best_candidates.append(candidate)

        # If, at the end of the search, no candidate has been found, don't return any action.
        if not best_candidates:
            return {"active_campaigns": [], "bids": []}

        # Forced exploration: with probability alpha pick a random budget-feasible super-arm.
        # Otherwise, pick randomly from the list of best candidates.
        # Note that the alpha exploration is disabled under the "nd_full_feedback" workflow:
        # with full feedback every arm is estimated every round regardless of what
        # is played. It stays active for the bandit CUSUM path.
        if (
            self.alpha > 0.0
            and self.detector_type != "nd_full_feedback"
            and feasible_candidates
            and float(self.rng.random()) < self.alpha
        ):
            chosen = feasible_candidates[
                int(self.rng.integers(len(feasible_candidates)))
            ]
        else:
            chosen = best_candidates[int(self.rng.integers(len(best_candidates)))]

        # Return the chosen, either picked randomly from all the feasible possibilities or from the list of
        # best candidates, candidate.
        return {
            "active_campaigns": chosen["active_campaigns"],
            "bids": chosen["bids"],
        }

    def select_unplayed_arm(self) -> Optional[Dict[str, Any]]:
        """Initial forced-exploration sweep: return the first affordable arm that
        has NEVER been played, as a single-campaign action; return None when all the 
        action has been played at least once.""" 
        # For every campaign and possible bid value...
        for campaign in range(self.number_of_campaigns):
            for bid_idx, bid in enumerate(self.bid_values):
                # ... If the couple has not been played even once, and it is affordable...
                if (
                    not self.ever_pulled[campaign, bid_idx]
                    and bid <= self.budget_left
                ):
                    # ... Return it.
                    return {
                        "active_campaigns": [campaign],
                        "bids": [float(bid)],
                    }
        return None

    def update(self, action: Any, step_result: Any) -> None:
        """Update the current-regime estimates and run the selected change detector.
        Two workflows, depending on detector_type:
          - "cusum" (BANDIT): update only the played arms and run the per-arm
            CUSUM detector on their win indicators.
          - "nd_full_feedback" (FULL FEEDBACK): update every arm from the observed
            competing bids and run the joint ND detector.
        Same interface as CUCB / SW-CUCB."""
        # Extract the information obtained from the action played and the environment response to that action.
        active_campaigns: List[int] = action["active_campaigns"]
        chosen_bids: List[float] = action["bids"]                   # Has one entry only for the active campaigns.
        wins: List[int] = step_result["wins"]                       # Has one entry for every campaign, value 1 if the campaign was won at the current round, 0 otherwise.
        effective_bids: List[float] = step_result["bids"]           # Has one entry for every campaign (0.0 value for non-active campaigns).
        reward = float(step_result["reward"])
        cost = float(step_result["cost"])

        # Increment the count of the rounds.
        self.t += 1

        if self.detector_type == "nd_full_feedback":
            # FULL-FEEDBACK workflow: knowing the competing-bid vector m(t), 
            # all arms are updated every round -- not only the played ones.
            self.observe_full_feedback(self.read_competing_bids(step_result))
        else:
            # BANDIT workflow (CUSUM / default): update only the played arms with
            # what was actually observed for them.
            # For every played campaign of the round...
            for pos, i in enumerate(active_campaigns):
                # ... get the relative bid value.
                chosen_bid = float(chosen_bids[pos])
                # Find the index of the bid played in the selected campaign.
                matching_indices = np.where(
                    np.abs(self.bid_values - chosen_bid) <= BID_SET_TOLERANCE
                )[0]
                if len(matching_indices) == 0:
                    # Skip the bid (and the campaign) if the index is not found, so the bid apparently is
                    # not selected from the set of possible bids.
                    continue
                b_idx = int(matching_indices[0])

                effective_bid_i = float(effective_bids[i])
                win_i = 1.0 if wins[i] else 0.0
                # Compute the specific reward (and cost) obtained by playing the considered arm.
                r_i = (
                    float(self.campaign_values[i]) - effective_bid_i
                    if wins[i]
                    else 0.0
                )
                c_i = effective_bid_i if wins[i] else 0.0

                self.observe_arm(i, b_idx, r_i, c_i, win_i)

        # Update the global trackers.
        self.budget_left -= cost
        self.cumulative_reward += reward
        self.cumulative_cost += cost

        # Joint Normal Discrepancy detection (only when it is the selected
        # detector). It exploit the FULL-FEEDBACK competing-bid vector.
        # The CUSUM detector can never access that information, so it remains
        # a bandit-feedback scenario.
        if self.detector_type == "nd_full_feedback":
            self.nd_detect(step_result)

    def read_competing_bids(self, step_result: Any) -> np.ndarray:
        """Extract the full-feedback competing-bid vector m(t) from the environment
        step result (info["debug"]["highest_competing_bids"]). Shared by the
        full-feedback estimate update and the ND detector; raises if the channel is
        absent (i.e. the environment is not exposing full feedback)."""
        info = step_result.get("info", {}) if isinstance(step_result, dict) else {}
        debug = info.get("debug", {}) if isinstance(info, dict) else {}
        competing = debug.get("highest_competing_bids")
        if competing is None:
            raise ValueError(
                "detector_type='nd_full_feedback' requires the environment to expose "
                "the competing bids under info['debug']['highest_competing_bids'] "
                "(full feedback)."
            )
        return np.asarray(competing, dtype=float)

    def observe_full_feedback(self, competing_bids: np.ndarray) -> None:
        """full-feedback estimate update (used only by the ND full-feedback
        workflow). Given the competing-bid vector m(t), the reward
        and cost of every (campaign, bid) arm are known exactly, so all arms are updated every
        round and not just the played ones."""
        m = np.asarray(competing_bids, dtype=float)
        # would_win[i, b] = 1 if bid b would have won campaign i this round.
        would_win = (self.bid_values[None, :] >= m[:, None]).astype(float) 
        # Per-arm reward (value - bid on a win) and cost (bid on a win).
        r = (self.campaign_values[:, None] - self.bid_values[None, :]) * would_win
        c = self.bid_values[None, :] * would_win
        # Every arm receives an observation this round: mark it, bump the count,
        # and apply the incremental mean update to all arms at once.
        self.ever_pulled[:, :] = True
        self.n_pulls += 1.0
        self.mean_reward += (r - self.mean_reward) / self.n_pulls
        self.mean_cost += (c - self.mean_cost) / self.n_pulls

    def nd_detect(self, step_result: Any) -> None:
        """Full-feedback Normal Discrepancy detector.
        Reads the competing-bid vector m(t) from the environment's
        full-feedback channel (info["debug"]["highest_competing_bids"]), pushes it
        onto the sliding window, and once the window holds nd_window_size rounds
        computes the joint ND statistic.
        Since the detector is global, the only possible reset option is the global one."""
        # Append the current competing-bid vector m(t) to the window (raises if the
        # full-feedback channel is absent).
        self.nd_window.append(self.read_competing_bids(step_result))

        # If the window is not yet fully populated, don't even compute the ND statistic.
        if len(self.nd_window) < self.nd_window_size:
            return

        # Create a matrix with the constructed window. Shape (nd_window_size, number of campaigns).
        window = np.vstack(self.nd_window) 
        # Compute the ND metric of the current window.
        nd_value = normal_discrepancy(window)
        # If the metric surpass the threshold, detect a change.
        if nd_value > self.nd_threshold:
            self.detected_changes += 1
            # Fire a global restart of the learner.
            self.reset_all()
            # Clear the window from the old bids related to a previous distribution.
            self.nd_window.clear()

    def observe_arm(
        self,
        campaign: int,
        bid_idx: int,
        reward: float,
        cost: float,
        win_signal: float,
    ) -> None:
        """Feed the received observation of an arm to both the CD-UCB statistics and
        the CUSUM detector. If the detector detects a change, reset the information
        only about that particular arm."""
        # Update the current estimates incrementally.
        self.ever_pulled[campaign, bid_idx] = True
        self.n_pulls[campaign, bid_idx] += 1
        count = self.n_pulls[campaign, bid_idx]
        self.mean_reward[campaign, bid_idx] += (
            reward - self.mean_reward[campaign, bid_idx]
        ) / count
        self.mean_cost[campaign, bid_idx] += (
            cost - self.mean_cost[campaign, bid_idx]
        ) / count

        # If the selected detector is not CUSUM (so Normal Discrepancy), the below code is skipped.
        # It is executed only if the detector is CUSUM.
        # With the latest implementation, this method is directly invoked only in the case of the CUSUM
        # detector, so this check is not necessary.
        if self.detector_type != "cusum":
            return

        # Feed the CUSUM detector with the win indicator.
        if not self.cusum_initialized[campaign, bid_idx]:
            # Warmup phase: accumulate the first M samples to estimate the reference mean.
            self.cusum_M_sum[campaign, bid_idx] += win_signal
            self.cusum_M_count[campaign, bid_idx] += 1
            # If we reach the amount of M samples gathered, effectively compute the reference mean and set up the
            # auxiliary values of CUSUM (g+, g-).
            if (
                self.cusum_M_count[campaign, bid_idx]
                >= self.cusum_M
            ):
                self.cusum_M_reference[campaign, bid_idx] = (
                    self.cusum_M_sum[campaign, bid_idx]
                    / self.cusum_M_count[campaign, bid_idx]
                )
                self.cusum_initialized[campaign, bid_idx] = True
                self.cusum_g_plus[campaign, bid_idx] = 0.0
                self.cusum_g_minus[campaign, bid_idx] = 0.0
            return

        # If the considered arm has already gathered M samples, we execute the actual core of CUSUM.
        # Take the reference mean, compute the positive and negative difference from the reference mean,
        # add the newly found differences to the two metrics g+ and g-.
        reference = self.cusum_M_reference[campaign, bid_idx]
        s_plus = win_signal - reference - self.cusum_epsilon
        s_minus = reference - win_signal - self.cusum_epsilon
        self.cusum_g_plus[campaign, bid_idx] = max(
            0.0, self.cusum_g_plus[campaign, bid_idx] + s_plus
        )
        self.cusum_g_minus[campaign, bid_idx] = max(
            0.0, self.cusum_g_minus[campaign, bid_idx] + s_minus
        )

        # If either one of the two metrics surpass the trheshold, detect a change and reset.
        # The counters always record which arm fired the alarm; and based on the reset mode...
        # reset_mode: "per_arm" forgets only the firing arm, "global" restarts every arm
        if (
            self.cusum_g_plus[campaign, bid_idx] > self.cusum_threshold
            or self.cusum_g_minus[campaign, bid_idx] > self.cusum_threshold
        ):
            self.detected_changes += 1
            self.detections_per_arm[campaign, bid_idx] += 1
            if self.reset_mode == "global":
                self.reset_all()
            else:
                self.reset_arm(campaign, bid_idx)

    def reset_arm(self, campaign: int, bid_idx: int) -> None:
        """Forget everything about an arm (campaign, bid) after a detected change 
        (possible only for CUSUM): its CUCB estimates and its detector state.
        Note that ever_pulled is not resetted, the arm won't pass through the initial
        exploration phase multiple times."""
        self.n_pulls[campaign, bid_idx] = 0
        self.mean_reward[campaign, bid_idx] = 0.0
        self.mean_cost[campaign, bid_idx] = 0.0

        # Reset the information about the first M samples in order to recompute the baseline
        # empirical mean. As well as the accumulated differences (g+ and g-) for the specific arm.
        self.cusum_initialized[campaign, bid_idx] = False
        self.cusum_M_count[campaign, bid_idx] = 0
        self.cusum_M_sum[campaign, bid_idx] = 0.0
        self.cusum_M_reference[campaign, bid_idx] = 0.0
        self.cusum_g_plus[campaign, bid_idx] = 0.0
        self.cusum_g_minus[campaign, bid_idx] = 0.0

    def reset_all(self) -> None:
        """Global reset: forget every arm's CUCB estimates and detector state at
        once, used when reset_mode == "global".
        Note that ever_pulled is not cleared, so the initial exploration sweep
        does not re-run."""
        self.n_pulls[:, :] = 0
        self.mean_reward[:, :] = 0.0
        self.mean_cost[:, :] = 0.0

        self.cusum_initialized[:, :] = False
        self.cusum_M_count[:, :] = 0
        self.cusum_M_sum[:, :] = 0.0
        self.cusum_M_reference[:, :] = 0.0
        self.cusum_g_plus[:, :] = 0.0
        self.cusum_g_minus[:, :] = 0.0
