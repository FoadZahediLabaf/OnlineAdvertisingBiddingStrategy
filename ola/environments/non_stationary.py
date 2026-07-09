"""
Slightly non-stationary multi-campaign environment for Requirement 4.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Do not:
    - Add Requirement 1 single-campaign logic here.
    - Add Requirement 2 stationary-only environment logic here.
    - Add Requirement 3 adversarial full-feedback logic here.
    - Add sliding-window or change-detector algorithm logic here.
    - Change the standard step output format.
"""

# Design notes:
#    The feedback model is different as the one seen in requirement 3, here 
#    we can't assume full feedback, but instead we must deal with bandit feedback. 
#    The learner will observe only the set of won auctions and the resulting reward/cost.
#    Still, the realised highest competing bids are available as a pre-computed sequence 
#    info["debug"]. But nonetheless, the adaptive CUCB learners must not use them.
#    This information are only accessed by the change detection algorithm, only if the 
#    selected detector is Normal Discrepancy; which is outside the scope of the requirement
#    4 assignment, so we decided, out of curiosity, to implement a third option with full-feedback.
#    In this last case, the full-feedback approach is exploited by both the detector algorithm
#    and the UCB estimates.

#    The key difference required by Requirement 4 is the "slightly" non-stationary
#    structure: rounds are partitioned into a small number of fixed intervals,
#    each interval keeps a fixed distribution, and each interval uses a different
#    distribution. The distribution of every interval is drawn from a Gaussian 
#    constellation generated with the tools in ola/utils/constellation: 
#    note that the pairwise symmetric-KL divergence is controlled, so the difference 
#    between intervals can be tuned.

#    The interaction interface mirrors the Requirement 2 MultiCampaignEnvironment
#    so that the sliding-window / change-detector CUCB learners (which subclass the
#    Req 2 Combinatorial-UCB) run unchanged. 

#    For convenience the step method also accepts a plain per-campaign bid vector 
#    (the Req 3 action format), so the primal-dual learner can submit its actions to 
#    the very same environment. Note that this only concerns the action format: the 
#    feedback returned is the bandit feedback described above, the same for every learner.
#    Unsed feature at the end of the algorithm, never submitted this option to the environment.

#    It's also been implemented a replay mode which takes as input a pre-computed dataset so that every algorithm
#    can be run on the exact same realization and it is not required to re-generate the same
#    thing over and over again, checking that the realized outcome is the same. 

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ola.environments.base import BaseEnvironment
from ola.utils.bidding import first_price_cost, first_price_reward, wins_auction
from ola.utils.constellation.constellation import (
    generate_constellation_gd_rotation,
    generate_constellation_gd_rotation_range,
)
from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


BID_SET_TOLERANCE = 1e-9


class NonStationaryMultiCampaignEnvironment(BaseEnvironment):
    """Slightly non-stationary multi-campaign auction environment.
    In the classic case, rounds are partitioned into number_of_intervals consecutive intervals.
    Within an interval the highest competing bids are sampled i.i.d. from a fixed
    joint distribution (a multivariate Gaussian clipped to [0, 1]); each
    interval uses a different Gaussian taken from a controlled constellation.

    The environment supports two modes:
    - GENERATE mode (default): the highest-competing-bid sequence is sampled from
      the Gaussian constellation as described above.
    - REPLAY mode: when competing_bid_sequence is passed, that pre-computed
      (horizon, number_of_campaigns) matrix is replayed exactly and the
      constellation / sampling machinery is bypassed."""

    # Remember: the values below are used by default, if during the invocation of the constructor these parameters
    # are specified, their specified values override the ones seen below. If the constructor, on the other hand, 
    # is invoked without specifying some parameters, then the default values are used.
    def __init__(
        self,
        number_of_campaigns: int = 3,
        campaign_values: Optional[Sequence[float]] = None,                          # Winning values for each campaign
        bid_values: Optional[Sequence[float]] = None,                               # Set of allowed bid values.
        budget: float = 500.0,
        horizon: int = 1000,    
        number_of_intervals: int = 4,                                               # Number of intervals in which the horizon is partitioned. Expose different probability distributions of the highest competing bids.
        interval_lengths: Optional[Sequence[int]] = None,                           # Used if the length of the intervals are to be fixed explicitly.
        conflict_edges: Optional[Sequence[Sequence[int]]] = None,                   # Conflict graph between campaigns.
        target_skl: float = 2.0,                                                    # Defines how much the various distributions differ from one another.
        skl_range: Optional[Tuple[float, float]] = None,                            # Exploit a range of SKL divergences instead of a singel fixed value.
        constellation: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,    # Manually provided constellation of gaussians for the intervals, optional. Each element should be a pair (mu/mean, sigma/covariance matrix).
        constellation_seed: Optional[int] = None,                                   # Seed used for generating constellation, use for reproducibility.
        constellation_max_iter: int = 5000,
        max_std: float = 0.1,
        clip_low: float = 0.0,
        clip_high: float = 1.0,
        random_seed: Optional[int] = None,                                          # Seed used for the random generation of competing bids, use for reproducibility.
        competing_bid_sequence: Optional[np.ndarray] = None,                        # Optional pre-computed (horizon, number_of_campaigns) matrix of highest competing bids.
        phase_ids: Optional[Sequence[int]] = None,                                  # Optional per-round regime id (length horizon), only used together with competing_bid_sequence for regime markers / oracle. If omitted the whole run is treated as a single phase.
    ) -> None:
        # Default checks that mirror the Req 2 multi-campaign environment.
        if campaign_values is None:
            campaign_values = [1.0] * number_of_campaigns
        if bid_values is None:
            # If no bid values are provided, the default is 11 evenly spaced bids values in [0,1].
            bid_values = [round(b, 1) for b in np.linspace(0.0, 1.0, 11)]

        # Checks for the feasibility of the provided parameters.
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
        if number_of_intervals <= 0:
            raise ValueError("number_of_intervals must be positive.")
        if number_of_intervals > horizon:
            raise ValueError("number_of_intervals cannot exceed the horizon.")
        if target_skl < 0:
            raise ValueError("target_skl must be non-negative.")
        if max_std <= 0:
            raise ValueError("max_std must be positive.")
        if clip_low >= clip_high:
            raise ValueError("clip_low must be strictly smaller than clip_high.")

        # Assigne the parameters to the actual variables after the checks have passed.
        self.number_of_campaigns = int(number_of_campaigns)
        self.campaign_values = np.asarray(campaign_values, dtype=float)
        self.bid_values = np.asarray(sorted(bid_values), dtype=float)
        self.initial_budget = float(budget)                             # Save the initial value of the available budget.
        self.horizon = int(horizon)
        self.target_skl = float(target_skl)
        self.skl_range = skl_range
        self.constellation_seed = constellation_seed
        self.constellation_max_iter = int(constellation_max_iter)
        self.max_std = float(max_std)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.random_seed = random_seed

        # Check that all the possible rewards associated to the campaigns are non-negative.
        if np.any(self.campaign_values < 0.0):
            raise ValueError("Campaign values must be non-negative.")

        # Take the conflict edges present in the input parameter and build the conflict graph (invoking the imported method) based on those.
        self.conflict_edges: List[List[int]] = (
            [list(e) for e in conflict_edges] if conflict_edges else []
        )
        self.conflict_graph = build_conflict_graph(
            number_of_campaigns=self.number_of_campaigns,
            edges=[tuple(e) for e in self.conflict_edges],
        )

        # Set the random number generator using the provided seed.
        self.rng = np.random.default_rng(self.random_seed)

        if competing_bid_sequence is not None:
            # REPLAY MODE
            # A pre-computed sequence of highest competing bids is injected and
            # replayed exactly. The constellation / per-round sampling machinery
            # is bypassed entirely.
            self._init_from_sequence(competing_bid_sequence, phase_ids)
        else:
            # GENERATE MODE (default)
            # Construct the actual interval lengths (and number of intervals) of the environmet
            # by invoking the related method with the parameters obtained.
            self.interval_lengths = self._build_interval_lengths(
                number_of_intervals=int(number_of_intervals),
                interval_lengths=interval_lengths,
            )
            self.number_of_intervals = len(self.interval_lengths)

            # interval_boundaries[k] = first round (0-based) of interval k;
            # an extra trailing entry equal to the horizon closes the last interval.
            # So, computes the starting round of each interval.
            self.interval_boundaries = np.concatenate(
                ([0], np.cumsum(self.interval_lengths))
            ).astype(int)

            # Build the constellation of gaussians which will be used for the various intervals.
            self.constellation = self._build_constellation(constellation)

            # Create a matrix to store all the highest competing bids for each round and campaign.
            self.highest_competing_bids = np.zeros(
                (self.horizon, self.number_of_campaigns), dtype=float
            )
            # Create a vector to store the interval index for each round.
            self.phase_ids = np.zeros(self.horizon, dtype=int)

            # Fill the highest_competing_bids and phase_ids matrices sampling from the generated constellation.
            self._generate_sequence()

        # Reset the environment to the initial state, granting the possibility to re-use the exact same scenario
        # with different algorithm multiple times.
        self.reset()

    # CONSTRUCTORS HELPERS CALLED DURING INITIALIZATION.

    def _init_from_sequence(
        self,
        competing_bid_sequence: np.ndarray,
        phase_ids: Optional[Sequence[int]],
    ) -> None:
        """REPLAY-mode initialisation from a pre-computed competing-bid sequence.
        Validates the injected (horizon, number_of_campaigns) matrix, stores
        it as the highest-competing-bid sequence, and derives the interval /
        phase metadata from phase_ids (or a single phase when it is omitted)."""
        sequence = np.asarray(competing_bid_sequence, dtype=float)
        if sequence.shape != (self.horizon, self.number_of_campaigns):
            raise ValueError(
                "competing_bid_sequence must have shape "
                f"(horizon, number_of_campaigns) = ({self.horizon}, "
                f"{self.number_of_campaigns}); got {sequence.shape}. "
                "Pass matching horizon and number_of_campaigns."
            )
        if not np.all(np.isfinite(sequence)):
            raise ValueError("competing_bid_sequence contains non-finite values.")

        # Defensive clip to the competing-bid box (the shared realizations are
        # already in [0, 1], so this is normally a no-op).
        self.highest_competing_bids = np.clip(sequence, self.clip_low, self.clip_high)
        # No constellation is generated in replay mode, so set it to None.
        self.constellation = None

        # If the phase_ids are provided, validate them and compute the metadata of the intervals,
        # otherwise consider a single phase for the whole run.
        if phase_ids is not None:
            phases = np.asarray([int(p) for p in phase_ids], dtype=int)
            if phases.shape != (self.horizon,):
                raise ValueError(
                    "phase_ids must have length horizon "
                    f"({phases.shape[0]} != {self.horizon})."
                )
            self.phase_ids = phases
            # Interval metadata = contiguous runs of equal phase id (the change
            # points of the injected schedule), so plots / oracles can use them.
            change_points = np.where(np.diff(phases) != 0)[0] + 1
            self.interval_boundaries = np.concatenate(
                ([0], change_points, [self.horizon])
            ).astype(int)
            self.interval_lengths = np.diff(self.interval_boundaries).astype(int)
            self.number_of_intervals = len(self.interval_lengths)
        else:
            # No regime information supplied: treat the whole run as one phase.
            self.phase_ids = np.zeros(self.horizon, dtype=int)
            self.interval_lengths = np.array([self.horizon], dtype=int)
            self.interval_boundaries = np.array([0, self.horizon], dtype=int)
            self.number_of_intervals = 1

    def _build_interval_lengths(
        self,
        number_of_intervals: int,
        interval_lengths: Optional[Sequence[int]],
    ) -> np.ndarray:
        """Return the length (in rounds) of every interval, as a vector.
        If explicit lengths are given they must be positive and sum to the
        horizon. Otherwise, if the interval lengths are not given, the 
        horizon is split as evenly as possible, with any remainder assigned 
        to the earliest intervals."""
        # If the interval lengths are explicitly provided, use them to create the interval lenghts vector.
        if interval_lengths is not None:
            lengths = np.asarray([int(x) for x in interval_lengths], dtype=int)
            # If a value for an interval length is negative or zero, return an error.
            if np.any(lengths <= 0):
                raise ValueError("Every interval length must be positive.")
            # If the sum of all the lengths of the intervals is differnt from the total time horizon, return an error.
            if int(lengths.sum()) != self.horizon:
                raise ValueError(
                    "interval_lengths must sum to the horizon "
                    f"({int(lengths.sum())} != {self.horizon})."
                )
            # If the checks are passed, return the interval lengths vector, with the values explicitly provided.
            return lengths

        # If the interval lengths are not specified, divide as evenly as possible the time horizon among the number of intervals.
        # Divide the horizon into equal parts, and then compute the remainder which will be assigned to the first intervals.
        base = self.horizon // number_of_intervals
        remainder = self.horizon % number_of_intervals
        # Create the vector lengths (with number_of_intervals cells) filled with the value of base.
        lengths = np.full(number_of_intervals, base, dtype=int)
        # Add the remainder starting from the first intervals until finished.
        lengths[:remainder] += 1
        return lengths

    def _build_constellation(
        self,
        constellation: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build one (mu, sigma) Gaussian for every interval. 
        The couples are (mean vector, covariance matrix) of the Gaussian distribution.
        When an explicit constellation is supplied it is validated and used
        as it is. Otherwise, the constellation is generated with the builder from
        ola.utils.constellation."""
        # Save the dimension of each Gaussians distribution. It must be equal to the number of campaigns. If we have
        # 3 campaigns, we need to have a 3-dimensional Gaussian distribution: accordingly the Gaussian should have
        # a mean vector of 3 elements and a covariance matrix which is 3x3.
        dim = self.number_of_campaigns

        # If a constellation is provided in input, check that it is a valid one and return it.
        if constellation is not None:
            # First convert the given Gaussian into arrays of floats. Each element of the list modes will be a pair (mu, sigma) associated to an interval.
            # So, every element of modes describes the possible behaviour of the highest competing bids, in the various campaigns.
            modes = [
                (np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float))
                for mu, sigma in constellation
            ]
            # From specification, we should have a number of intervals all of them which expose different behaviours of the highest competing bids.
            # So, this means we must have a number of modes (length of modes) equal to the number of intervals. If that doesn't happen, return an error.
            if len(modes) != self.number_of_intervals:
                raise ValueError(
                    "Provided constellation must have exactly one Gaussian per "
                    f"interval ({len(modes)} != {self.number_of_intervals})."
                )
            # Lastly, for every possible behaviour of the highest competing bids, the number of elements in the mean vector must be equal to the number of campaings,
            # and the covariance matrix must be number_of_campaigns x number_of_campaigns. If that doesn't happen, return an error.
            for mu, sigma in modes:
                if mu.shape != (dim,) or sigma.shape != (dim, dim):
                    raise ValueError(
                        "Each constellation entry must be a (mu, sigma) pair with "
                        f"mu of shape ({dim},) and sigma of shape ({dim}, {dim})."
                    )
            # If the checks are passed, return the constellation provided in input.
            return modes

        # Instead, if there is not a provided constellation, generate it.
        # In particular, if the skl_range is provided invoke the related method for generating the constellation, otherwise invoke the alternative
        # method which take as a parameter the specific skl_target.
        if self.skl_range is not None:
            result = generate_constellation_gd_rotation_range(
                dim=dim,
                n_modes=self.number_of_intervals,
                skl_range=self.skl_range,
                max_iter=self.constellation_max_iter,
                seed=self.constellation_seed,
                max_std=self.max_std,
            )
        else:
            result = generate_constellation_gd_rotation(
                dim=dim,
                n_modes=self.number_of_intervals,
                target_skl=self.target_skl,
                max_iter=self.constellation_max_iter,
                seed=self.constellation_seed,
                max_std=self.max_std,
            )

        # Define the modes vector as in the previous case. As, a list of couples (mu, sigma) associated to the various intervals.
        modes = [
            (np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float))
            for mu, sigma in result["constellation"]
        ]
        # The builder may return fewer modes than requested only in degenerate
        # cases; guard against it so the interval -> mode mapping stays valid.
        # Use RuntimeError beacuse it is associated to an unexpected behaviour of the constellation builder, and not a wrong input parameter.
        if len(modes) != self.number_of_intervals:
            raise RuntimeError(
                "Constellation generation returned "
                f"{len(modes)} modes, expected {self.number_of_intervals}."
            )
        # If the check about the generated modes is passed, the new constellation is returned.
        return modes

    def _generate_sequence(self) -> None:
        """Pre-compute the highest competing bids and the phase id of each round."""
        # For every time instant, until the time horizon is completed...
        for round_index in range(self.horizon): 
            # ... get the current phase by invoking the related method.
            phase = self.get_current_phase(round_index)
            # Knowing the phase we are in, get the mean vector and covariance matrix of the Gaussian distribution associated to the current phase.
            mean, covariance = self.constellation[phase]
            # Then, randomly (exploit the random generator of the constructor, so it is reproducibile thanks to the random_seed) 
            # sample the highest competing bids of the round, based on the related Gaussian distribution characteristics.
            sample = self.rng.multivariate_normal(mean, covariance)
            # Save the highest competing bids sampled in the related matrix. Clip allows to keep the values inside the boundary [0,1].
            self.highest_competing_bids[round_index] = np.clip(
                sample, self.clip_low, self.clip_high
            )
            # Lastly, save the associated phase index for every time instant inside the phase_ids vector.
            self.phase_ids[round_index] = phase

    # PUBLIC QUERY HELPERS

    def get_current_phase(self, round_index: int) -> int:
        """Map a 0-based round index to its interval (phase) index."""
        # If the round_index is outside the feasible value range, return an error.
        if round_index < 0 or round_index >= self.horizon:
            raise ValueError("round_index out of range.")
        # searchsorted sorts interval_boundaries and then finds the correct interval for round_index.
        # If the round_index is exactly on a boundary, it is assigned to the right interval.
        return int(np.searchsorted(self.interval_boundaries, round_index, side="right") - 1)

    def get_competing_bids(self, round_index: int) -> np.ndarray:
        """Return the full (all campaigns) highest-competing-bid vector for round_index."""
        # If the round_index is outside the feasible value range, return an error.
        if round_index < 0 or round_index >= self.horizon:
            raise ValueError("round_index out of range.")
        return self.highest_competing_bids[round_index]

    def evaluate_bids(
        self,
        bids: Sequence[float],                              # Sequence of bids for each campaign.
        highest_competing_bids: np.ndarray,                 # Array of the highest competing bids for the current round.
        active_campaigns: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-campaign wins, rewards and costs for a full bid vector.
        Only the campaigns in active_campaigns are considered; when it is
        None every campaign with a strictly positive bid participates. 
        This method is used by the runner as an offline oracle which uses the known 
        competing-bid sequence. It is not used by the learners, but it is rather an experimenter tool.
        """
        # Create a NumPy array of float, based on the bids vector.
        bids = np.asarray(bids, dtype=float)
        # If the shape of the bids vector is different from the number of campaigns, return an error.
        if bids.shape != (self.number_of_campaigns,):
            raise ValueError("bids must contain one entry per campaign.")
        # If the active campaigns are not specified, get directly the indices of the active ones by considering only the campaigns with a strictly positive bid.
        if active_campaigns is None:
            active_campaigns = np.where(bids > 0.0)[0].tolist()

        # Create the vectors which will be returned, having shape equal to the number of campaigns, filled with zeros.
        wins = np.zeros(self.number_of_campaigns, dtype=int)
        rewards = np.zeros(self.number_of_campaigns, dtype=float)
        costs = np.zeros(self.number_of_campaigns, dtype=float)
        # For every active campaign...
        for i in active_campaigns:
            # ... invoke the related method to check whether the auction is won or not, and save the result in the wins vector, saved as an int.
            wins[i] = int(wins_auction(bids[i], highest_competing_bids[i]))
            # Then, compute the reward and cost for the campaign invoking the related methods. Save the results in the rewards and costs vectors.
            rewards[i] = first_price_reward(
                value=float(self.campaign_values[i]),
                bid=float(bids[i]),
                competing_bid=float(highest_competing_bids[i]),
            )
            costs[i] = first_price_cost(
                bid=float(bids[i]),
                competing_bid=float(highest_competing_bids[i]),
            )
        # Lastly, return the three computed vectors.
        return wins, rewards, costs

    # Base Environment Interfaces

    def reset(self) -> Dict[str, Any]:
        """Reset round/budget counters. The competing-bid sequence is preserved.
        The pre-computed sequence is intentionally not regenerated so that
        several learners can be compared on the exact same non-stationary
        realisation."""
        # Reset to the initial values the variables we want to modify another time with a different algorithm.
        self.current_round = 0
        self.budget_left = self.initial_budget
        self.done = False

        # Return then all the other information, which remain the same as before. For example: same sampled highest_competing_bids.
        return {
            "round": self.current_round,
            "budget_left": self.budget_left,
            "done": self.done,
            "info": {
                "number_of_campaigns": self.number_of_campaigns,
                "campaign_values": self.campaign_values.tolist(),
                "bid_values": self.bid_values.tolist(),
                "horizon": self.horizon,
                "conflict_edges": self.conflict_edges,
                "number_of_intervals": self.number_of_intervals,
                "interval_lengths": self.interval_lengths.tolist(),
                "interval_boundaries": self.interval_boundaries.tolist(),
                "initial_budget": self.initial_budget,
            },
        }

    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one non-stationary multi-campaign first-price auction round.
        The possible actions are either a super-arm dictionary, for CUCB learners,
        or a N-length per-campaign bid vector (used for requirement 3 primal-dual action format algorithm).
        Useful in the case of using both req 3 and req 4 algorithms with the same environment.
        At the end, we opted for generating externally the realizations and not submit directly the algorithms
        to the same environment. But still since it's harmless it is left as is."""
        # If the environment is consumed (budget depleted or horizon terminated), return an error.
        if self.done:
            raise RuntimeError(
                "Cannot call step() after the environment is done; call reset()."
            )

        # First of all, invoke the _normalise_action in order to normalise to a standard format the input received.
        # Standard format used is: active_campaigns + effective_bids.
        active_campaigns, effective_bids = self._normalise_action(action)

        # Scale bids proportionally if their total exceeds the remaining budget, to satisfy the budget requirement. 
        # Redundant because the learners don't ever submit bids that exceed the budget. Problem solved at the root, but
        # left here as added safety check.
        total_requested = float(effective_bids.sum())
        if total_requested > self.budget_left and total_requested > 0.0:
            effective_bids = effective_bids * (self.budget_left / total_requested)

        # Obtain the vector of highest_competing_bids of the current round, invoking the related method.
        competing_bids = self.get_competing_bids(self.current_round)
        # Invoke the method to evaluate the bids, giving as parameters the bids chosen by the action, the highest competing bids of the round and the
        # active campaigns for the action. 
        wins, rewards, costs = self.evaluate_bids(
            bids=effective_bids,
            highest_competing_bids=competing_bids,
            active_campaigns=active_campaigns,
        )
        # Sum all the rewards and costs obtained by the action in the current round.
        reward = float(rewards.sum())
        cost = float(costs.sum())

        # Get the id of the interval related to the current round.
        phase = int(self.phase_ids[self.current_round])
        # Update the value left for the budget by subtracting the costs sustained in the round.
        self.budget_left = max(self.budget_left - cost, 0.0)
        # Increase the round counter by one.
        self.current_round += 1
        # Rise the flag done only if the current round surpassed the horizon or if the budget is depleted. 
        # Use a small value for the budget as tolerance, for possible problems with floating-point arithmetic.  
        self.done = (
            self.current_round >= self.horizon or self.budget_left <= 1e-9
        )

        # Return the result of the round as a dictionary.
        return {
            "round": self.current_round,
            "action": action,
            "bids": effective_bids.tolist(),
            "wins": wins.tolist(),
            "reward": reward,
            "cost": cost,
            "budget_left": self.budget_left,
            "done": self.done,
            "info": {
                "active_campaigns": list(active_campaigns),
                "campaign_values": self.campaign_values.tolist(),
                "phase_id": phase,
                # NOTE: even if the highest competing bids are returned, they must not be used by the SW/CUSUM learners of requirement 4. 
                # The full-feedback information are allowed only for requirement 3 algorithm, and it is used for the Normal Discrepancy
                # algorithm which lies outside the scope of the requirement 4 assignment.
                "debug": {
                    "highest_competing_bids": competing_bids.tolist(),
                },
            },
        }

    # Internal Helpers

    def _normalise_action(
        self, action: Any
    ) -> Tuple[List[int], np.ndarray]:
        """Convert any supported action (either req 3 or req 2/4) into (active_campaigns, bid_vector).
        bid_vector always has length N but with zeros on inactive campaigns.
        Abandoned: the idea to submit different algorithms to the same environment was abandoned, but still left as is since it is harmless."""

        # Check whether the action is from a req 4 learner, which is in a form of a dictionary, and invoke the normalising method accordingly.
        if isinstance(action, dict):
            return self._normalise_superarm(action)
        # If the action format is not a dictionary, it comes from a req 3 learner so invoke the alternative method.
        return self._normalise_bid_vector(action)

    def _normalise_superarm(
        self, action: Dict[str, Any]
    ) -> Tuple[List[int], np.ndarray]:
        """Validate a Req 2/4 super-arm dictionary and expand it to a full active campaigns list and bid vector."""
        # Extract the active campaigns list and bid vector from the action obtained as parameter.
        active_campaigns: List[int] = list(action["active_campaigns"])
        bids_for_active: List[float] = list(action["bids"])

        # Check that there is a bid for every active campaign and that there is not a campaign played more than once.
        if len(active_campaigns) != len(bids_for_active):
            raise ValueError("active_campaigns and bids must have the same length.")
        if len(set(active_campaigns)) != len(active_campaigns):
            raise ValueError("active_campaigns must not contain duplicates.")
        
        # Check that the active_campaigns list does not contain an out of bound index.
        for c in active_campaigns:
            if not 0 <= c < self.number_of_campaigns:
                raise ValueError("active_campaigns contains an out-of-range index.")
        
        # Check that the active bids are both in the feasible set of values, and in particular is a value inside the effective bid set.
        for b in bids_for_active:
            if not 0 <= b <= 1:
                raise ValueError("Bid must be in [0, 1].")
            if not self._is_in_bid_set(b):
                raise ValueError("Bid must belong to the discrete bid set.")
            
        # Check whether the active campaigns satisfy the conflict graph constraints or not. 
        if not is_feasible_campaign_set(active_campaigns, self.conflict_graph):
            raise ValueError(
                "Campaign set is infeasible: two or more selected campaigns "
                "are non-compatible."
            )

        # If the checks are passed, define the resulting bid vector and return it with the list of active campaigns.
        bid_vector = np.zeros(self.number_of_campaigns, dtype=float)
        for c, b in zip(active_campaigns, bids_for_active):
            bid_vector[c] = float(b)
        return active_campaigns, bid_vector

    def _normalise_bid_vector(self, action: Any) -> Tuple[List[int], np.ndarray]:
        """Validate a N-length per-campaign bid vector (related to Req 3 format).
        Never really used since the idea to submit req 3 algorithms to the same environment was discarded."""
        # Define the bid vector as an array of float, given action, and flatten it.
        bid_vector = np.asarray(action, dtype=float).reshape(-1)
        # If the shape of the defined bid vector is different from the number of campaigns, return an error.
        if bid_vector.shape != (self.number_of_campaigns,):
            raise ValueError("Bid vector must contain one bid per campaign.")
        # Check whether a bid is outside of the feasible interval.
        if np.any(bid_vector < 0.0):
            raise ValueError("Bids must be non-negative.")
        # Get the active campaigns by considering only the bid values which are strictly greater than 0.
        active_campaigns = np.where(bid_vector > 0.0)[0].tolist()
        # Check whether the active campaigns satisfy the conflict graph constraints.
        if not is_feasible_campaign_set(active_campaigns, self.conflict_graph):
            raise ValueError(
                "Campaign set is infeasible: two or more bid-on campaigns "
                "are non-compatible."
            )
        
        # Return the active campaigns found and the bid vector. 
        return active_campaigns, bid_vector

    def _is_in_bid_set(self, bid: float) -> bool:
        """Return True if the bid belongs to the discrete bid set within tolerance."""
        return bool(np.any(np.abs(self.bid_values - bid) <= BID_SET_TOLERANCE))