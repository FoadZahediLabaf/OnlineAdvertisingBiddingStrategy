"""Exact piecewise-static hindsight oracle under one global budget."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


def _campaign_set_is_feasible(campaign_set: tuple[int, ...], conflict_edges) -> bool:
    selected = set(campaign_set)
    return all(first not in selected or second not in selected for first, second in conflict_edges)


def _normalise_allowed_campaign_sets(campaigns, allowed_campaign_sets, conflict_edges):
    if allowed_campaign_sets is None:
        if not conflict_edges:
            return [tuple(range(campaigns))]
        feasible_sets = []
        for first in range(campaigns):
            feasible_sets.append((first,))
        for first in range(campaigns):
            for second in range(first + 1, campaigns):
                candidate = (first, second)
                if _campaign_set_is_feasible(candidate, conflict_edges):
                    feasible_sets.append(candidate)
        return feasible_sets

    normalised = []
    for campaign_set in allowed_campaign_sets:
        candidate = tuple(sorted(int(campaign) for campaign in campaign_set))
        if not candidate:
            continue
        if any(campaign < 0 or campaign >= campaigns for campaign in candidate):
            raise ValueError(f"Invalid campaign set {campaign_set}.")
        if len(set(candidate)) != len(candidate):
            raise ValueError(f"Campaign set {campaign_set} contains duplicates.")
        if not _campaign_set_is_feasible(candidate, conflict_edges):
            raise ValueError(f"Campaign set {campaign_set} violates conflict_edges.")
        normalised.append(candidate)

    if not normalised:
        raise ValueError("allowed_campaign_sets must contain at least one non-empty set.")
    return normalised


def _build_feasible_actions(campaigns, bids, allowed_campaign_sets):
    actions = []
    for campaign_set in allowed_campaign_sets:
        for bid_tuple in product(bids, repeat=len(campaign_set)):
            action = np.zeros(campaigns, dtype=float)
            for campaign, bid in zip(campaign_set, bid_tuple):
                action[campaign] = bid
            actions.append(action)
    return np.unique(np.asarray(actions, dtype=float), axis=0)


def _action_utility_and_cost(candidate_actions, market, campaign_values):
    wins = candidate_actions[:, None, :] >= market[None, :, :]
    utility = ((campaign_values - candidate_actions)[:, None, :] * wins).sum(axis=2)
    cost = (candidate_actions[:, None, :] * wins).sum(axis=2)
    return utility, cost


def solve_fixed_mixture_opt_a(
    market_matrix,
    bid_set,
    values,
    budget,
    allowed_campaign_sets=None,
    conflict_edges=None,
):
    """Best fixed randomized feasible action in hindsight.

    This is the adversarial clairvoyant comparator often denoted OPT-A. After
    observing the full reward/cost sequence, it chooses one fixed distribution
    over feasible actions whose average cost is at most rho = B / T.
    """
    market = np.asarray(market_matrix, dtype=float)
    bids = np.asarray(bid_set, dtype=float)
    campaign_values = np.asarray(values, dtype=float)
    horizon, campaigns = market.shape
    conflict_edges = [tuple(edge) for edge in (conflict_edges or [])]
    allowed_campaign_sets = _normalise_allowed_campaign_sets(
        campaigns,
        allowed_campaign_sets,
        conflict_edges,
    )
    candidate_actions = _build_feasible_actions(campaigns, bids, allowed_campaign_sets)
    utilities, costs = _action_utility_and_cost(candidate_actions, market, campaign_values)
    average_utility = utilities.mean(axis=1)
    average_cost = costs.mean(axis=1)
    rho = float(budget) / float(horizon)

    result = linprog(
        c=-average_utility,
        A_ub=average_cost[None, :],
        b_ub=np.array([rho]),
        A_eq=np.ones((1, candidate_actions.shape[0])),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * candidate_actions.shape[0],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"OPT-A LP failed: {result.message}")

    probabilities = np.maximum(result.x, 0.0)
    probabilities /= probabilities.sum()
    utility_over_time = probabilities @ utilities
    cost_over_time = probabilities @ costs
    return {
        "utility": utility_over_time,
        "cost": cost_over_time,
        "cumulative_utility": np.cumsum(utility_over_time),
        "cumulative_cost": np.cumsum(cost_over_time),
        "action_probabilities": probabilities,
        "candidate_actions": candidate_actions,
        "average_utility": average_utility,
        "average_cost": average_cost,
        "rho": rho,
        "opt_a_value_per_round": float(-result.fun),
    }


def solve_piecewise_static_oracle(
    market_matrix,
    change_points,
    bid_set,
    values,
    budget,
    allowed_campaign_sets=None,
    conflict_edges=None,
):
    """Choose one feasible super-arm action per regime under a global budget.

    The binary multiple-choice program has one group for every regime and one
    choice for every feasible super-arm/bid combination.
    """
    market = np.asarray(market_matrix, dtype=float)
    bids = np.asarray(bid_set, dtype=float)
    campaign_values = np.asarray(values, dtype=float)
    horizon, campaigns = market.shape
    conflict_edges = [tuple(edge) for edge in (conflict_edges or [])]
    allowed_campaign_sets = _normalise_allowed_campaign_sets(
        campaigns,
        allowed_campaign_sets,
        conflict_edges,
    )
    candidate_actions = _build_feasible_actions(campaigns, bids, allowed_campaign_sets)
    boundaries = np.array([0, *change_points, horizon], dtype=int)
    windows = list(zip(boundaries[:-1], boundaries[1:]))

    groups = len(windows)
    choices = candidate_actions.shape[0]
    variable_count = groups * choices
    utilities = np.zeros(variable_count)
    costs = np.zeros(variable_count)

    for window_index, (start, end) in enumerate(windows):
        observations = market[start:end]
        offset = window_index * choices
        action_utilities, action_costs = _action_utility_and_cost(
            candidate_actions,
            observations,
            campaign_values,
        )
        utilities[offset : offset + choices] = action_utilities.sum(axis=1)
        costs[offset : offset + choices] = action_costs.sum(axis=1)

    variable_indices = np.arange(variable_count)
    group_rows = np.repeat(np.arange(groups), choices)
    rows = np.concatenate([group_rows, np.full(variable_count, groups)])
    columns = np.concatenate([variable_indices, variable_indices])
    data = np.concatenate([np.ones(variable_count), costs])
    matrix = coo_matrix((data, (rows, columns)), shape=(groups + 1, variable_count)).tocsr()
    lower = np.concatenate([np.ones(groups), [-np.inf]])
    upper = np.concatenate([np.ones(groups), [float(budget)]])

    result = milp(
        c=-utilities,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"Oracle MILP failed: {result.message}")

    selected = np.rint(result.x).astype(int).reshape(groups, choices).argmax(axis=1)
    selected_actions = candidate_actions[selected]
    utility_over_time = np.zeros(horizon)
    cost_over_time = np.zeros(horizon)
    actions_over_time = np.zeros((horizon, campaigns))
    for window_index, (start, end) in enumerate(windows):
        action = selected_actions[window_index]
        observations = market[start:end]
        wins = action[None, :] >= observations
        utility_over_time[start:end] = ((campaign_values - action)[None, :] * wins).sum(axis=1)
        cost_over_time[start:end] = (action[None, :] * wins).sum(axis=1)
        actions_over_time[start:end] = action

    if cost_over_time.sum() > budget + 1e-6:
        raise RuntimeError("Oracle solution violates the global budget.")
    return {
        "utility": utility_over_time,
        "cost": cost_over_time,
        "cumulative_utility": np.cumsum(utility_over_time),
        "cumulative_cost": np.cumsum(cost_over_time),
        "actions": actions_over_time,
        "window_actions": selected_actions,
        "boundaries": boundaries,
    }
