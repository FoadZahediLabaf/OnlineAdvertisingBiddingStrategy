"""Exact piecewise-static hindsight oracle under one global budget."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def solve_piecewise_static_oracle(
    market_matrix,
    change_points,
    bid_set,
    values,
    budget,
):
    """Choose one bid per regime and campaign with a shared global budget.

    The binary multiple-choice program has one group for every
    ``(regime, campaign)`` pair and one choice for every bid-grid value.
    """
    market = np.asarray(market_matrix, dtype=float)
    bids = np.asarray(bid_set, dtype=float)
    campaign_values = np.asarray(values, dtype=float)
    horizon, campaigns = market.shape
    boundaries = np.array([0, *change_points, horizon], dtype=int)
    windows = list(zip(boundaries[:-1], boundaries[1:]))

    groups = len(windows) * campaigns
    choices = bids.size
    variable_count = groups * choices
    utilities = np.zeros(variable_count)
    costs = np.zeros(variable_count)

    for window_index, (start, end) in enumerate(windows):
        observations = market[start:end]
        for campaign in range(campaigns):
            group = window_index * campaigns + campaign
            offset = group * choices
            wins = bids[:, None] >= observations[:, campaign][None, :]
            utilities[offset : offset + choices] = (
                (campaign_values[campaign] - bids)[:, None] * wins
            ).sum(axis=1)
            costs[offset : offset + choices] = (bids[:, None] * wins).sum(axis=1)

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
    selected_bids = bids[selected].reshape(len(windows), campaigns)
    utility_over_time = np.zeros(horizon)
    cost_over_time = np.zeros(horizon)
    actions_over_time = np.zeros((horizon, campaigns))
    for window_index, (start, end) in enumerate(windows):
        action = selected_bids[window_index]
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
        "window_actions": selected_bids,
        "boundaries": boundaries,
    }
