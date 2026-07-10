"""
Tests for the Requirement 2 code (Combinatorial-UCB, multi-campaign env, LP benchmark).

Owner:
    Luigi

These cover the invariants the Req 2 write-up relies on:
  - the learner never selects an infeasible or unaffordable super-arm,
  - the environment rejects infeasible / over-budget / off-grid actions,
  - the clairvoyant LP benchmark dominates the best single feasible super-arm.

Run with pytest, or standalone:  python tests/test_req2.py
"""

from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_req2 import build_expected_reward_table, compute_req2_benchmark
from ola.algorithms.combinatorial_ucb import CombinatorialUCB
from ola.environments.multi_campaign import MultiCampaignEnvironment
from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


# A small 4-campaign path 0-1-2-3 instance, cheap enough to run to the horizon.
CAMPAIGNS = 4
VALUES = [1.0, 1.0, 1.0, 1.0]
BIDS = [0.0, 0.25, 0.5, 0.75, 1.0]
EDGES = [[0, 1], [1, 2], [2, 3]]
BUDGET = 120.0
HORIZON = 200
MEAN = [0.5, 0.5, 0.5, 0.5]
STD = 0.1
CORR = 0.5


def _make_env(seed: int = 0) -> MultiCampaignEnvironment:
    return MultiCampaignEnvironment(
        number_of_campaigns=CAMPAIGNS, campaign_values=VALUES, bid_values=BIDS,
        budget=BUDGET, horizon=HORIZON, conflict_edges=EDGES,
        mean_competing_bids=MEAN, std_competing_bids=STD,
        correlation_on_edge=CORR, random_seed=seed,
    )


def _make_agent(seed: int = 0, c: float = 0.5) -> CombinatorialUCB:
    return CombinatorialUCB(
        number_of_campaigns=CAMPAIGNS, campaign_values=VALUES, bid_values=BIDS,
        budget=BUDGET, horizon=HORIZON, conflict_edges=EDGES,
        random_seed=seed, exploration_coefficient=c,
    )


def test_selected_superarms_are_feasible_affordable_and_within_budget() -> None:
    """Every action the learner emits must be a conflict-free, non-duplicated,
    affordable super-arm, and the full run must never overspend the budget."""
    graph = build_conflict_graph(CAMPAIGNS, [tuple(e) for e in EDGES])
    env, agent = _make_env(1), _make_agent(1)
    reset_info = env.reset()
    done = bool(reset_info["done"])
    total_cost = 0.0
    while not done:
        budget_before = agent.budget_left
        action = agent.select_action()
        campaigns = action["active_campaigns"]
        bids = action["bids"]

        assert len(campaigns) == len(bids)
        assert len(set(campaigns)) == len(campaigns)               # no duplicates
        assert is_feasible_campaign_set(campaigns, graph)          # conflict-free
        assert all(b in BIDS for b in bids)                        # on the grid
        assert sum(bids) <= budget_before + 1e-9                   # affordable now

        step_result = env.step(action)                            # env would raise otherwise
        agent.update(action, step_result)
        total_cost += float(step_result["cost"])
        done = bool(step_result["done"])

    assert total_cost <= BUDGET + 1e-9                             # never overspent


def test_warmup_pulls_every_arm_once_before_optimising() -> None:
    """The first N*|B| selections are singletons covering each (campaign, bid) arm."""
    env, agent = _make_env(2), _make_agent(2)
    env.reset()
    seen = set()
    for _ in range(CAMPAIGNS * len(BIDS)):
        action = agent.select_action()
        assert len(action["active_campaigns"]) == 1                # warm-up is singletons
        campaign = action["active_campaigns"][0]
        bid = action["bids"][0]
        seen.add((campaign, round(bid, 6)))
        step_result = env.step(action)
        agent.update(action, step_result)
    assert len(seen) == CAMPAIGNS * len(BIDS)                      # all arms covered once


def test_env_rejects_infeasible_over_budget_and_offgrid_actions() -> None:
    env = _make_env(3)
    env.reset()
    # two conflicting campaigns (edge 0-1) in one round
    _assert_raises(lambda: env.step({"active_campaigns": [0, 1], "bids": [0.25, 0.25]}))
    # a bid that is not on the discrete grid
    _assert_raises(lambda: env.step({"active_campaigns": [0], "bids": [0.3]}))
    # out-of-range campaign index
    _assert_raises(lambda: env.step({"active_campaigns": [99], "bids": [0.25]}))

    tiny = MultiCampaignEnvironment(
        number_of_campaigns=CAMPAIGNS, campaign_values=VALUES, bid_values=BIDS,
        budget=0.25, horizon=HORIZON, conflict_edges=EDGES,
        mean_competing_bids=MEAN, std_competing_bids=STD,
        correlation_on_edge=CORR, random_seed=0,
    )
    tiny.reset()
    # requested spend (0.5 + 0.5) exceeds the 0.25 budget -> must raise
    _assert_raises(lambda: tiny.step({"active_campaigns": [0, 2], "bids": [0.5, 0.5]}))


def test_benchmark_dominates_best_pure_superarm_and_is_nonnegative() -> None:
    """OPT^S (mixed LP) must be >= the best single feasible super-arm and >= 0."""
    benchmark = compute_req2_benchmark(
        campaign_values=VALUES, bid_values=BIDS, initial_budget=BUDGET,
        horizon=HORIZON, conflict_edges=EDGES, mean_competing_bids=MEAN,
        std_competing_bids=STD, correlation_on_edge=CORR,
    )
    assert benchmark >= 0.0

    bid_grid, reward_table = build_expected_reward_table(VALUES, BIDS, MEAN, STD)
    rho = BUDGET / HORIZON
    from scipy.stats import truncnorm
    cost_table = np.zeros_like(reward_table)
    for i in range(CAMPAIGNS):
        a, b = (0 - MEAN[i]) / STD, (1 - MEAN[i]) / STD
        cost_table[i] = bid_grid * truncnorm.cdf(bid_grid, a, b, loc=MEAN[i], scale=STD)

    graph = build_conflict_graph(CAMPAIGNS, [tuple(e) for e in EDGES])
    best_pure = 0.0
    for size in range(1, CAMPAIGNS + 1):
        for combo in combinations(range(CAMPAIGNS), size):
            if not is_feasible_campaign_set(combo, graph):
                continue
            for bid_combo in product(range(len(BIDS)), repeat=len(combo)):
                reward = sum(reward_table[i, bid_combo[p]] for p, i in enumerate(combo))
                cost = sum(cost_table[i, bid_combo[p]] for p, i in enumerate(combo))
                if cost <= rho + 1e-9:
                    best_pure = max(best_pure, reward)
    assert benchmark >= best_pure - 1e-6


def _assert_raises(thunk) -> None:
    try:
        thunk()
    except ValueError:
        return
    raise AssertionError("expected a ValueError but none was raised")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("All Req 2 tests passed.")
