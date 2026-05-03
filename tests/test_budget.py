"""Tests for the Requirement 1 budget-aware UCB learner."""

import pytest

from ola.algorithms.budgeted_ucb import BudgetedUCB
from experiments.run_req1 import (
    compute_req1_benchmarks,
    expected_cost_for_uniform_competing_bid,
    expected_reward_for_uniform_competing_bid,
)


def test_budgeted_ucb_returns_zero_when_budget_is_depleted() -> None:
    learner = BudgetedUCB(actions=[0.1, 0.2], horizon=10, initial_budget=0.0)

    assert learner.select_action() == pytest.approx(0.0)


def test_budgeted_ucb_does_not_select_initial_bid_above_budget() -> None:
    learner = BudgetedUCB(actions=[0.3, 0.7], horizon=10, initial_budget=0.5)

    assert learner.select_action() == pytest.approx(0.3)


def test_budgeted_ucb_updates_reward_cost_and_budget() -> None:
    learner = BudgetedUCB(actions=[0.1, 0.2], horizon=10, initial_budget=1.0)

    learner.update(action=0.2, reward=0.8, cost=0.2)

    assert learner.counts[1] == 1
    assert learner.total_rewards[1] == pytest.approx(0.8)
    assert learner.total_costs[1] == pytest.approx(0.2)
    assert learner.budget_left == pytest.approx(0.8)
    assert learner.t == 1


def test_budgeted_ucb_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        BudgetedUCB(actions=[], horizon=10, initial_budget=1.0)

    with pytest.raises(ValueError, match="Horizon must be positive"):
        BudgetedUCB(actions=[0.1], horizon=0, initial_budget=1.0)

    with pytest.raises(ValueError, match="Initial budget must be non-negative"):
        BudgetedUCB(actions=[0.1], horizon=10, initial_budget=-1.0)


def test_budgeted_ucb_rejects_unknown_action_update() -> None:
    learner = BudgetedUCB(actions=[0.1, 0.2], horizon=10, initial_budget=1.0)

    with pytest.raises(ValueError, match="Unknown action"):
        learner.update(action=0.9, reward=1.0, cost=0.0)


def test_req1_expected_reward_and_cost_uniform_competing_bid() -> None:
    reward = expected_reward_for_uniform_competing_bid(
        bid=0.5,
        value=1.0,
        min_competing_bid=0.0,
        max_competing_bid=1.0,
    )
    cost = expected_cost_for_uniform_competing_bid(
        bid=0.5,
        min_competing_bid=0.0,
        max_competing_bid=1.0,
    )

    assert reward == pytest.approx(0.25)
    assert cost == pytest.approx(0.25)


def test_req1_expected_reward_allows_negative_utility_when_bid_exceeds_value() -> None:
    reward = expected_reward_for_uniform_competing_bid(
        bid=1.0,
        value=0.8,
        min_competing_bid=0.0,
        max_competing_bid=1.0,
    )

    assert reward == pytest.approx(-0.2)


def test_req1_budgeted_benchmark_uses_budget_feasibility() -> None:
    ucb1_benchmark, budgeted_benchmark = compute_req1_benchmarks(
        bids=[0.0, 0.5, 1.0],
        value=1.0,
        initial_budget=1.0,
        horizon=10,
        min_competing_bid=0.0,
        max_competing_bid=1.0,
    )

    assert ucb1_benchmark == pytest.approx(0.25)
    assert budgeted_benchmark == pytest.approx(0.0)