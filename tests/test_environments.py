"""Tests for project environments."""

import pytest

from ola.environments.base import STEP_OUTPUT_KEYS
from ola.environments.single_campaign import SingleCampaignEnvironment


def test_single_campaign_reset_returns_initial_state() -> None:
    env = SingleCampaignEnvironment(
        value=1.0,
        initial_budget=5.0,
        horizon=10,
        min_competing_bid=0.0,
        max_competing_bid=1.0,
        random_seed=123,
    )

    state = env.reset()

    assert state["round"] == 0
    assert state["budget_left"] == pytest.approx(5.0)
    assert state["done"] is False
    assert state["info"]["value"] == pytest.approx(1.0)


def test_single_campaign_step_has_standard_output_keys() -> None:
    env = SingleCampaignEnvironment(
        value=1.0,
        initial_budget=5.0,
        horizon=10,
        min_competing_bid=0.0,
        max_competing_bid=0.0,
        random_seed=123,
    )
    env.reset()

    outcome = env.step(0.5)

    assert set(STEP_OUTPUT_KEYS).issubset(outcome.keys())
    assert outcome["round"] == 1
    assert outcome["action"] == pytest.approx(0.5)
    assert outcome["bids"] == pytest.approx(0.5)
    assert outcome["wins"] == 1
    assert outcome["reward"] == pytest.approx(0.5)
    assert outcome["cost"] == pytest.approx(0.5)
    assert outcome["budget_left"] == pytest.approx(4.5)
    assert outcome["done"] is False
    assert outcome["info"]["highest_competing_bid"] == pytest.approx(0.0)


def test_single_campaign_rejects_bid_above_remaining_budget() -> None:
    # The environment intentionally does NOT clip an over-budget bid -- callers
    # (the learners / experiment loop) are responsible for never offering a bid
    # above env.budget_left. See the comment in experiments/run_req1.py.
    env = SingleCampaignEnvironment(
        value=1.0,
        initial_budget=0.3,
        horizon=10,
        min_competing_bid=0.0,
        max_competing_bid=0.0,
        random_seed=123,
    )
    env.reset()

    with pytest.raises(ValueError, match="Bid cannot exceed the remaining budget"):
        env.step(1.0)


def test_single_campaign_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Value must be non-negative"):
        SingleCampaignEnvironment(value=-1.0)

    with pytest.raises(ValueError, match="Initial budget must be non-negative"):
        SingleCampaignEnvironment(initial_budget=-1.0)

    with pytest.raises(ValueError, match="Horizon must be positive"):
        SingleCampaignEnvironment(horizon=0)

    with pytest.raises(ValueError, match="Minimum competing bid must be non-negative"):
        SingleCampaignEnvironment(min_competing_bid=-0.1)

    with pytest.raises(ValueError, match="Maximum competing bid"):
        SingleCampaignEnvironment(min_competing_bid=0.8, max_competing_bid=0.2)


def test_single_campaign_rejects_step_after_horizon() -> None:
    env = SingleCampaignEnvironment(
        value=1.0,
        initial_budget=5.0,
        horizon=1,
        min_competing_bid=0.0,
        max_competing_bid=0.0,
        random_seed=123,
    )
    env.reset()
    env.step(0.1)

    with pytest.raises(RuntimeError, match="horizon is reached"):
        env.step(0.1)


def test_single_campaign_rejects_negative_bid() -> None:
    env = SingleCampaignEnvironment(random_seed=123)
    env.reset()

    with pytest.raises(ValueError, match="Bid must be non-negative"):
        env.step(-0.1)