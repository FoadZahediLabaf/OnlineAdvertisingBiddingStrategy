"""Tests for the Requirement 1 UCB1 learner."""

import pytest

from ola.algorithms.ucb1 import UCB1


def test_ucb1_plays_each_action_once_before_using_ucb() -> None:
    learner = UCB1(actions=[0.1, 0.2, 0.3], horizon=20)

    assert learner.select_action() == pytest.approx(0.1)
    learner.update(action=0.1, reward=0.0)

    assert learner.select_action() == pytest.approx(0.2)
    learner.update(action=0.2, reward=1.0)

    assert learner.select_action() == pytest.approx(0.3)
    learner.update(action=0.3, reward=0.5)


def test_ucb1_prefers_better_empirical_action_when_counts_are_equal() -> None:
    learner = UCB1(actions=[0.1, 0.2], horizon=20)

    learner.update(action=0.1, reward=0.0)
    learner.update(action=0.2, reward=1.0)

    assert learner.select_action() == pytest.approx(0.2)


def test_ucb1_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Action set must be non-empty"):
        UCB1(actions=[], horizon=10)

    with pytest.raises(ValueError, match="Horizon must be positive"):
        UCB1(actions=[0.1], horizon=0)


def test_ucb1_rejects_unknown_action_update() -> None:
    learner = UCB1(actions=[0.1, 0.2], horizon=10)

    with pytest.raises(ValueError, match="Unknown action"):
        learner.update(action=0.9, reward=1.0)