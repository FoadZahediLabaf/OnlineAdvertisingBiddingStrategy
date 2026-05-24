"""
Experiment runner for Requirement 3.

Owner:
    Mengyao

Support:
    Simone may edit this file when actively helping Mengyao.

Purpose:
    Run best-of-both-worlds experiments using the primal-dual bidding method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.algorithms.primal_dual import PrimalDualBidder, generate_feasible_actions
from ola.environments.adversarial import AdversarialMultiCampaignEnvironment
from ola.utils.logging import save_results


CONFIG_PATH = PROJECT_ROOT / "configs" / "req3.yaml"


def load_req3_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load Requirement 3 parameters from the YAML config file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Missing Req 3 config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid Req 3 config file: {config_path}")

    return config


def compute_action_outcomes(
    actions: np.ndarray,
    highest_competing_bid_sequence: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reward and cost for every action at every round."""

    number_of_actions = actions.shape[0]
    horizon = highest_competing_bid_sequence.shape[0]
    rewards_per_round = np.zeros((number_of_actions, horizon))
    costs_per_round = np.zeros((number_of_actions, horizon))

    for round_index, highest_competing_bids in enumerate(highest_competing_bid_sequence):
        wins = actions >= highest_competing_bids
        rewards_per_round[:, round_index] = np.sum((values - actions) * wins, axis=1)
        costs_per_round[:, round_index] = np.sum(actions * wins, axis=1)

    return rewards_per_round, costs_per_round


def compute_piecewise_static_oracle(
    actions: np.ndarray,
    highest_competing_bid_sequence: np.ndarray,
    values: np.ndarray,
    initial_budget: float,
    regime_length: int,
) -> Dict[str, np.ndarray]:
    """Compute a piecewise-static oracle under one global budget constraint."""

    rewards_per_round, costs_per_round = compute_action_outcomes(
        actions,
        highest_competing_bid_sequence,
        values,
    )
    horizon = highest_competing_bid_sequence.shape[0]
    block_ranges = [
        (start_round, min(start_round + regime_length, horizon))
        for start_round in range(0, horizon, regime_length)
    ]
    number_of_blocks = len(block_ranges)
    number_of_actions = actions.shape[0]

    block_rewards = np.zeros((number_of_blocks, number_of_actions))
    block_costs = np.zeros((number_of_blocks, number_of_actions))
    for block_index, (start_round, end_round) in enumerate(block_ranges):
        block_rewards[block_index] = rewards_per_round[:, start_round:end_round].sum(
            axis=1
        )
        block_costs[block_index] = costs_per_round[:, start_round:end_round].sum(axis=1)

    positive_bids = np.unique(actions[actions > 0.0])
    if positive_bids.size > 1:
        cost_unit = float(np.min(np.diff(np.sort(positive_bids))))
    elif positive_bids.size == 1:
        cost_unit = float(positive_bids[0])
    else:
        cost_unit = 1.0

    budget_units = int(np.floor(initial_budget / cost_unit + 1e-9))
    block_cost_units = np.rint(block_costs / cost_unit).astype(int)

    dp = np.full((number_of_blocks + 1, budget_units + 1), -np.inf)
    previous_budget = np.full((number_of_blocks + 1, budget_units + 1), -1, dtype=int)
    previous_action = np.full((number_of_blocks + 1, budget_units + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for block_index in range(number_of_blocks):
        for action_index in range(number_of_actions):
            action_cost = block_cost_units[block_index, action_index]
            action_reward = block_rewards[block_index, action_index]
            if action_cost > budget_units:
                continue

            candidate = dp[block_index, : budget_units + 1 - action_cost] + action_reward
            target = dp[block_index + 1, action_cost:]
            improved = candidate > target
            if np.any(improved):
                improved_budgets = np.where(improved)[0] + action_cost
                dp[block_index + 1, improved_budgets] = candidate[improved]
                previous_budget[block_index + 1, improved_budgets] = (
                    improved_budgets - action_cost
                )
                previous_action[block_index + 1, improved_budgets] = action_index

    best_budget = int(np.nanargmax(dp[number_of_blocks]))
    block_oracle_indices = [0] * number_of_blocks
    current_budget = best_budget
    for block_index in range(number_of_blocks, 0, -1):
        action_index = previous_action[block_index, current_budget]
        if action_index < 0:
            action_index = 0
        block_oracle_indices[block_index - 1] = int(action_index)
        current_budget = max(0, previous_budget[block_index, current_budget])

    oracle_rewards = np.zeros(horizon)
    oracle_costs = np.zeros(horizon)
    for block_index, (start_round, end_round) in enumerate(block_ranges):
        action_index = block_oracle_indices[block_index]
        oracle_rewards[start_round:end_round] = rewards_per_round[
            action_index,
            start_round:end_round,
        ]
        oracle_costs[start_round:end_round] = costs_per_round[
            action_index,
            start_round:end_round,
        ]

    return {
        "oracle_rewards": oracle_rewards,
        "oracle_costs": oracle_costs,
        "cumulative_oracle_reward": np.cumsum(oracle_rewards),
        "cumulative_oracle_cost": np.cumsum(oracle_costs),
    }


def run_primal_dual_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run one reproducible Req 3 experiment and return standard result rows."""

    experiment_config = config["experiment"]
    campaign_config = config["campaigns"]
    bids_config = config["bids"]
    environment_config = config["environment"]
    algorithm_config = config["algorithm"]

    horizon = int(experiment_config["horizon"])
    random_seed = int(experiment_config["random_seed"])
    number_of_campaigns = int(campaign_config["number_of_campaigns"])
    values = np.array(campaign_config["values"], dtype=float)
    initial_budget = float(campaign_config["initial_budget"])
    bid_set = np.array(bids_config["values"], dtype=float)
    conflict_edges = [tuple(edge) for edge in campaign_config["conflict_edges"]]
    regime_length = int(environment_config["regime_length"])

    actions = generate_feasible_actions(
        bid_set=bid_set,
        number_of_campaigns=number_of_campaigns,
        conflict_edges=conflict_edges,
    )
    env = AdversarialMultiCampaignEnvironment(
        values=values,
        initial_budget=initial_budget,
        horizon=horizon,
        regime_length=regime_length,
        sigma_scale=float(environment_config["sigma_scale"]),
        random_seed=random_seed,
    )
    learner = PrimalDualBidder(
        actions=actions,
        values=values,
        initial_budget=initial_budget,
        horizon=horizon,
        learning_rate=algorithm_config["learning_rate"],
        dual_learning_rate=algorithm_config["dual_learning_rate"],
        random_seed=random_seed + 1,
    )
    oracle = compute_piecewise_static_oracle(
        actions=actions,
        highest_competing_bid_sequence=env.highest_competing_bids,
        values=values,
        initial_budget=initial_budget,
        regime_length=regime_length,
    )

    env.reset()
    rows = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0

    for _ in range(horizon):
        action_index, action = learner.select_action()
        outcome = env.step(action)

        reward = float(outcome["reward"])
        cost = float(outcome["cost"])
        cumulative_reward += reward
        cumulative_cost += cost

        learner.update(
            action=action,
            reward=reward,
            cost=cost,
            feedback=outcome["info"],
        )

        round_index = int(outcome["round"]) - 1
        benchmark_reward = float(oracle["oracle_rewards"][round_index])
        instant_regret = benchmark_reward - reward
        cumulative_regret = float(
            oracle["cumulative_oracle_reward"][round_index] - cumulative_reward
        )

        rows.append(
            {
                "algorithm": "primal_dual",
                "round": outcome["round"],
                "action": int(action_index),
                "bids": json.dumps(outcome["bids"]),
                "wins": json.dumps(outcome["wins"]),
                "reward": reward,
                "cost": cost,
                "budget_left": outcome["budget_left"],
                "cumulative_reward": cumulative_reward,
                "cumulative_cost": cumulative_cost,
                "benchmark_reward": benchmark_reward,
                "instant_regret": instant_regret,
                "cumulative_regret": cumulative_regret,
                "seed": random_seed,
                "environment_name": environment_config["sequence_type"],
                "config_name": experiment_config["name"],
                "highest_competing_bid": json.dumps(
                    outcome["info"]["highest_competing_bid"]
                ),
                "done": outcome["done"],
            }
        )

        if outcome["done"]:
            break

    return rows


def save_req3_plots(rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    """Generate standard Req 3 plots from the saved result rows."""

    plots_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)

    plot_specs = [
        (
            "cumulative_reward",
            "Cumulative reward",
            "Requirement 3 - Cumulative reward",
            "req3_cumulative_reward.png",
        ),
        (
            "cumulative_regret",
            "Cumulative regret",
            "Requirement 3 - Cumulative regret",
            "req3_cumulative_regret.png",
        ),
        (
            "budget_left",
            "Budget left",
            "Requirement 3 - Budget left",
            "req3_budget_left.png",
        ),
        (
            "cumulative_cost",
            "Cumulative cost",
            "Requirement 3 - Cumulative cost",
            "req3_cumulative_cost.png",
        ),
    ]

    for metric, ylabel, title, filename in plot_specs:
        plt.figure(figsize=(10, 6))
        plt.plot(dataframe["round"], dataframe[metric], label="primal_dual")
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_path = plots_dir / filename
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved plot to {output_path}")


def main() -> None:
    config = load_req3_config()
    output_config = config["outputs"]

    rows = run_primal_dual_experiment(config)

    results_path = PROJECT_ROOT / output_config["primal_dual_results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(rows, str(results_path))
    print(f"Saved {len(rows)} rows to {results_path}")

    plots_dir = PROJECT_ROOT / output_config["plots_dir"]
    save_req3_plots(rows, plots_dir)

    print("Req 3 experiment completed.")


if __name__ == "__main__":
    main()
