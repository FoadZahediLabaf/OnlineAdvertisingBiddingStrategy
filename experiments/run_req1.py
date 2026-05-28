"""
Experiment runner for Requirement 1.

Owner:
    Foad

Purpose:
    Run single-campaign experiments for UCB1 and Budgeted UCB.

Do not:
    - Add Requirement 2, Requirement 3, or Requirement 4 logic here.
    - Change the standard CSV result format.
    - Save final results only as screenshots.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.algorithms.budgeted_ucb import BudgetedUCB
from ola.algorithms.ucb1 import UCB1
from ola.environments.single_campaign import SingleCampaignEnvironment
from ola.utils.logging import save_results
from ola.utils.plotting import generate_standard_req1_plots, load_result_csvs



CONFIG_PATH = PROJECT_ROOT / "configs" / "req1.yaml"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def expected_reward_for_uniform_competing_bid(
    bid: float,
    value: float,
    min_competing_bid: float,
    max_competing_bid: float,
) -> float:
    """Compute E[(value - bid) 1[bid >= m]] for m uniformly distributed."""
    if max_competing_bid < min_competing_bid:
        raise ValueError("max_competing_bid must be >= min_competing_bid")

    if bid < min_competing_bid:
        win_probability = 0.0
    elif bid >= max_competing_bid:
        win_probability = 1.0
    else:
        support_width = max_competing_bid - min_competing_bid
        if support_width == 0.0:
            win_probability = 1.0
        else:
            win_probability = (bid - min_competing_bid) / support_width

    return (value - bid) * win_probability


def expected_cost_for_uniform_competing_bid(
    bid: float,
    min_competing_bid: float,
    max_competing_bid: float,
) -> float:
    """Compute E[bid 1[bid >= m]] for m uniformly distributed."""
    if max_competing_bid < min_competing_bid:
        raise ValueError("max_competing_bid must be >= min_competing_bid")

    if bid < min_competing_bid:
        win_probability = 0.0
    elif bid >= max_competing_bid:
        win_probability = 1.0
    else:
        support_width = max_competing_bid - min_competing_bid
        if support_width == 0.0:
            win_probability = 1.0
        else:
            win_probability = (bid - min_competing_bid) / support_width

    return bid * win_probability


def compute_budgeted_lp_benchmark(
    expected_rewards: List[float],
    expected_costs: List[float],
    rho: float,
) -> float:
    """Solve the single-campaign budgeted benchmark LP over bid distributions.

    The LP is:
        maximize    sum_b gamma_b expected_reward(b)
        subject to  sum_b gamma_b expected_cost(b) <= rho
                    sum_b gamma_b = 1
                    gamma_b >= 0

    With one budget constraint, an optimal solution has support on at most two bids.
    We therefore enumerate all feasible single bids and all two-bid mixtures.
    """
    if rho < 0.0:
        raise ValueError("rho must be non-negative")
    if len(expected_rewards) != len(expected_costs):
        raise ValueError("expected_rewards and expected_costs must have same length")
    if not expected_rewards:
        raise ValueError("At least one bid is required")

    best_reward = -float("inf")
    number_of_bids = len(expected_rewards)

    for reward, cost in zip(expected_rewards, expected_costs):
        if cost <= rho:
            best_reward = max(best_reward, reward)

    for i in range(number_of_bids):
        for j in range(i + 1, number_of_bids):
            cost_i = expected_costs[i]
            cost_j = expected_costs[j]

            if cost_i == cost_j:
                continue

            probability_i = (rho - cost_j) / (cost_i - cost_j)
            probability_j = 1.0 - probability_i

            if 0.0 <= probability_i <= 1.0 and 0.0 <= probability_j <= 1.0:
                mixed_reward = (
                    probability_i * expected_rewards[i]
                    + probability_j * expected_rewards[j]
                )
                best_reward = max(best_reward, mixed_reward)

    if best_reward == -float("inf"):
        return 0.0

    return float(best_reward)


def compute_req1_benchmarks(
    bids: np.ndarray,
    value: float,
    initial_budget: float,
    horizon: int,
    min_competing_bid: float,
    max_competing_bid: float,
) -> Tuple[float, float]:
    """Compute non-budgeted and budget-feasible per-round benchmarks for Req 1."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    expected_rewards = [
        expected_reward_for_uniform_competing_bid(
            bid=float(bid),
            value=value,
            min_competing_bid=min_competing_bid,
            max_competing_bid=max_competing_bid,
        )
        for bid in bids
    ]
    ucb1_benchmark = max(expected_rewards)

    expected_costs = [
        expected_cost_for_uniform_competing_bid(
            bid=float(bid),
            min_competing_bid=min_competing_bid,
            max_competing_bid=max_competing_bid,
        )
        for bid in bids
    ]
    rho = initial_budget / horizon
    budgeted_benchmark = compute_budgeted_lp_benchmark(
        expected_rewards=expected_rewards,
        expected_costs=expected_costs,
        rho=rho,
    )

    return float(ucb1_benchmark), float(budgeted_benchmark)


def load_req1_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load Requirement 1 parameters from the YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Req 1 config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid Req 1 config file: {config_path}")

    return config


def run_algorithm(
    algorithm_name: str,
    learner: Any,
    value: float,
    initial_budget: float,
    horizon: int,
    random_seed: int,
    min_competing_bid: float,
    max_competing_bid: float,
    benchmark_reward_per_round: float,
) -> List[Dict[str, Any]]:
    """Run one learner in the single-campaign environment."""
    env = SingleCampaignEnvironment(
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        random_seed=random_seed,
    )
    env.reset()

    rows: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    cumulative_regret = 0.0

    for _ in range(horizon):
        action = learner.select_action()
        outcome = env.step(action)

        reward = float(outcome["reward"])
        cost = float(outcome["cost"])

        cumulative_reward += reward
        cumulative_cost += cost

        # Update the learner using the originally selected action, not the
        # environment's effective/clipped bid. The learner's action set is the
        # discrete bid set, while outcome["action"] may be clipped by the
        # remaining budget and therefore may not belong to that discrete set.
        if algorithm_name == "budgeted_ucb":
            learner.update(action=action, reward=reward, cost=cost)
        else:
            learner.update(action=action, reward=reward)

        benchmark_reward = float(benchmark_reward_per_round)
        instant_regret = benchmark_reward - reward
        cumulative_regret += instant_regret

        rows.append(
            {
                "algorithm": algorithm_name,
                "round": outcome["round"],
                "action": action,
                "bids": action,
                "wins": outcome["wins"],
                "reward": reward,
                "cost": cost,
                "budget_left": outcome["budget_left"],
                "cumulative_reward": cumulative_reward,
                "cumulative_cost": cumulative_cost,
                "benchmark_reward": benchmark_reward,
                "instant_regret": instant_regret,
                "cumulative_regret": cumulative_regret,
                "highest_competing_bid": outcome["info"]["highest_competing_bid"],
                "done": outcome["done"],
                "effective_bid": outcome["action"],
            }
        )

        if outcome["done"]:
            break

    return rows


def save_algorithm_results(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save one algorithm's results using the shared CSV format."""
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(rows, str(output_path))
    print(f"Saved {len(rows)} rows to {output_path}")
    return output_path


def main() -> None:
    config = load_req1_config()

    experiment_config = config["experiment"]
    campaign_config = config["campaign"]
    bids_config = config["bids"]
    environment_config = config["environment"]
    outputs_config = config["outputs"]

    horizon = int(experiment_config["horizon"])
    value = float(campaign_config["value"])
    initial_budget = float(campaign_config["initial_budget"])
    random_seed = int(experiment_config["random_seed"])
    bids = np.array(bids_config["values"], dtype=float)
    min_competing_bid = float(environment_config["min_competing_bid"])
    max_competing_bid = float(environment_config["max_competing_bid"])

    ucb1_benchmark, budgeted_ucb_benchmark = compute_req1_benchmarks(
        bids=bids,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
    )

    ucb1 = UCB1(actions=bids, horizon=horizon)
    ucb1_rows = run_algorithm(
        algorithm_name="ucb1",
        learner=ucb1,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        benchmark_reward_per_round=ucb1_benchmark,
    )
    ucb1_path = save_algorithm_results(
        ucb1_rows,
        Path(outputs_config["ucb1_results_file"]),
    )

    budgeted_ucb = BudgetedUCB(
        actions=bids,
        horizon=horizon,
        initial_budget=initial_budget,
    )
    budgeted_ucb_rows = run_algorithm(
        algorithm_name="budgeted_ucb",
        learner=budgeted_ucb,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        benchmark_reward_per_round=budgeted_ucb_benchmark,
    )
    budgeted_ucb_path = save_algorithm_results(
        budgeted_ucb_rows,
        Path(outputs_config["budgeted_ucb_results_file"]),
    )

    combined_results = load_result_csvs([ucb1_path, budgeted_ucb_path])
    generate_standard_req1_plots(combined_results, plots_dir=PLOTS_DIR)

    print("Req 1 experiments completed.")


if __name__ == "__main__":
    main()