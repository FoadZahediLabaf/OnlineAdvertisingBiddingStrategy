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

from ola.algorithms.ucb_variants import (
    UCB1Tuned,
    ExpectedRewardScaledUCB,
    BudgetedUCB1Tuned,
    BudgetedExpectedRewardScaledUCB,
)
from ola.algorithms.budgeted_ucb import BudgetedUCB
from ola.algorithms.ucb1 import UCB1
from ola.environments.single_campaign import SingleCampaignEnvironment
from ola.utils.logging import save_results
from ola.utils.plotting import generate_standard_req1_plots, load_result_csvs

BUDGETED_ALGORITHM_NAMES = {
    "budgeted_ucb",
    "budgeted_ucb1_tuned",
    "budgeted_expected_reward_scaled_ucb",
}

CONFIG_PATH = PROJECT_ROOT / "configs" / "req1.yaml"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def expected_win_probability(
    bid: float,
    min_competing_bid: float,
    max_competing_bid: float,
    competing_bid_distribution: str = "uniform",
    n_competitors: int = 3,
) -> float:
    """Compute P(m <= bid) for the configured highest competing bid model."""
    if max_competing_bid < min_competing_bid:
        raise ValueError("max_competing_bid must be >= min_competing_bid")

    if bid < min_competing_bid:
        base_probability = 0.0
    elif bid >= max_competing_bid:
        base_probability = 1.0
    else:
        support_width = max_competing_bid - min_competing_bid
        if support_width == 0.0:
            base_probability = 1.0
        else:
            base_probability = (bid - min_competing_bid) / support_width

    if competing_bid_distribution == "uniform":
        return float(base_probability)
    if competing_bid_distribution == "max_uniform":
        return float(base_probability ** int(n_competitors))

    raise ValueError(f"Unknown competing_bid_distribution: {competing_bid_distribution}")


def expected_reward_for_uniform_competing_bid(
    bid: float,
    value: float,
    min_competing_bid: float,
    max_competing_bid: float,
    competing_bid_distribution: str = "uniform",
    n_competitors: int = 3,
) -> float:
    """Compute expected first-price reward for the configured bid model.

    The function name is kept for notebook compatibility. With
    competing_bid_distribution="uniform", this is the original uniform model.
    With competing_bid_distribution="max_uniform", the highest competing bid is
    the maximum of n_competitors independent uniform competitor bids.
    """
    win_probability = expected_win_probability(
        bid=bid,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    return float((value - bid) * win_probability)


def expected_cost_for_uniform_competing_bid(
    bid: float,
    min_competing_bid: float,
    max_competing_bid: float,
    competing_bid_distribution: str = "uniform",
    n_competitors: int = 3,
) -> float:
    """Compute expected first-price cost for the configured bid model.

    The function name is kept for notebook compatibility.
    """
    win_probability = expected_win_probability(
        bid=bid,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    return float(bid * win_probability)


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
    competing_bid_distribution: str = "uniform",
    n_competitors: int = 3,
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
            competing_bid_distribution=competing_bid_distribution,
            n_competitors=n_competitors,
        )
        for bid in bids
    ]
    ucb1_benchmark = max(expected_rewards)

    expected_costs = [
        expected_cost_for_uniform_competing_bid(
            bid=float(bid),
            min_competing_bid=min_competing_bid,
            max_competing_bid=max_competing_bid,
            competing_bid_distribution=competing_bid_distribution,
            n_competitors=n_competitors,
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


def compute_full_horizon_regret(
    total_reward: float,
    benchmark_reward_per_round: float,
    horizon: int,
) -> float:
    """Compute regret over the full horizon, including rounds after early stopping.

    Raw experiment rows contain only real auction interactions. If an algorithm
    stops before the horizon because the budget is depleted, the missing rounds
    must still count in the final regret comparison against the T-round
    benchmark.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    benchmark_total_reward = float(horizon) * float(benchmark_reward_per_round)
    return float(benchmark_total_reward - float(total_reward))


def add_full_horizon_regret_summary_columns(
    rows: List[Dict[str, Any]],
    benchmark_reward_per_round: float,
    horizon: int,
) -> List[Dict[str, Any]]:
    """Add final full-horizon regret metadata to the last real row only.

    This keeps the raw trajectory clean: no fake stopped-round rows are added.
    The notebook can read these final columns from the last row of each run.
    """
    if not rows:
        return rows

    total_reward = float(rows[-1]["cumulative_reward"])
    rounds_played = int(rows[-1]["round"])
    full_horizon_regret = compute_full_horizon_regret(
        total_reward=total_reward,
        benchmark_reward_per_round=benchmark_reward_per_round,
        horizon=horizon,
    )
    missing_rounds_after_stop = max(0, int(horizon) - rounds_played)

    rows[-1]["full_horizon_regret"] = full_horizon_regret
    rows[-1]["average_full_horizon_regret"] = full_horizon_regret / float(horizon)
    rows[-1]["missing_rounds_after_stop"] = missing_rounds_after_stop
    rows[-1]["stopped_early"] = bool(rounds_played < horizon)

    return rows


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
    competing_bid_distribution: str = "uniform",
    n_competitors: int = 3,
) -> List[Dict[str, Any]]:
    """Run one learner in the single-campaign environment."""
    env = SingleCampaignEnvironment(
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
        random_seed=random_seed,
    )
    initial_state = env.reset()
    if bool(initial_state["done"]):
        return []

    rows: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    cumulative_regret = 0.0

    for _ in range(horizon):
        action = learner.select_action()
        if float(action) > env.budget_left:
            break
        outcome = env.step(action)

        reward = float(outcome["reward"])
        cost = float(outcome["cost"])

        cumulative_reward += reward
        cumulative_cost += cost

        # Update the learner using the selected discrete bid. The environment
        # no longer clips bids, so this action is the same bid used to compute
        # reward and cost.
        if algorithm_name in BUDGETED_ALGORITHM_NAMES:
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
                "competing_bid_distribution": competing_bid_distribution,
                "n_competitors": n_competitors,
                "done": outcome["done"],
                "effective_bid": outcome["action"],
                "full_horizon_regret": np.nan,
                "average_full_horizon_regret": np.nan,
                "missing_rounds_after_stop": np.nan,
                "stopped_early": np.nan,
            }
        )

        if outcome["done"]:
            break

    return add_full_horizon_regret_summary_columns(
        rows=rows,
        benchmark_reward_per_round=benchmark_reward_per_round,
        horizon=horizon,
    )


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
    competing_bid_distribution = environment_config.get("competing_bid_distribution", "uniform")
    n_competitors = int(environment_config.get("n_competitors", 3))

    ucb1_benchmark, budgeted_ucb_benchmark = compute_req1_benchmarks(
        bids=bids,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
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
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    ucb1_path = save_algorithm_results(
        ucb1_rows,
        Path(outputs_config["ucb1_results_file"]),
    )

    learner_random_seed = random_seed + 1_000_003  # decorrelated stream, distinct from env's RNG

    budgeted_ucb = BudgetedUCB(
            actions=bids,
            horizon=horizon,
            initial_budget=initial_budget,
            random_seed=learner_random_seed,
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
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    budgeted_ucb_path = save_algorithm_results(
        budgeted_ucb_rows,
        Path(outputs_config["budgeted_ucb_results_file"]),
    )
    ucb1_tuned = UCB1Tuned(actions=bids, horizon=horizon)
    ucb1_tuned_rows = run_algorithm(
        algorithm_name="ucb1_tuned",
        learner=ucb1_tuned,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        benchmark_reward_per_round=ucb1_benchmark,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    ucb1_tuned_path = save_algorithm_results(
        ucb1_tuned_rows,
        Path(outputs_config["ucb1_tuned_results_file"]),
    )

    # ucb1_benchmark IS max(expected_reward(b) for b in bids) already -- no
    # need to recompute it separately for this variant.
    expected_reward_scaled_ucb = ExpectedRewardScaledUCB(
        actions=bids,
        horizon=horizon,
        max_expected_reward=ucb1_benchmark,
    )
    expected_reward_scaled_ucb_rows = run_algorithm(
        algorithm_name="expected_reward_scaled_ucb",
        learner=expected_reward_scaled_ucb,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        benchmark_reward_per_round=ucb1_benchmark,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    expected_reward_scaled_ucb_path = save_algorithm_results(
        expected_reward_scaled_ucb_rows,
        Path(outputs_config["expected_reward_scaled_ucb_results_file"]),
    )

    budgeted_ucb1_tuned_seed = random_seed + 2_000_003  # own decorrelated stream
    budgeted_ucb1_tuned = BudgetedUCB1Tuned(
        actions=bids,
        horizon=horizon,
        initial_budget=initial_budget,
        random_seed=budgeted_ucb1_tuned_seed,
    )
    budgeted_ucb1_tuned_rows = run_algorithm(
        algorithm_name="budgeted_ucb1_tuned",
        learner=budgeted_ucb1_tuned,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        benchmark_reward_per_round=budgeted_ucb_benchmark,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    budgeted_ucb1_tuned_path = save_algorithm_results(
        budgeted_ucb1_tuned_rows,
        Path(outputs_config["budgeted_ucb1_tuned_results_file"]),
    )

    budgeted_expected_reward_scaled_ucb_seed = random_seed + 3_000_003  # distinct stream
    budgeted_expected_reward_scaled_ucb = BudgetedExpectedRewardScaledUCB(
        actions=bids,
        horizon=horizon,
        initial_budget=initial_budget,
        max_expected_reward=ucb1_benchmark,
        random_seed=budgeted_expected_reward_scaled_ucb_seed,
    )
    budgeted_expected_reward_scaled_ucb_rows = run_algorithm(
        algorithm_name="budgeted_expected_reward_scaled_ucb",
        learner=budgeted_expected_reward_scaled_ucb,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
        min_competing_bid=min_competing_bid,
        max_competing_bid=max_competing_bid,
        benchmark_reward_per_round=budgeted_ucb_benchmark,
        competing_bid_distribution=competing_bid_distribution,
        n_competitors=n_competitors,
    )
    budgeted_expected_reward_scaled_ucb_path = save_algorithm_results(
        budgeted_expected_reward_scaled_ucb_rows,
        Path(outputs_config["budgeted_expected_reward_scaled_ucb_results_file"]),
    )
    combined_results = load_result_csvs([
        ucb1_path,
        budgeted_ucb_path,
        ucb1_tuned_path,
        expected_reward_scaled_ucb_path,
        budgeted_ucb1_tuned_path,
        budgeted_expected_reward_scaled_ucb_path,
    ])
    generate_standard_req1_plots(combined_results, plots_dir=PLOTS_DIR)

    print("Req 1 experiments completed.")


if __name__ == "__main__":
    main()