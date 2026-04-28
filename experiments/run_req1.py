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
from typing import Any, Dict, List
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.algorithms.budgeted_ucb import BudgetedUCB
from ola.algorithms.ucb1 import UCB1
from ola.environments.single_campaign import SingleCampaignEnvironment
from ola.utils.logging import save_results
from ola.utils.plotting import generate_standard_req1_plots, load_result_csvs


RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def run_algorithm(
    algorithm_name: str,
    learner: Any,
    value: float,
    initial_budget: float,
    horizon: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    """Run one learner in the single-campaign environment."""
    env = SingleCampaignEnvironment(
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        min_competing_bid=0.0,
        max_competing_bid=1.0,
        random_seed=random_seed,
    )
    env.reset()

    rows: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0

    for _ in range(horizon):
        action = learner.select_action()
        outcome = env.step(action)

        reward = float(outcome["reward"])
        cost = float(outcome["cost"])

        if isinstance(learner, BudgetedUCB):
            learner.update(action=outcome["action"], reward=reward, cost=cost)
        else:
            learner.update(action=outcome["action"], reward=reward)

        cumulative_reward += reward
        cumulative_cost += cost

        rows.append(
            {
                "algorithm": algorithm_name,
                "round": outcome["round"],
                "action": outcome["action"],
                "bids": outcome["bids"],
                "wins": outcome["wins"],
                "reward": reward,
                "cost": cost,
                "budget_left": outcome["budget_left"],
                "cumulative_reward": cumulative_reward,
                "cumulative_cost": cumulative_cost,
                "benchmark_reward": None,
                "instant_regret": None,
                "cumulative_regret": None,
                "highest_competing_bid": outcome["info"]["highest_competing_bid"],
                "done": outcome["done"],
            }
        )

        if outcome["done"]:
            break

    return rows


def save_algorithm_results(rows: List[Dict[str, Any]], filename: str) -> Path:
    """Save one algorithm's results using the shared CSV format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename
    save_results(rows, str(output_path))
    print(f"Saved {len(rows)} rows to {output_path}")
    return output_path


def main() -> None:
    horizon = 1000
    value = 1.0
    initial_budget = 300.0
    random_seed = 42
    bids = np.linspace(0.0, 1.0, 11)

    ucb1 = UCB1(actions=bids, horizon=horizon)
    ucb1_rows = run_algorithm(
        algorithm_name="ucb1",
        learner=ucb1,
        value=value,
        initial_budget=initial_budget,
        horizon=horizon,
        random_seed=random_seed,
    )
    ucb1_path = save_algorithm_results(ucb1_rows, "req1_ucb1.csv")

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
    )
    budgeted_ucb_path = save_algorithm_results(
        budgeted_ucb_rows,
        "req1_budgeted_ucb.csv",
    )

    combined_results = load_result_csvs([ucb1_path, budgeted_ucb_path])
    generate_standard_req1_plots(combined_results, plots_dir=PLOTS_DIR)

    print("Req 1 experiments completed.")


if __name__ == "__main__":
    main()
