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
from typing import Dict, Any, List
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.algorithms.budgeted_ucb import BudgetedUCB
from ola.algorithms.ucb1 import UCB1
from ola.environments.single_campaign import SingleCampaignEnvironment

RESULTS_DIR = PROJECT_ROOT / "results" / "raw"

def run_algorithm(
        algorithm_name: str,
        learner: Any,
        value: float,
        initial_budget:float,
        horizon: int,
        random_seed: int,
)-> List[Dict[str, Any]]:
    """Run one learner in the single_campaign environment"""
    env = SingleCampaignEnvironment(value=value, initial_budget=initial_budget, horizon=horizon, min_competing_bid=0.0, max_competing_bid=1.0, random_seed=random_seed)
    env.reset()

    rows=[]
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    for t in range(horizon):
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

        row = {
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
        rows.append(row)
        if outcome["done"]:
            break

    return rows

def save_results(rows: List[Dict[str, Any]], filename: str) -> None:
    """Save results to a CSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / filename
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")

def main() -> None:
    horizon = 1000
    value = 1.0
    initial_budget = 300.0
    random_seed = 42

    bids = np.linspace(0.0, 1.0, 11)

    ucb1= UCB1(actions=bids, horizon=horizon)
    ucb1_rows = run_algorithm(algorithm_name="ucb1", learner=ucb1, value=value, initial_budget=initial_budget, horizon=horizon, random_seed=random_seed)
    save_results(ucb1_rows, "req1_ucb1.csv")

    budget_ucb = BudgetedUCB(actions=bids, horizon=horizon, initial_budget=initial_budget)
    budget_rows = run_algorithm(algorithm_name="budgeted_ucb", learner=budget_ucb, value=value, initial_budget=initial_budget, horizon=horizon, random_seed=random_seed)
    save_results(budget_rows, "req1_budgeted_ucb.csv")
    print("Req 1 experiments completed.")


if __name__ == "__main__":
    main()
