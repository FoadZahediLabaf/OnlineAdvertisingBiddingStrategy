"""Run Req3 factorized primal-dual on the 90 saved market matrices."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.piecewise_static_oracle import solve_piecewise_static_oracle
from ola.algorithms.primal_dual import FactorizedPrimalDualHedgeLearner


INPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req4_shared_realizations"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req3_on_req4_shared"
T = 3000
N = 10
BUDGET = 10000.0
BID_SET = np.linspace(0.0, 1.0, 11)
VALUES = np.ones(N)


def run_factorized_primal_dual_trial(
    env_seed,
    learner_seed,
    *,
    T,
    N,
    K_value,
    bid_set,
    values,
    budget,
    environment="precomputed",
    environment_kwargs=None,
):
    """Run one learner on an exact precomputed ``T x N`` realization."""

    del env_seed, K_value
    if environment != "precomputed":
        raise ValueError("Only the precomputed environment is supported.")
    matrix = np.load(Path(environment_kwargs["path"]))
    if matrix.shape != (T, N):
        raise ValueError(f"Expected {(T, N)}, found {matrix.shape}.")

    learner = FactorizedPrimalDualHedgeLearner(
        bid_set=bid_set,
        values=values,
        initial_budget=budget,
        horizon=T,
        learning_rate=0.2,
        random_seed=learner_seed,
    )
    utilities = np.zeros(T)
    costs = np.zeros(T)
    lambdas = np.zeros(T)
    remaining_budgets = np.zeros(T)
    chosen_actions = np.zeros((T, N))
    for round_index in range(T):
        action = learner.select_action()
        utility, cost, dual = learner.update(action, matrix[round_index])
        utilities[round_index] = utility
        costs[round_index] = cost
        lambdas[round_index] = dual
        remaining_budgets[round_index] = learner.budget_left
        chosen_actions[round_index] = action
    return {
        "utilities": utilities,
        "costs": costs,
        "lambdas": lambdas,
        "remaining_budgets": remaining_budgets,
        "chosen_actions": chosen_actions,
        "cumulative_utility": np.cumsum(utilities),
        "cumulative_cost": np.cumsum(costs),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timeseries_dir = OUTPUT_DIR / "timeseries"
    timeseries_dir.mkdir(exist_ok=True)
    raw_rows = []
    summary_rows = []

    for setting_dir in sorted(path for path in INPUT_DIR.iterdir() if path.is_dir()):
        config = json.loads((setting_dir / "config.json").read_text(encoding="utf-8"))
        setting_rows = []
        for realization, matrix_path in enumerate(sorted(setting_dir.glob("realization_*.npy"))):
            result = run_factorized_primal_dual_trial(
                0,
                2000 + realization,
                T=T,
                N=N,
                K_value=config["oracle_block_length"],
                bid_set=BID_SET,
                values=VALUES,
                budget=BUDGET,
                environment_kwargs={"path": matrix_path},
            )
            oracle = solve_piecewise_static_oracle(
                np.load(matrix_path),
                config["change_points"],
                BID_SET,
                VALUES,
                BUDGET,
            )
            regret = oracle["cumulative_utility"] - result["cumulative_utility"]
            compact = {
                "cumulative_utility": result["cumulative_utility"],
                "cumulative_cost": result["cumulative_cost"],
                "cumulative_oracle_utility": oracle["cumulative_utility"],
                "cumulative_oracle_cost": oracle["cumulative_cost"],
                "regret": regret,
                "remaining_budget": result["remaining_budgets"],
                "lambda_dual": result["lambdas"],
                "mean_bid": result["chosen_actions"].mean(axis=1),
            }
            setting_rows.append(compact)
            raw_rows.append(
                {
                    "setting_id": config["setting_id"],
                    "frequency_label": config["frequency_label"],
                    "skl_label": config["skl_label"],
                    "target_skl": config["target_skl"],
                    "change_frequency": config["change_frequency"],
                    "realization": realization,
                    "matrix_file": str(matrix_path.relative_to(PROJECT_ROOT)),
                    "final_utility": float(result["cumulative_utility"][-1]),
                    "final_cost": float(result["cumulative_cost"][-1]),
                    "final_oracle_utility": float(oracle["cumulative_utility"][-1]),
                    "final_oracle_cost": float(oracle["cumulative_cost"][-1]),
                    "final_regret": float(regret[-1]),
                    "budget_left": float(result["remaining_budgets"][-1]),
                }
            )
        stacked = {key: np.stack([row[key] for row in setting_rows]) for key in setting_rows[0]}
        np.savez_compressed(timeseries_dir / f"{config['setting_id']}.npz", **stacked)

    for setting_id in sorted({row["setting_id"] for row in raw_rows}):
        group = [row for row in raw_rows if row["setting_id"] == setting_id]
        first = group[0]
        summary = {
            key: np.asarray([row[key] for row in group], dtype=float)
            for key in ["final_utility", "final_cost", "final_oracle_utility", "final_oracle_cost", "final_regret", "budget_left"]
        }
        summary_rows.append(
            {
                "setting_id": setting_id,
                "frequency_label": first["frequency_label"],
                "skl_label": first["skl_label"],
                "target_skl": first["target_skl"],
                "change_frequency": first["change_frequency"],
                "n_realizations": len(group),
                **{
                    f"{name.removeprefix('final_')}_{stat}": float(
                        values.mean() if stat == "mean" else values.std(ddof=1)
                    )
                    for name, values in summary.items()
                    if name != "budget_left"
                    for stat in ["mean", "std"]
                },
                "budget_left_mean": float(summary["budget_left"].mean()),
                "budget_left_max": float(summary["budget_left"].max()),
                "n_change_points": len(
                    json.loads((INPUT_DIR / setting_id / "config.json").read_text())["change_points"]
                ),
            }
        )

    for path, rows in [
        (OUTPUT_DIR / "req3_factorized_raw.csv", raw_rows),
        (OUTPUT_DIR / "req3_factorized_summary.csv", summary_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
