"""Run the Req3 factorized primal-dual learner on saved realization matrices."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.piecewise_static_oracle import solve_piecewise_static_oracle  # noqa: E402
from ola.algorithms.primal_dual import FactorizedPrimalDualHedgeLearner  # noqa: E402


INPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req4_shared_realizations"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req3_on_req4_shared"
T = 3000
N = 5
BUDGET = 2500.0
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
    allowed_campaign_sets=None,
    conflict_edges=None,
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
        allowed_campaign_sets=allowed_campaign_sets,
        conflict_edges=conflict_edges,
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


def run_one(setting_dir: Path, realization_path: Path, realization_idx: int, config: dict):
    matrix = np.load(realization_path)
    result = run_factorized_primal_dual_trial(
        env_seed=0,
        learner_seed=2000 + realization_idx,
        T=T,
        N=N,
        K_value=int(config["oracle_block_length"]),
        bid_set=BID_SET,
        values=VALUES,
        budget=BUDGET,
        allowed_campaign_sets=config.get("allowed_campaign_sets"),
        conflict_edges=config.get("conflict_edges"),
        environment="precomputed",
        environment_kwargs={"path": realization_path},
    )
    oracle = solve_piecewise_static_oracle(
        market_matrix=matrix,
        change_points=config["change_points"],
        bid_set=BID_SET,
        values=VALUES,
        budget=BUDGET,
        allowed_campaign_sets=config.get("allowed_campaign_sets"),
        conflict_edges=config.get("conflict_edges"),
    )
    regret = oracle["cumulative_utility"] - result["cumulative_utility"]
    return {
        "setting_id": config["setting_id"],
        "frequency_label": config["frequency_label"],
        "skl_label": config["skl_label"],
        "target_skl": config["target_skl"],
        "change_frequency": config["change_frequency"],
        "realization": realization_idx,
        "matrix_file": str(realization_path.relative_to(PROJECT_ROOT)),
        "final_utility": float(result["cumulative_utility"][-1]),
        "final_cost": float(result["cumulative_cost"][-1]),
        "final_oracle_utility": float(oracle["cumulative_utility"][-1]),
        "final_oracle_cost": float(oracle["cumulative_cost"][-1]),
        "final_regret": float(regret[-1]),
        "budget_left": float(result["remaining_budgets"][-1]),
        "mean_bid": float(result["chosen_actions"].mean()),
        "mean_positive_bids_per_round": float((result["chosen_actions"] > 0).sum(axis=1).mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting_dir in sorted(path for path in INPUT_DIR.iterdir() if path.is_dir()):
        with (setting_dir / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        for realization_idx, realization_path in enumerate(sorted(setting_dir.glob("realization_*.npy"))):
            row = run_one(setting_dir, realization_path, realization_idx, config)
            rows.append(row)
            print(
                f"{row['setting_id']} r{realization_idx}: "
                f"utility={row['final_utility']:.2f}, cost={row['final_cost']:.2f}"
            )

    raw_path = OUTPUT_DIR / "req3_factorized_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    keys = sorted({(row["frequency_label"], row["skl_label"]) for row in rows})
    for frequency_label, skl_label in keys:
        group = [row for row in rows if row["frequency_label"] == frequency_label and row["skl_label"] == skl_label]
        metrics = {
            "utility": np.array([row["final_utility"] for row in group]),
            "cost": np.array([row["final_cost"] for row in group]),
            "oracle_utility": np.array([row["final_oracle_utility"] for row in group]),
            "oracle_cost": np.array([row["final_oracle_cost"] for row in group]),
            "regret": np.array([row["final_regret"] for row in group]),
        }
        summary_rows.append(
            {
                "frequency_label": frequency_label,
                "skl_label": skl_label,
                "n_realizations": len(group),
                **{
                    f"{name}_{stat}": float(values.mean() if stat == "mean" else values.std(ddof=1))
                    for name, values in metrics.items()
                    for stat in ["mean", "std"]
                },
                "budget_left_mean": float(np.mean([row["budget_left"] for row in group])),
            }
        )

    summary_path = OUTPUT_DIR / "req3_factorized_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
