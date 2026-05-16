"""
Experiment runner for Requirement 2.

Owner:
    Luigi

Purpose:
    Run multi-campaign stochastic experiments for Combinatorial-UCB with budget.

Do not:
    - Add Requirement 1, Requirement 3, or Requirement 4 logic here.
    - Change the standard CSV result format.
    - Save final results only as screenshots.
"""

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import truncnorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.algorithms.combinatorial_ucb import CombinatorialUCB
from ola.environments.multi_campaign import MultiCampaignEnvironment
from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set
from ola.utils.logging import save_results


CONFIG_PATH = PROJECT_ROOT / "configs" / "req2.yaml"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RAW_DIR = PROJECT_ROOT / "results" / "raw"


def load_req2_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load Requirement 2 parameters from the YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Req 2 config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid Req 2 config file: {config_path}")

    return config


def compute_req2_benchmark(
    campaign_values: List[float],
    bid_values: List[float],
    initial_budget: float,
    horizon: int,
    conflict_edges: List[List[int]],
    mean_competing_bids: List[float],
    std_competing_bids: float,
    correlation_on_edge: float,
) -> float:
    """Compute the per-round pseudo-regret benchmark for Req 2.

    Implements T * max_{a in S} sum_{i in a} mu(i) from slide 15 of
    8combinatorialBandits.pdf (Castiglioni), expressed per round. The
    maximisation is over feasible super-arms (sets of mutually compatible
    campaigns) under the conflict graph. Competing bids per campaign are
    modelled as independent Gaussians with the configured mean and std,
    truncated to [0, 1] via scipy.stats.truncnorm; correlation_on_edge is
    accepted for signature symmetry with the environment but is not used here
    since the marginal distribution per campaign is what drives the win
    probability. For each (campaign, bid):
        P(win | b) = CDF_truncnorm(b)
        E[r | b]   = (value - b) * P(win | b)
        E[c | b]   = b * P(win | b)
    The best super-arm picks, for each campaign in it, the bid that maximises
    E[r | b]. The reward is scaled down proportionally if its expected cost
    exceeds the per-round budget share rho = initial_budget / horizon.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if initial_budget < 0:
        raise ValueError("initial_budget must be non-negative.")
    if len(bid_values) == 0:
        raise ValueError("Must have at least one bid value.")
    number_of_campaigns = len(campaign_values)
    if len(mean_competing_bids) != number_of_campaigns:
        raise ValueError("mean_competing_bids length must match campaign_values.")
    if std_competing_bids <= 0:
        raise ValueError("std_competing_bids must be positive.")
    del correlation_on_edge  # not used for marginal per-campaign probabilities

    bids = np.array(bid_values, dtype=float)

    # Per-campaign expected reward and cost under truncated Gaussian on [0, 1].
    expected_reward = np.zeros((number_of_campaigns, len(bids)))
    expected_cost = np.zeros((number_of_campaigns, len(bids)))
    for i in range(number_of_campaigns):
        mu = float(mean_competing_bids[i])
        sigma = float(std_competing_bids)
        a = (0.0 - mu) / sigma
        b = (1.0 - mu) / sigma
        win_probability = truncnorm.cdf(bids, a, b, loc=mu, scale=sigma)
        expected_reward[i] = (float(campaign_values[i]) - bids) * win_probability
        expected_cost[i] = bids * win_probability

    # Enumerate non-empty feasible super-arms under the conflict graph.
    conflict_graph = build_conflict_graph(
        number_of_campaigns=number_of_campaigns,
        edges=[tuple(e) for e in conflict_edges],
    )
    feasible_sets: List[frozenset] = []
    for size in range(1, number_of_campaigns + 1):
        for combo in combinations(range(number_of_campaigns), size):
            if is_feasible_campaign_set(combo, conflict_graph):
                feasible_sets.append(frozenset(combo))

    best_reward = 0.0
    best_cost = 0.0
    for feasible_set in feasible_sets:
        superarm_reward = 0.0
        superarm_cost = 0.0
        for i in feasible_set:
            best_bid_idx = int(np.argmax(expected_reward[i]))
            superarm_reward += float(expected_reward[i, best_bid_idx])
            superarm_cost += float(expected_cost[i, best_bid_idx])
        if superarm_reward > best_reward:
            best_reward = superarm_reward
            best_cost = superarm_cost

    rho = initial_budget / horizon
    if best_cost > rho and best_cost > 0.0:
        best_reward *= rho / best_cost

    return float(best_reward)


def run_single_seed(
    config: Dict[str, Any],
    seed: int,
    benchmark_reward_per_round: float,
) -> Tuple[List[Dict[str, Any]], Path, int]:
    """Run one seed end-to-end and return its rows, the per-seed CSV path, and total wins."""
    experiment_config = config["experiment"]
    campaigns_config = config["campaigns"]
    bids_config = config["bids"]
    conflict_graph_config = config["conflict_graph"]
    environment_config = config["environment"]
    outputs_config = config["outputs"]

    horizon = int(experiment_config["horizon"])
    number_of_campaigns = int(campaigns_config["number_of_campaigns"])
    campaign_values = [float(v) for v in campaigns_config["values"]]
    initial_budget = float(campaigns_config["initial_budget"])
    bid_values = [float(b) for b in bids_config["values"]]
    conflict_edges = [list(e) for e in conflict_graph_config["edges"]]
    mean_competing_bids = [float(m) for m in environment_config["mean_competing_bids"]]
    std_competing_bids = float(environment_config["std_competing_bids"])
    correlation_on_edge = float(environment_config["correlation_on_edge"])

    env = MultiCampaignEnvironment(
        number_of_campaigns=number_of_campaigns,
        campaign_values=campaign_values,
        bid_values=bid_values,
        budget=initial_budget,
        horizon=horizon,
        conflict_edges=conflict_edges,
        mean_competing_bids=mean_competing_bids,
        std_competing_bids=std_competing_bids,
        correlation_on_edge=correlation_on_edge,
        random_seed=seed,
    )
    reset_info = env.reset()
    info = reset_info["info"]

    agent = CombinatorialUCB(
        number_of_campaigns=int(info["number_of_campaigns"]),
        campaign_values=list(info["campaign_values"]),
        bid_values=list(info["bid_values"]),
        budget=initial_budget,
        horizon=int(info["horizon"]),
        conflict_edges=conflict_edges,
        random_seed=seed,
    )

    rows: List[Dict[str, Any]] = []
    cumulative_regret = 0.0
    total_wins = 0
    done = bool(reset_info["done"])

    while not done:
        action = agent.select_action()
        step_result = env.step(action)
        agent.update(action, step_result)

        reward = float(step_result["reward"])
        cost = float(step_result["cost"])
        instant_regret = float(benchmark_reward_per_round) - reward
        cumulative_regret += instant_regret
        # Track wins locally for the summary; not exported in the CSV.
        total_wins += int(sum(step_result["wins"]))

        rows.append(
            {
                "algorithm": "combinatorial_ucb",
                "round": step_result["round"],
                "action": str(action["active_campaigns"]),
                "bids": str(step_result["bids"]),
                "wins": str(step_result["wins"]),
                "reward": reward,
                "cost": cost,
                "budget_left": step_result["budget_left"],
                "cumulative_reward": agent.cumulative_reward,
                "cumulative_cost": agent.cumulative_cost,
                "benchmark_reward": float(benchmark_reward_per_round),
                "instant_regret": instant_regret,
                "cumulative_regret": cumulative_regret,
                "seed": int(seed),
                "done": step_result["done"],
            }
        )

        done = bool(step_result["done"])

    per_seed_filename = f"req2_combinatorial_ucb_seed{seed}.csv"
    output_path = RAW_DIR / per_seed_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(rows, str(output_path))

    return rows, output_path, total_wins


def plot_with_uncertainty_band(
    combined: pd.DataFrame,
    metric: str,
    horizon: int,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    """Plot mean +/- std across seeds with an uncertainty band.

    Local plotter for Req 2 with uncertainty bands. ola/utils/plotting.py does
    not yet support multi-seed uncertainty bands nor cumulative_cost; this
    local plotter will be removed once the shared utility is extended.

    Seeds that terminate early (done=True before T due to budget depletion) are
    forward-filled with their last observed value up to the horizon so that
    every series has the same length when aggregated by round.
    """
    pivot = combined.pivot(index="round", columns="seed", values=metric)
    full_index = pd.RangeIndex(1, horizon + 1, name="round")
    pivot = pivot.reindex(full_index).ffill()

    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(full_index, mean, label="combinatorial_ucb")
    plt.fill_between(full_index, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {output_path}")


def print_summary(
    per_seed_stats: List[Dict[str, Any]],
    horizon: int,
) -> None:
    """Print per-seed termination and final stats plus cross-seed mean/std."""
    print("\nPer-seed summary:")
    print(f"{'seed':>5} {'done_round':>11} {'cum_reward':>12} {'cum_cost':>10} {'wins':>6}")
    for row in per_seed_stats:
        print(
            f"{row['seed']:>5} "
            f"{row['done_round']:>11} "
            f"{row['cum_reward']:>12.4f} "
            f"{row['cum_cost']:>10.4f} "
            f"{row['wins']:>6}"
        )

    rewards = np.array([r["cum_reward"] for r in per_seed_stats])
    costs = np.array([r["cum_cost"] for r in per_seed_stats])
    wins = np.array([r["wins"] for r in per_seed_stats])
    done_rounds = np.array([r["done_round"] for r in per_seed_stats])

    print("\nCross-seed aggregates (mean +/- std over {} seeds, horizon {}):".format(
        len(per_seed_stats), horizon
    ))
    print(f"  done_round       : {done_rounds.mean():.2f} +/- {done_rounds.std():.2f}")
    print(f"  cumulative_reward: {rewards.mean():.4f} +/- {rewards.std():.4f}")
    print(f"  cumulative_cost  : {costs.mean():.4f} +/- {costs.std():.4f}")
    print(f"  total_wins       : {wins.mean():.2f} +/- {wins.std():.2f}")


def main() -> None:
    config = load_req2_config()

    experiment_config = config["experiment"]
    campaigns_config = config["campaigns"]
    bids_config = config["bids"]
    conflict_graph_config = config["conflict_graph"]
    environment_config = config["environment"]
    outputs_config = config["outputs"]

    horizon = int(experiment_config["horizon"])
    seeds = [int(s) for s in experiment_config["seeds"]]
    campaign_values = [float(v) for v in campaigns_config["values"]]
    initial_budget = float(campaigns_config["initial_budget"])
    bid_values = [float(b) for b in bids_config["values"]]
    conflict_edges = [list(e) for e in conflict_graph_config["edges"]]
    mean_competing_bids = [float(m) for m in environment_config["mean_competing_bids"]]
    std_competing_bids = float(environment_config["std_competing_bids"])
    correlation_on_edge = float(environment_config["correlation_on_edge"])

    benchmark = compute_req2_benchmark(
        campaign_values=campaign_values,
        bid_values=bid_values,
        initial_budget=initial_budget,
        horizon=horizon,
        conflict_edges=conflict_edges,
        mean_competing_bids=mean_competing_bids,
        std_competing_bids=std_competing_bids,
        correlation_on_edge=correlation_on_edge,
    )

    all_rows: List[Dict[str, Any]] = []
    per_seed_stats: List[Dict[str, Any]] = []

    for seed in seeds:
        rows, per_seed_path, total_wins = run_single_seed(
            config=config,
            seed=seed,
            benchmark_reward_per_round=benchmark,
        )
        all_rows.extend(rows)

        # done_round = first round where done=True, otherwise the horizon.
        done_round = horizon
        for row in rows:
            if row["done"]:
                done_round = int(row["round"])
                break

        final_row = rows[-1] if rows else None
        cum_reward = float(final_row["cumulative_reward"]) if final_row else 0.0
        cum_cost = float(final_row["cumulative_cost"]) if final_row else 0.0

        per_seed_stats.append(
            {
                "seed": seed,
                "done_round": done_round,
                "cum_reward": cum_reward,
                "cum_cost": cum_cost,
                "wins": total_wins,
            }
        )
        print(f"Seed {seed}: done at round {done_round}, "
              f"cum_reward={cum_reward:.4f}, cum_cost={cum_cost:.4f}, wins={total_wins} "
              f"-> {per_seed_path.name}")

    combined_path = Path(outputs_config["combinatorial_ucb_results_file"])
    if not combined_path.is_absolute():
        combined_path = PROJECT_ROOT / combined_path
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_rows, str(combined_path))
    print(f"\nSaved combined {len(all_rows)} rows to {combined_path}")

    combined_df = pd.DataFrame(all_rows)

    plot_with_uncertainty_band(
        combined=combined_df,
        metric="cumulative_reward",
        horizon=horizon,
        output_path=PLOTS_DIR / "req2_cumulative_reward.png",
        ylabel="Cumulative reward",
        title="Requirement 2 - Cumulative reward",
    )
    plot_with_uncertainty_band(
        combined=combined_df,
        metric="cumulative_regret",
        horizon=horizon,
        output_path=PLOTS_DIR / "req2_cumulative_regret.png",
        ylabel="Cumulative regret",
        title="Requirement 2 - Cumulative regret",
    )
    plot_with_uncertainty_band(
        combined=combined_df,
        metric="budget_left",
        horizon=horizon,
        output_path=PLOTS_DIR / "req2_budget_left.png",
        ylabel="Budget left",
        title="Requirement 2 - Budget left",
    )
    plot_with_uncertainty_band(
        combined=combined_df,
        metric="cumulative_cost",
        horizon=horizon,
        output_path=PLOTS_DIR / "req2_cumulative_cost.png",
        ylabel="Cumulative cost",
        title="Requirement 2 - Cumulative cost",
    )

    print_summary(per_seed_stats, horizon=horizon)
    print("\nReq 2 experiments completed.")


if __name__ == "__main__":
    main()
