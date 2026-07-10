"""
Experiment runner for Requirement 4.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Run the SLIDING-WINDOW Combinatorial-UCB on the Requirement 4 slightly
    non-stationary multi-campaign environment, for three SKL (non-stationarity)
    levels, repeated several times, and produce the cumulative regret / reward /
    cost plots with mean +/- std uncertainty bands.

Do not:
    - Add Requirement 1, Requirement 2, or Requirement 3 experiment logic here.
    - Change the standard CSV result format.
    - Save final results only as screenshots.
"""

from __future__ import annotations

import argparse
from itertools import combinations, product
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.algorithms.sw_cucb import SlidingWindowCUCB
from ola.environments.non_stationary import NonStationaryMultiCampaignEnvironment
from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set
from ola.utils.logging import save_results


CONFIG_PATH = PROJECT_ROOT / "configs" / "req4.yaml"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RAW_DIR = PROJECT_ROOT / "results" / "raw"

ALGORITHM_NAME = "sw_cucb"
# Seed offsets so the three seed streams (constellation / sampling / agent) never
# collide for any (skl_index, repetition) pair.
_CONSTELLATION_SEED_OFFSET = 0
_SAMPLING_SEED_OFFSET = 500000
_AGENT_SEED_OFFSET = 900000
_SKL_SEED_STRIDE = 10000


# CONFIG LOADING.
def load_req4_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load Requirement 4 parameters from the YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Req 4 config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Req 4 config file: {config_path}")
    return config


# Conflict graph: expand the co-runnable clusters into explicit conflict edges.
def build_conflict_edges_from_clusters(
    number_of_campaigns: int,
    clusters: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Two campaigns conflict iff they belong to different clusters.
    Within a cluster any subset of campaigns may run together; across clusters
    the campaigns are mutually exclusive. The clusters must be a partition of
    {0, ..., number_of_campaigns - 1}."""
    flat = [int(c) for cluster in clusters for c in cluster]
    if sorted(flat) != list(range(number_of_campaigns)):
        raise ValueError(
            "clusters must be a partition of all campaigns "
            f"(got {sorted(flat)} for N={number_of_campaigns})."
        )
    cluster_of = {}
    for cluster_index, cluster in enumerate(clusters):
        for campaign in cluster:
            cluster_of[int(campaign)] = cluster_index

    edges: List[List[int]] = []
    for i in range(number_of_campaigns):
        for j in range(i + 1, number_of_campaigns):
            if cluster_of[i] != cluster_of[j]:
                edges.append([i, j])
    return edges


def enumerate_feasible_sets(
    number_of_campaigns: int,
    conflict_edges: Sequence[Sequence[int]],
) -> List[Tuple[int, ...]]:
    """Return all non-empty feasible campaign sets (independent sets of the graph)."""
    graph = build_conflict_graph(
        number_of_campaigns=number_of_campaigns,
        edges=[tuple(e) for e in conflict_edges],
    )
    feasible: List[Tuple[int, ...]] = []
    for size in range(1, number_of_campaigns + 1):
        for combo in combinations(range(number_of_campaigns), size):
            if is_feasible_campaign_set(combo, graph):
                feasible.append(combo)
    return feasible


def count_candidate_super_arms(
    feasible_sets: Sequence[Tuple[int, ...]],
    number_of_bids: int,
) -> int:
    """Per-round candidate count = sum over feasible sets of |B|^|set|."""
    return int(sum(number_of_bids ** len(s) for s in feasible_sets))


# PIECEWISE PER-INTERVAL ORACLE (BENCHMARK) AND PER-INTERVAL EXPECTATIONS.
def compute_piecewise_expectations(
    env: NonStationaryMultiCampaignEnvironment,
    feasible_sets: Sequence[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute, for each interval, the per-(campaign, bid) expected reward and the
    benchmark (best feasible super-arm) expected per-round reward.
    The expectation is taken analytically over the true per-interval Gaussian
    distribution of the highest competing bids.
    Returns:
        benchmark : (number_of_intervals) best feasible expected per-round reward
        e_reward  : (number_of_intervals, N, |B|) expected reward of each specific arm"""
    number_of_campaigns = env.number_of_campaigns
    bids = np.asarray(env.bid_values, dtype=float)  # sorted ascending
    values = env.campaign_values
    number_of_intervals = env.number_of_intervals
    clip_low = float(env.clip_low)
    clip_high = float(env.clip_high)

    # Extract the expected reward for each (campaign, bid) on each interval, and the benchmark which
    # is the best expected reward per-round in the selected interval.
    e_reward = np.zeros((number_of_intervals, number_of_campaigns, len(bids)))
    benchmark = np.zeros(number_of_intervals)

    for interval in range(number_of_intervals):
        # Take the Gaussian (mean vector, covariance matrix) of this interval's regime.
        mean, covariance = env.constellation[interval]
        mean = np.asarray(mean, dtype=float)
        # Marginal std per campaign = sqrt of the covariance diagonal.
        std = np.sqrt(np.clip(np.diag(np.asarray(covariance, dtype=float)), 1e-18, None))
        for campaign in range(number_of_campaigns):
            value = float(values[campaign])
            mu_i = float(mean[campaign])
            sd_i = float(std[campaign])
            # Compute the P(win | b) under the Gaussian competing bid.
            win_prob = norm.cdf((bids - mu_i) / sd_i)
            # If the bid is above the clip threshold, it'll always win; on the other hand, if the bid is below
            # the clip threshold, it'll always lose. In any case, the bid values should be found inside the clip interval.
            win_prob = np.where(bids >= clip_high, 1.0, win_prob)
            win_prob = np.where(bids < clip_low, 0.0, win_prob)

            # Compute the expected reward for the many bids, for every campaign, in all the possible intervals.
            e_reward[interval, campaign] = (value - bids) * win_prob

        # Best feasible super-arm: rewards are separable across campaigns and the
        # per-campaign best expected reward is non-negative, so the best feasible
        # set just sums each member's best bid.
        per_campaign_best = e_reward[interval].max(axis=1)  # Take the best (max) expected reward found for each campaign, inside the given interval.
        best = 0.0 
        # Check, for all the feasible sets of campaigns, which is the value of the combined best bids, and set
        # that value as the benchmark for the relative interval.
        for feasible_set in feasible_sets:
            value = float(sum(per_campaign_best[i] for i in feasible_set))
            if value > best:
                best = value
        benchmark[interval] = best

    return benchmark, e_reward


# SINGLE RUN: ONE (SKL, repetition) OF SLIDING-WINDOW CUCB

def run_single_run(
    *,
    number_of_campaigns: int,
    campaign_values: List[float],
    bid_values: List[float],
    budget: float,
    horizon: int,
    number_of_intervals: int,
    conflict_edges: List[List[int]],
    feasible_sets: Sequence[Tuple[int, ...]],
    target_skl: float,
    max_std: float,
    constellation_max_iter: int,
    clip_low: float,
    clip_high: float,
    window_size: int,
    exploration_coefficient: float,
    constellation_seed: int,
    sampling_seed: int,
    agent_seed: int,
    repetition: int,
) -> List[Dict[str, Any]]:
    """Run one sliding-window CUCB experiment and return its per-round CSV rows."""
    env = NonStationaryMultiCampaignEnvironment(
        number_of_campaigns=number_of_campaigns,
        campaign_values=campaign_values,
        bid_values=bid_values,
        budget=budget,
        horizon=horizon,
        number_of_intervals=number_of_intervals,
        conflict_edges=conflict_edges,
        target_skl=target_skl,
        max_std=max_std,
        constellation_max_iter=constellation_max_iter,
        clip_low=clip_low,
        clip_high=clip_high,
        constellation_seed=constellation_seed,
        random_seed=sampling_seed,
    )
    reset_info = env.reset()
    info = reset_info["info"]

    benchmark, e_reward = compute_piecewise_expectations(env, feasible_sets)
    bid_grid = np.asarray(sorted(bid_values), dtype=float)

    agent = SlidingWindowCUCB(
        number_of_campaigns=int(info["number_of_campaigns"]),
        campaign_values=list(info["campaign_values"]),
        bid_values=list(info["bid_values"]),
        budget=budget,
        horizon=int(info["horizon"]),
        window_size=window_size,
        conflict_edges=conflict_edges,
        random_seed=agent_seed,
        exploration_coefficient=exploration_coefficient,
    )

    rows: List[Dict[str, Any]] = []
    cumulative_regret = 0.0
    done = bool(reset_info["done"])

    while not done:
        action = agent.select_action()
        step_result = env.step(action)
        agent.update(action, step_result)

        phase = int(step_result["info"]["phase_id"])

        # Expected reward of the chosen action under the current interval, using
        # the same per-interval means as the benchmark -> pseudo-regret.
        expected_reward = 0.0
        for position, campaign in enumerate(action["active_campaigns"]):
            bid_index = int(np.argmin(np.abs(bid_grid - float(action["bids"][position]))))
            expected_reward += float(e_reward[phase, campaign, bid_index])

        instant_regret = float(benchmark[phase] - expected_reward)
        cumulative_regret += instant_regret

        rows.append(
            {
                "algorithm": ALGORITHM_NAME,
                "round": step_result["round"],
                "action": str(action["active_campaigns"]),
                "bids": str(step_result["bids"]),
                "wins": str(step_result["wins"]),
                "reward": float(step_result["reward"]),
                "cost": float(step_result["cost"]),
                "budget_left": float(step_result["budget_left"]),
                "cumulative_reward": float(agent.cumulative_reward),
                "cumulative_cost": float(agent.cumulative_cost),
                "benchmark_reward": float(benchmark[phase]),
                "instant_regret": instant_regret,
                "cumulative_regret": cumulative_regret,
                # Extra columns
                "skl": float(target_skl),
                "repetition": int(repetition),
                "phase_id": phase,
                "constellation_seed": int(constellation_seed),
                "sampling_seed": int(sampling_seed),
                "done": bool(step_result["done"]),
            }
        )
        done = bool(step_result["done"])

    return rows


# PLOTTING

def plot_metric_by_skl(
    combined: pd.DataFrame,         
    metric: str,                    # Metric to be plotted.
    skl_values: Sequence[float],
    horizon: int,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    """Plot one metric over rounds, one mean +/- std band per SKL level.
    Repetitions that terminate early (budget depletion) are forward-filled with
    their last value up to the horizon so all series align by round."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Get the full list of rounds from 1 to the horizon.
    full_index = pd.RangeIndex(1, horizon + 1, name="round")

    plt.figure(figsize=(10, 6))
    for skl in skl_values:
        # Consider only the values associated to the related SKL value.
        subset = combined[np.isclose(combined["skl"], skl)]
        if subset.empty:
            continue
        # Create a table having rounds as rows, repetions as columns and taking as values the metric selected.
        pivot = subset.pivot(index="round", columns="repetition", values=metric)
        # Force the table to have all the rounds up to the length of the horizon.
        pivot = pivot.reindex(full_index).ffill()
        # Compute the mean and std of the metric across repetitions, for each round.
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1)
        # Add the mean line and the std band.
        line = plt.plot(full_index, mean, label=f"SKL = {skl:g}")[0]
        plt.fill_between(
            full_index, mean - std, mean + std, alpha=0.2, color=line.get_color()
        )

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {output_path}")


# MAIN
def main() -> None:
    parser = argparse.ArgumentParser(description="Requirement 4 sliding-window CUCB experiment.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick reduced run (short horizon, few repetitions) for validation.",
    )
    # Optional overrides for the SW parameters tuned during the experiments; each
    # falls back to the config value when not provided. The YAML stays the canonical
    # default.
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override sliding_window_cucb.window_size (in rounds). If neither this "
             "nor the config sets it, the default round(sqrt(horizon)) is used.",
    )
    parser.add_argument(
        "--exploration-coefficient",
        type=float,
        default=None,
        help="Override sliding_window_cucb.exploration_coefficient from the config.",
    )
    parser.add_argument(
        "--number-of-intervals",
        type=int,
        default=None,
        help="Override environment.number_of_intervals (number of fixed regimes).",
    )
    args = parser.parse_args()

    config = load_req4_config()

    # Extract the sections of the config file. Then get the various parameters from the sections.
    experiment_config = config["experiment"]
    campaigns_config = config["campaigns"]
    bids_config = config["bids"]
    conflict_config = config["conflict_graph"]
    environment_config = config["environment"]
    sw_config = config["sliding_window_cucb"]
    outputs_config = config["outputs"]

    horizon = int(experiment_config["horizon"])
    number_of_repetitions = int(experiment_config["number_of_repetitions"])
    base_seed = int(experiment_config["base_seed"])

    number_of_campaigns = int(campaigns_config["number_of_campaigns"])
    campaign_values = [float(v) for v in campaigns_config["values"]]
    budget = float(campaigns_config["initial_budget"])
    bid_values = [float(b) for b in bids_config["values"]]

    # Conflict graph: prefer an explicit conflict-edge list; otherwise expand
    # co-runnable clusters into the cross-cluster conflict edges.
    if conflict_config.get("edges") is not None:
        conflict_edges = [[int(a), int(b)] for a, b in conflict_config["edges"]]
        conflict_description = f"edges {conflict_edges}"
    else:
        clusters = [[int(c) for c in cluster] for cluster in conflict_config["clusters"]]
        conflict_edges = build_conflict_edges_from_clusters(number_of_campaigns, clusters)
        conflict_description = f"clusters {clusters}"

    number_of_intervals = int(environment_config["number_of_intervals"])
    if args.number_of_intervals is not None:
        number_of_intervals = int(args.number_of_intervals)
    skl_values = [float(s) for s in environment_config["skl_values"]]
    max_std = float(environment_config["max_std"])
    constellation_max_iter = int(environment_config["constellation_max_iter"])
    clip_low = float(environment_config["clip_low"])
    clip_high = float(environment_config["clip_high"])

    # Take the sliding window parameters from the config file, applying any CLI overrides.
    exploration_coefficient = float(sw_config["exploration_coefficient"])
    if args.exploration_coefficient is not None:
        exploration_coefficient = float(args.exploration_coefficient)
    configured_window = sw_config.get("window_size", None)
    if args.window_size is not None:
        configured_window = int(args.window_size)

    # Set a smoke run if the relative argument is passed.
    if args.smoke:
        horizon = 240
        number_of_repetitions = 2
        constellation_max_iter = 300
        print("[smoke] reduced run: horizon=240, repetitions=2, constellation_max_iter=300")

    # window_size: configured value, otherwise the theory default round(sqrt(T)).
    if configured_window is None:
        window_size = max(1, round(sqrt(horizon)))
    else:
        window_size = int(configured_window)

    # Compute the enumeration of the feasible campaign sets and count the candidate super-arms.
    feasible_sets = enumerate_feasible_sets(number_of_campaigns, conflict_edges)
    candidate_count = count_candidate_super_arms(feasible_sets, len(bid_values))

    # Print some information about the actual configuration of the experiment.
    print(
        f"Conflict graph: {conflict_description} -> {len(conflict_edges)} conflict edges, "
        f"{len(feasible_sets)} feasible campaign sets, "
        f"{candidate_count} candidate super-arms per round."
    )
    print(
        f"SKL levels {skl_values}, repetitions={number_of_repetitions}, "
        f"horizon={horizon}, window_size={window_size}, budget={budget} "
        f"(rho=B/T={budget / horizon:.3f})."
    )

    # Create two storage for the experiment's results.
    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    # For every SKL value, use it repeatedly a number_of_repetitions times. For every case, build the seed to be used
    # and run the experiment.
    for skl_index, target_skl in enumerate(skl_values):
        for repetition in range(number_of_repetitions):
            seed_base = base_seed + _SKL_SEED_STRIDE * (skl_index + 1) + repetition
            constellation_seed = seed_base + _CONSTELLATION_SEED_OFFSET
            sampling_seed = seed_base + _SAMPLING_SEED_OFFSET
            agent_seed = seed_base + _AGENT_SEED_OFFSET

            rows = run_single_run(
                number_of_campaigns=number_of_campaigns,
                campaign_values=campaign_values,
                bid_values=bid_values,
                budget=budget,
                horizon=horizon,
                number_of_intervals=number_of_intervals,
                conflict_edges=conflict_edges,
                feasible_sets=feasible_sets,
                target_skl=target_skl,
                max_std=max_std,
                constellation_max_iter=constellation_max_iter,
                clip_low=clip_low,
                clip_high=clip_high,
                window_size=window_size,
                exploration_coefficient=exploration_coefficient,
                constellation_seed=constellation_seed,
                sampling_seed=sampling_seed,
                agent_seed=agent_seed,
                repetition=repetition,
            )
            # Add the new rows, individually, to the storage.
            all_rows.extend(rows)

            # Take the last row and append it to the summary. The last row contains the final cumulative values of the experiment.
            final = rows[-1]
            summary.append(
                {
                    "skl": target_skl,
                    "repetition": repetition,
                    "final_cumulative_regret": float(final["cumulative_regret"]),
                    "final_cumulative_reward": float(final["cumulative_reward"]),
                    "final_cumulative_cost": float(final["cumulative_cost"]),
                    "done_round": int(final["round"]),
                }
            )
            print(
                f"SKL={target_skl:g} rep={repetition:>2}: "
                f"cum_regret={final['cumulative_regret']:.3f}, "
                f"cum_reward={final['cumulative_reward']:.3f}, "
                f"cum_cost={final['cumulative_cost']:.3f}, "
                f"done@{final['round']}"
            )

    # Save combined raw CSV in the official format.
    raw_path = Path(outputs_config["raw_results_file"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_rows, str(raw_path))
    print(f"\nSaved {len(all_rows)} rows to {raw_path}")

    # Create a dataframe with all the gathered rows.
    combined_df = pd.DataFrame(all_rows)

    # Three plots, each with one mean +/- std band per SKL level.
    plot_specs = [
        ("cumulative_regret", "cumulative_regret_plot",
         "Cumulative pseudo-regret", "Requirement 4 (SW-CUCB) - Cumulative pseudo-regret"),
        ("cumulative_reward", "cumulative_reward_plot",
         "Cumulative reward", "Requirement 4 (SW-CUCB) - Cumulative reward"),
        ("cumulative_cost", "cumulative_cost_plot",
         "Cumulative cost", "Requirement 4 (SW-CUCB) - Cumulative cost"),
    ]

    # For every metric, plot the results invoking the related method.
    for metric, output_key, ylabel, title in plot_specs:
        output_path = Path(outputs_config[output_key])
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        plot_metric_by_skl(
            combined=combined_df,
            metric=metric,
            skl_values=skl_values,
            horizon=horizon,
            output_path=output_path,
            ylabel=ylabel,
            title=title,
        )

    # Cross-repetition summary per SKL level.
    summary_df = pd.DataFrame(summary)
    print("\nFinal cumulative pseudo-regret (mean +/- std over repetitions):")
    # For every SKL value, take the subset of rows related to that value and print the mean and std of the final
    # cumulative metrics.
    for skl in skl_values:
        sub = summary_df[np.isclose(summary_df["skl"], skl)]
        print(
            f"  SKL={skl:g}: "
            f"regret={sub['final_cumulative_regret'].mean():.3f} "
            f"+/- {sub['final_cumulative_regret'].std():.3f}, "
            f"reward={sub['final_cumulative_reward'].mean():.3f} "
            f"+/- {sub['final_cumulative_reward'].std():.3f}"
        )

    print("\nReq 4 sliding-window experiment completed.")


if __name__ == "__main__":
    main()