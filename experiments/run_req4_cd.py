"""
Experiment runner for Requirement 4 - CHANGE-DETECTOR Combinatorial-UCB.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Run the change-detector (CUSUM) Combinatorial-UCB on the Requirement 4
    slightly non-stationary multi-campaign environment, for three SKL
    (non-stationarity) levels, repeated several times, and produce the
    cumulative regret / reward / cost plots with mean +/- std uncertainty bands,
    plus a plot of the number of detected changes over time.
    Added in a second moment, the logic for dealing with Normal Discrepancy change
    detection, which works only with global reset and has full-feedback.

Do not:
    - Add Requirement 1, Requirement 2, or Requirement 3 experiment logic here.
    - Add sliding-window or primal-dual learners here.
    - Change the standard CSV result format.
    - Save final results only as screenshots.
"""

from __future__ import annotations

import argparse
from itertools import combinations, product
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

from ola.algorithms.cd_cucb import ChangeDetectorCUCB
from ola.environments.non_stationary import NonStationaryMultiCampaignEnvironment
from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set
from ola.utils.logging import save_results


CONFIG_PATH = PROJECT_ROOT / "configs" / "req4.yaml"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RAW_DIR = PROJECT_ROOT / "results" / "raw"

ALGORITHM_NAME = "cd_cucb"
# Seed offsets so the three seed streams (constellation / sampling / agent) never
# collide for any (skl_index, repetition) pair. These MATCH run_req4.py so that,
# for a given (SKL, repetition), the change-detector learner sees the exact same
# non-stationary environment realisation as the sliding-window learner.
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


# CUSUM parameter derivation from the maximum number of changes U_T
def derive_cusum_parameters(horizon: int, U_T: int) -> Tuple[int, float, float]:
    """Derive the CUSUM-UCB parameters from the horizon T and the maximum number
    of distribution changes U_T, using the standard CUSUM-UCB formulas:
        M     = int(log(T / U_T))              # warmup / reference-window samples
        h     = 2 * log(T / U_T)               # detection threshold
        alpha = sqrt(U_T * log(T / U_T) / T)   # forced-exploration probability
    Returns (cusum_M, cusum_threshold, alpha) the three derived parameters.
    """
    if U_T < 1:
        raise ValueError(f"U_T must be >= 1 (got {U_T}).")
    if U_T >= horizon:
        raise ValueError(f"U_T must be < horizon (got U_T={U_T}, T={horizon}).")

    log_ratio = float(np.log(horizon / U_T))
    cusum_M = int(log_ratio)
    cusum_threshold = 2.0 * log_ratio
    alpha = float(np.sqrt(U_T * log_ratio / horizon))

    if cusum_M <= 0:
        raise ValueError(
            f"Derived cusum_M={cusum_M} <= 0 for T={horizon}, U_T={U_T}; "
            "U_T is too large relative to T (need T/U_T > e)."
        )
    return cusum_M, cusum_threshold, alpha


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
        # Marginal std per campaign = sqrt of the covariance diagonal (making sure that it is higher than a small positive value).
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


# SINGLE RUN: one (SKL, repetition) of change-detector CUCB
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
    cusum_M: int,
    cusum_epsilon: float,
    cusum_threshold: float,
    alpha: float,
    exploration_coefficient: float,
    reset_mode: str,
    detector_type: str,
    nd_window_size: int,
    nd_threshold: float,
    constellation_seed: int,
    sampling_seed: int,
    agent_seed: int,
    repetition: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Run one change-detector CUCB experiment.
    Returns:
        rows               : the per-round CSV rows.
        detections_per_arm : final (N, |B|) matrix of per-arm CUSUM (only) alarm counts."""
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

    agent = ChangeDetectorCUCB(
        number_of_campaigns=int(info["number_of_campaigns"]),
        campaign_values=list(info["campaign_values"]),
        bid_values=list(info["bid_values"]),
        budget=budget,
        horizon=int(info["horizon"]),
        conflict_edges=conflict_edges,
        random_seed=agent_seed,
        cusum_M=cusum_M,
        cusum_epsilon=cusum_epsilon,
        cusum_threshold=cusum_threshold,
        alpha=alpha,
        exploration_coefficient=exploration_coefficient,
        reset_mode=reset_mode,
        detector_type=detector_type,
        nd_window_size=nd_window_size,
        nd_threshold=nd_threshold,
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
        # the same per-interval empirical means as the benchmark -> pseudo-regret.
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
                # Cumulative number of CUSUM alarms raised up to this round, for
                # the "detected changes over time" plot.
                "detected_changes": int(agent.detected_changes),
                "constellation_seed": int(constellation_seed),
                "sampling_seed": int(sampling_seed),
                "done": bool(step_result["done"]),
            }
        )
        done = bool(step_result["done"])

    return rows, agent.detections_per_arm.copy()


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


def plot_detections_per_arm(
    detections_by_skl: Dict[float, List[np.ndarray]],   # Dict that stores the detection matrices (of the various repetitions) for each SKL value.
    skl_values: Sequence[float],
    bid_values: Sequence[float],
    number_of_campaigns: int,
    output_path: Path,
) -> None:
    """Heatmap of the MEAN per-arm CUSUM alarm count (campaign x bid), one panel
    per SKL level. Shows which arms drive the detection counts at each SKL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Take just the SKL values which have a non-empty detection list.
    panels = [s for s in skl_values if detections_by_skl.get(s)]
    if not panels:
        return

    # Shared colour scale across panels so they are visually comparable.
    vmax = max(
        float(np.mean(detections_by_skl[s], axis=0).max()) for s in panels
    )
    vmax = max(vmax, 1e-9)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4.8), squeeze=False
    )
    image = None
    for col, skl in enumerate(panels):
        ax = axes[0][col]
        mean_mat = np.mean(detections_by_skl[skl], axis=0)  # (number of campaigns, number of bids), average over the repetitions the detections.
        image = ax.imshow(
            mean_mat, aspect="auto", cmap="viridis", origin="upper",
            vmin=0.0, vmax=vmax,
        )
        ax.set_title(f"SKL = {skl:g}")
        ax.set_xlabel("Bid")
        ax.set_ylabel("Campaign")
        ax.set_xticks(range(len(bid_values)))
        ax.set_xticklabels([f"{b:g}" for b in bid_values], rotation=90, fontsize=7)
        ax.set_yticks(range(number_of_campaigns))
        ax.set_yticklabels(range(number_of_campaigns))
        for i in range(number_of_campaigns):
            for j in range(len(bid_values)):
                value = mean_mat[i, j]
                ax.text(
                    j, i, f"{value:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if value < vmax * 0.5 else "black",
                )

    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, pad=0.04,
                 label="Mean alarms per run")
    fig.suptitle("CD-CUCB - mean detections per arm (campaign x bid)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def save_detections_matrix_csv(
    detections_by_skl: Dict[float, List[np.ndarray]],
    skl_values: Sequence[float],
    bid_values: Sequence[float],
    number_of_campaigns: int,
    output_path: Path,
) -> None:
    """Save the per-arm detection counts as a long-format matrix CSV with columns
    (skl, campaign, bid, mean_detections, total_detections)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for skl in skl_values:
        mats = detections_by_skl.get(skl)
        if not mats:
            continue
        # Stack the obtained matrices of a single SKL value.
        stacked = np.stack(mats, axis=0)  # (reps, N, |B|)
        # Average (and sum)over the repetitions.
        mean_mat = stacked.mean(axis=0)
        total_mat = stacked.sum(axis=0)
        for i in range(number_of_campaigns):
            for j, bid in enumerate(bid_values):
                records.append(
                    {
                        "skl": float(skl),
                        "campaign": int(i),
                        "bid": float(bid),
                        "mean_detections": float(mean_mat[i, j]),
                        "total_detections": int(total_mat[i, j]),
                    }
                )
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Saved per-arm detection matrix to {output_path}")


# CHANGE-DETECTION DELAY (rounds between a true change and its first detection)

def compute_detection_delays(combined: pd.DataFrame) -> pd.DataFrame:
    """For each run, measure the delay (in rounds) between every true distribution
    change and the first CUSUM (Normal Discrepancy) alarm that follows it within that interval.
    Returns a long-format DataFrame:
        skl, repetition, change_index, change_round, detection_round, delay, detected"""
    records: List[Dict[str, Any]] = []
    for (skl, repetition), group in combined.groupby(["skl", "repetition"]):
        # Sort chronologically by round.
        group = group.sort_values("round")
        # Get the different list containing the actual number of round, the phase_id of the related round and the cumulative
        # number of detected changes up to that round.
        rounds = group["round"].to_numpy()
        phase = group["phase_id"].to_numpy()
        detected = group["detected_changes"].to_numpy()

        # Compute the rounds where an alarm fired (cumulative counter increased).
        alarm_increment = detected - np.concatenate(([0], detected[:-1]))
        alarm_rounds = rounds[alarm_increment > 0]

        # True change rounds: where the observed phase_id increments.
        change_positions = np.where(phase[1:] > phase[:-1])[0] + 1
        change_rounds = rounds[change_positions]

        for k, change_round in enumerate(change_rounds):
            # Find the next change round.
            next_change = (
                change_rounds[k + 1] if k + 1 < len(change_rounds) else np.inf
            )
            # Find the alarms that occurred after the change but before the next change.
            window = alarm_rounds[
                (alarm_rounds >= change_round) & (alarm_rounds < next_change)
            ]
            # If there is a detection in the window, take the first one and use it to compute the delay. Otherwise, mark the change as undetected.
            if window.size > 0:
                detection_round = int(window.min())
                delay = int(detection_round - change_round)
                was_detected = True
            else:
                detection_round = -1
                delay = np.nan
                was_detected = False

            records.append(
                {
                    "skl": float(skl),
                    "repetition": int(repetition),
                    "change_index": int(k + 1),          # 1-based change point
                    "change_round": int(change_round),
                    "detection_round": detection_round,
                    "delay": delay,
                    "detected": was_detected,
                }
            )
    return pd.DataFrame(records)


def plot_detection_delay(
    delays_df: pd.DataFrame,
    skl_values: Sequence[float],
    output_path: Path,
) -> None:
    """Bar chart of the mean detection delay per SKL (over detected change points),
    with std error bars and the detection rate annotated above each bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    means, stds, rates, labels = [], [], [], []
    for skl in skl_values:
        sub = delays_df[np.isclose(delays_df["skl"], skl)]
        if sub.empty:
            continue
        detected = sub[sub["detected"]]
        means.append(float(detected["delay"].mean()) if not detected.empty else 0.0)
        stds.append(float(detected["delay"].std()) if not detected.empty else 0.0)
        rates.append(float(sub["detected"].mean()))  # fraction of changes detected
        labels.append(f"SKL = {skl:g}")

    positions = range(len(labels))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(positions, means, yerr=stds, capsize=6, alpha=0.8)
    for pos, bar, rate in zip(positions, bars, rates):
        plt.text(
            pos, bar.get_height(), f"detected {rate * 100:.0f}%",
            ha="center", va="bottom", fontsize=9,
        )
    plt.xticks(list(positions), labels)
    plt.ylabel("Mean detection delay (rounds)")
    plt.title("Requirement 4 (CD-CUCB) - change-detection delay by SKL")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {output_path}")


# MAIN

def main() -> None:
    parser = argparse.ArgumentParser(description="Requirement 4 change-detector CUCB experiment.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick reduced run (short horizon, few repetitions) for validation.",
    )
    # Optional CD parameter overrides for sweeps; each falls back to the config
    # value when not provided. Use the YAML as the canonical default.
    parser.add_argument(
        "--exploration-coefficient",
        type=float,
        default=None,
        help="Override change_detector_cucb.exploration_coefficient from the config.",
    )
    parser.add_argument(
        "--cusum-epsilon",
        type=float,
        default=None,
        help="Override change_detector_cucb.cusum_epsilon from the config.",
    )
    parser.add_argument(
        "--U_T",
        type=int,
        default=None,
        help="Override change_detector_cucb.U_T (max number of changes) from the config.",
    )
    parser.add_argument(
        "--cusum-threshold",
        type=float,
        default=None,
        help="Override the CUSUM detection threshold h directly (decouples it from "
             "the U_T-derived value h = 2*log(T/U_T)). M and alpha stay U_T-derived.",
    )
    parser.add_argument(
        "--cusum-M",
        type=int,
        default=None,
        help="Override the CUSUM warm-up sample count M directly (decouples it from "
             "the U_T-derived value M = int(log(T/U_T))). h and alpha stay U_T-derived.",
    )
    parser.add_argument(
        "--reset-mode",
        choices=["per_arm", "global"],
        default=None,
        help="Override change_detector_cucb.reset_mode: 'per_arm' resets only the "
             "firing arm, 'global' restarts every arm + detector on any alarm.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Override experiment.number_of_repetitions (useful for quick parameter sweeps).",
    )
    parser.add_argument(
        "--detector-type",
        choices=["cusum", "nd_full_feedback"],
        default=None,
        help="Override change_detector_cucb.detector_type: 'cusum' (per-arm, pure "
             "bandit) or 'nd_full_feedback' (joint Normal Discrepancy on the competing "
             "bids -> FULL FEEDBACK, always global reset).",
    )
    parser.add_argument(
        "--nd-window-size",
        type=int,
        default=None,
        help="Override change_detector_cucb.nd_window_size (ND sliding window, in rounds).",
    )
    parser.add_argument(
        "--nd-threshold",
        type=float,
        default=None,
        help="Override change_detector_cucb.nd_threshold (ND alarm threshold).",
    )
    parser.add_argument(
        "--number-of-intervals",
        type=int,
        default=None,
        help="Override environment.number_of_intervals (number of fixed regimes). "
             "Remember U_T (max changes) is number_of_intervals - 1.",
    )
    args = parser.parse_args()

    config = load_req4_config()

    # Extract the sections of the config file. Then get the various parameters from the sections.
    experiment_config = config["experiment"]
    campaigns_config = config["campaigns"]
    bids_config = config["bids"]
    conflict_config = config["conflict_graph"]
    environment_config = config["environment"]
    cd_config = config["change_detector_cucb"]
    outputs_config = config["outputs_cd"]

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

    # Change-detector (CUSUM) parameters. M, threshold h and forced-exploration
    # alpha are derived from U_T (max number of changes) and the horizon T below;
    # epsilon and the exploration coefficient stay explicit.
    U_T = int(cd_config["U_T"])
    cusum_epsilon = float(cd_config["cusum_epsilon"])
    exploration_coefficient = float(cd_config["exploration_coefficient"])
    reset_mode = str(cd_config.get("reset_mode", "per_arm"))

    # Detector selection: "cusum" (per-arm bandit) or "nd_full_feedback" (only global reset and use
    # a full-feedback approach).
    detector_type = str(cd_config.get("detector_type", "cusum"))
    nd_window_size = int(cd_config.get("nd_window_size", 50))
    nd_threshold = float(cd_config.get("nd_threshold", 3.0))

    # Apply optional CLI overrides (for parameter sweeps) before deriving M/h/alpha.
    if args.U_T is not None:
        U_T = int(args.U_T)
    if args.cusum_epsilon is not None:
        cusum_epsilon = float(args.cusum_epsilon)
    if args.exploration_coefficient is not None:
        exploration_coefficient = float(args.exploration_coefficient)
    if args.reset_mode is not None:
        reset_mode = str(args.reset_mode)
    if args.detector_type is not None:
        detector_type = str(args.detector_type)
    if args.nd_window_size is not None:
        nd_window_size = int(args.nd_window_size)
    if args.nd_threshold is not None:
        nd_threshold = float(args.nd_threshold)

    if args.smoke:
        horizon = 240
        number_of_repetitions = 2
        constellation_max_iter = 300
        print("[smoke] reduced run: horizon=240, repetitions=2, constellation_max_iter=300")

    # Optional repetitions override (applied after --smoke so it wins if both given).
    if args.repetitions is not None:
        number_of_repetitions = int(args.repetitions)

    # Derive M, h and alpha from U_T and the (possibly smoke-reduced) horizon T,
    # so the detector parameters scale with the actual scenario.
    cusum_M, cusum_threshold, alpha = derive_cusum_parameters(horizon, U_T)

    # Optional direct override of the detection threshold h.
    if args.cusum_threshold is not None:
        cusum_threshold = float(args.cusum_threshold)
        if cusum_threshold <= 0:
            raise ValueError(f"--cusum-threshold must be > 0 (got {cusum_threshold}).")

    # Optional direct override of the warm-up sample count M.
    if args.cusum_M is not None:
        cusum_M = int(args.cusum_M)
        if cusum_M <= 0:
            raise ValueError(f"--cusum-M must be > 0 (got {cusum_M}).")

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
        f"horizon={horizon}, budget={budget} (rho=B/T={budget / horizon:.3f})."
    )
    if detector_type == "nd_full_feedback":
        print(
            f"Detector: nd_full_feedback (window={nd_window_size}, threshold={nd_threshold}), "
            f"exploration_coefficient={exploration_coefficient}. "
            "Reset is always GLOBAL (reset_mode ignored). The CUSUM params (M/h/alpha) "
            "are unused."
        )
        print(
            "  *** FULL-FEEDBACK detector: it reads the competing bids, so this run is "
            "NOT directly comparable to the bandit SW-CUCB / CUSUM-CUCB. Disclose the "
            "information asymmetry when comparing. ***"
        )
    else:
        print(
            f"Detector: cusum (per-arm, bandit). "
            f"Params (U_T={U_T} -> derived): M={cusum_M}, threshold={cusum_threshold:.4f}, "
            f"alpha={alpha:.4f}; epsilon={cusum_epsilon}, "
            f"exploration_coefficient={exploration_coefficient}, reset_mode={reset_mode}."
        )

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    # Per-SKL list of final (N, |B|) per-arm detection matrices, one per repetition.
    detections_by_skl: Dict[float, List[np.ndarray]] = {}

    for skl_index, target_skl in enumerate(skl_values):
        for repetition in range(number_of_repetitions):
            seed_base = base_seed + _SKL_SEED_STRIDE * (skl_index + 1) + repetition
            constellation_seed = seed_base + _CONSTELLATION_SEED_OFFSET
            sampling_seed = seed_base + _SAMPLING_SEED_OFFSET
            agent_seed = seed_base + _AGENT_SEED_OFFSET

            rows, detections_matrix = run_single_run(
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
                cusum_M=cusum_M,
                cusum_epsilon=cusum_epsilon,
                cusum_threshold=cusum_threshold,
                alpha=alpha,
                exploration_coefficient=exploration_coefficient,
                reset_mode=reset_mode,
                detector_type=detector_type,
                nd_window_size=nd_window_size,
                nd_threshold=nd_threshold,
                constellation_seed=constellation_seed,
                sampling_seed=sampling_seed,
                agent_seed=agent_seed,
                repetition=repetition,
            )
            # Add the new rows, individually, to the storage.
            all_rows.extend(rows)
            detections_by_skl.setdefault(target_skl, []).append(detections_matrix)

            # Take the last row and append it to the summary. The last row contains the final cumulative values of the experiment.
            final = rows[-1]
            summary.append(
                {
                    "skl": target_skl,
                    "repetition": repetition,
                    "final_cumulative_regret": float(final["cumulative_regret"]),
                    "final_cumulative_reward": float(final["cumulative_reward"]),
                    "final_cumulative_cost": float(final["cumulative_cost"]),
                    "final_detected_changes": int(final["detected_changes"]),
                    "done_round": int(final["round"]),
                }
            )
            print(
                f"SKL={target_skl:g} rep={repetition:>2}: "
                f"cum_regret={final['cumulative_regret']:.3f}, "
                f"cum_reward={final['cumulative_reward']:.3f}, "
                f"cum_cost={final['cumulative_cost']:.3f}, "
                f"detected_changes={final['detected_changes']}, "
                f"done@{final['round']}"
            )

    # Save combined raw CSV in the official format.
    raw_path = Path(outputs_config["raw_results_file"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_rows, str(raw_path))
    print(f"\nSaved {len(all_rows)} rows to {raw_path}")

    combined_df = pd.DataFrame(all_rows)

    # Four plots, each with one mean +/- std band per SKL level.
    plot_specs = [
        ("cumulative_regret", "cumulative_regret_plot",
         "Cumulative pseudo-regret", "Requirement 4 (CD-CUCB) - Cumulative pseudo-regret"),
        ("cumulative_reward", "cumulative_reward_plot",
         "Cumulative reward", "Requirement 4 (CD-CUCB) - Cumulative reward"),
        ("cumulative_cost", "cumulative_cost_plot",
         "Cumulative cost", "Requirement 4 (CD-CUCB) - Cumulative cost"),
        ("detected_changes", "detected_changes_plot",
         "Detected changes (cumulative)", "Requirement 4 (CD-CUCB) - Detected changes over time"),
    ]
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

    # Per-arm detection heatmap (campaign x bid) + the same data as a matrix CSV.
    det_plot_path = Path(outputs_config["detections_per_arm_plot"])
    if not det_plot_path.is_absolute():
        det_plot_path = PROJECT_ROOT / det_plot_path
    plot_detections_per_arm(
        detections_by_skl=detections_by_skl,
        skl_values=skl_values,
        bid_values=bid_values,
        number_of_campaigns=number_of_campaigns,
        output_path=det_plot_path,
    )

    det_matrix_path = Path(outputs_config["detections_per_arm_matrix"])
    if not det_matrix_path.is_absolute():
        det_matrix_path = PROJECT_ROOT / det_matrix_path
    save_detections_matrix_csv(
        detections_by_skl=detections_by_skl,
        skl_values=skl_values,
        bid_values=bid_values,
        number_of_campaigns=number_of_campaigns,
        output_path=det_matrix_path,
    )

    # Change-detection delay: rounds between each true change and its first alarm.
    delays_df = compute_detection_delays(combined_df)
    delay_matrix_path = Path(outputs_config["detection_delay_matrix"])
    if not delay_matrix_path.is_absolute():
        delay_matrix_path = PROJECT_ROOT / delay_matrix_path
    delay_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    delays_df.to_csv(delay_matrix_path, index=False)
    print(f"Saved detection-delay records to {delay_matrix_path}")

    delay_plot_path = Path(outputs_config["detection_delay_plot"])
    if not delay_plot_path.is_absolute():
        delay_plot_path = PROJECT_ROOT / delay_plot_path
    plot_detection_delay(delays_df, skl_values, delay_plot_path)

    print("\nChange-detection delay (over detected change points):")
    for skl in skl_values:
        sub = delays_df[np.isclose(delays_df["skl"], skl)]
        if sub.empty:
            continue
        detected = sub[sub["detected"]]
        rate = float(sub["detected"].mean()) * 100.0
        if detected.empty:
            print(f"  SKL={skl:g}: no change detected (detection rate 0%).")
        else:
            print(
                f"  SKL={skl:g}: mean delay={detected['delay'].mean():.1f} "
                f"+/- {detected['delay'].std():.1f} rounds, "
                f"median={detected['delay'].median():.0f}, "
                f"detection rate={rate:.0f}% "
                f"({int(sub['detected'].sum())}/{len(sub)} change points)."
            )

    # Cross-repetition summary per SKL level.
    summary_df = pd.DataFrame(summary)
    print("\nFinal cumulative pseudo-regret (mean +/- std over repetitions):")
    for skl in skl_values:
        sub = summary_df[np.isclose(summary_df["skl"], skl)]
        print(
            f"  SKL={skl:g}: "
            f"regret={sub['final_cumulative_regret'].mean():.3f} "
            f"+/- {sub['final_cumulative_regret'].std():.3f}, "
            f"reward={sub['final_cumulative_reward'].mean():.3f} "
            f"+/- {sub['final_cumulative_reward'].std():.3f}, "
            f"detected_changes={sub['final_detected_changes'].mean():.2f} "
            f"+/- {sub['final_detected_changes'].std():.2f}"
        )

    print("\nReq 4 change-detector experiment completed.")


if __name__ == "__main__":
    main()