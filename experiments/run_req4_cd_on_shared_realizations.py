"""
Experiment runner for Requirement 4 (change-detector CUCB) on the PRE-GENERATED
shared realizations.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Run the CHANGE-DETECTOR Combinatorial-UCB (CUSUM on the win indicator,
    or the joint Normal-Discrepancy full-feedback detector) on the saved Requirement
    4 shared realizations (results/processed/req4_shared_realizations), in REPLAY
    mode
    The CUSUM parameters (M, threshold h, forced-exploration alpha) are derived per
    setting from U_T = that setting's number of change points, via the standard
    CUSUM-UCB formulas in run_req4_cd.derive_cusum_parameters.

Do not:
    - Add Requirement 1, Requirement 2, or Requirement 3 experiment logic here.
    - Regenerate the shared realizations here (that is
      experiments/generate_req4_shared_realizations.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_req4 import count_candidate_super_arms, enumerate_feasible_sets
from experiments.run_req4_cd import derive_cusum_parameters
from ola.algorithms.cd_cucb import ChangeDetectorCUCB
from ola.environments.non_stationary import NonStationaryMultiCampaignEnvironment
from ola.utils.logging import save_results


INPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req4_shared_realizations"
PLOTS_BASE = PROJECT_ROOT / "results" / "plots" / "req4_comparison"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
RAW_RESULTS_FILE = RAW_DIR / "req4_cd_on_shared.csv"
DELAYS_RESULTS_FILE = RAW_DIR / "req4_cd_on_shared_delays.csv"

ALGORITHM_NAME = "cd_cucb"
# Agent seed scheme: deterministic per (setting, realization) so a re-run is
# reproducible. The competing-bid stream itself is fixed on disk, so only the
# agent's internal RNG (tie-breaking / forced exploration) needs a seed.
_AGENT_SEED_BASE = 900000


# CONFIG / DATA LOADING.
def load_setting(setting_dir: Path) -> Dict[str, Any]:
    """Load one setting's config, constellation modes and realization streams."""
    config = json.loads((setting_dir / "config.json").read_text(encoding="utf-8"))

    constellation = np.load(setting_dir / "constellation.npz")
    # modes[k] = (mu_k, sigma_k) for the k-th recurring Gaussian mode.
    modes = list(zip(constellation["mus"], constellation["sigmas"]))

    # Each realization is a (horizon, number_of_campaigns) competing-bid matrix.
    realization_paths = sorted(setting_dir.glob("realization_*.npy"))
    streams = [np.load(path) for path in realization_paths]

    return {"config": config, "modes": modes, "streams": streams}


def build_phase_ids(
    change_points: Sequence[int],
    mode_ids_by_window: Sequence[int],
    horizon: int,
) -> np.ndarray:
    """Expand the saved change schedule into a per-round mode id array."""
    boundaries = [0, *[int(c) for c in change_points], int(horizon)]
    windows = list(zip(boundaries[:-1], boundaries[1:]))
    if len(windows) != len(mode_ids_by_window):
        raise ValueError(
            "mode_ids_by_window must have one entry per window "
            f"({len(mode_ids_by_window)} != {len(windows)})."
        )
    phase_ids = np.zeros(int(horizon), dtype=int)
    for (start, end), mode_id in zip(windows, mode_ids_by_window):
        phase_ids[start:end] = int(mode_id)
    return phase_ids


# ANALYTIC ORACLE (same methodology as run_req4.py).
def compute_mode_expectations(
    modes: Sequence[Tuple[np.ndarray, np.ndarray]],
    campaign_values: Sequence[float],
    bid_values: Sequence[float],
    feasible_sets: Sequence[Tuple[int, ...]],
    clip_low: float,
    clip_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute, for each recurring mode, the per-(campaign, bid) expected reward and
    the benchmark (best feasible super-arm) expected per-round reward.
    The expectation is taken analytically over that mode's true Gaussian.
    Returns:
        benchmark : (number_of_modes,) best feasible expected per-round reward
        e_reward  : (number_of_modes, N, |B|) expected reward of each specific arm
    """
    bids = np.asarray(sorted(bid_values), dtype=float)
    values = np.asarray(campaign_values, dtype=float)
    number_of_campaigns = len(values)
    number_of_modes = len(modes)

    e_reward = np.zeros((number_of_modes, number_of_campaigns, len(bids)))
    benchmark = np.zeros(number_of_modes)

    for mode_index, (mean, covariance) in enumerate(modes):
        mean = np.asarray(mean, dtype=float)
        # Marginal std per campaign = sqrt of the covariance diagonal (floored).
        std = np.sqrt(np.clip(np.diag(np.asarray(covariance, dtype=float)), 1e-18, None))
        for campaign in range(number_of_campaigns):
            value = float(values[campaign])
            mu_i = float(mean[campaign])
            sd_i = float(std[campaign])
            # P(win | b) under the Gaussian competing bid, with clip corrections.
            win_prob = norm.cdf((bids - mu_i) / sd_i)
            win_prob = np.where(bids >= clip_high, 1.0, win_prob)
            win_prob = np.where(bids < clip_low, 0.0, win_prob)
            e_reward[mode_index, campaign] = (value - bids) * win_prob

        # Rewards are separable across campaigns and per-campaign best is >= 0, so
        # the best feasible super-arm just sums each member's best bid.
        per_campaign_best = e_reward[mode_index].max(axis=1)
        best = 0.0
        for feasible_set in feasible_sets:
            total = float(sum(per_campaign_best[i] for i in feasible_set))
            if total > best:
                best = total
        benchmark[mode_index] = best

    return benchmark, e_reward


# SINGLE REALIZATION: replay one saved stream through CD-CUCB.
def run_single_realization(
    *,
    stream: np.ndarray,
    phase_ids: np.ndarray,
    number_of_campaigns: int,
    campaign_values: List[float],
    bid_values: List[float],
    budget: float,
    horizon: int,
    conflict_edges: List[List[int]],
    benchmark: np.ndarray,
    e_reward: np.ndarray,
    detector_type: str,
    reset_mode: str,
    cusum_M: int,
    cusum_epsilon: float,
    cusum_threshold: float,
    alpha: float,
    exploration_coefficient: float,
    nd_window_size: int,
    nd_threshold: float,
    agent_seed: int,
    setting_id: str,
    frequency_label: str,
    skl_label: str,
    target_skl: float,
    realization: int,
) -> List[Dict[str, Any]]:
    """Replay one saved competing-bid stream through CD-CUCB and return CSV rows."""
    env = NonStationaryMultiCampaignEnvironment(
        number_of_campaigns=number_of_campaigns,
        campaign_values=campaign_values,
        bid_values=bid_values,
        budget=budget,
        horizon=horizon,
        conflict_edges=conflict_edges,
        clip_low=0.0,
        clip_high=1.0,
        competing_bid_sequence=stream,
        phase_ids=phase_ids,
    )
    reset_info = env.reset()
    info = reset_info["info"]

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

        # phase_id returned by the env is the active mode id (0..number_of_modes-1).
        mode = int(step_result["info"]["phase_id"])

        # Expected reward of the chosen action under the current mode, using the
        # same per-mode means as the benchmark -> non-negative pseudo-regret.
        expected_reward = 0.0
        for position, campaign in enumerate(action["active_campaigns"]):
            bid_index = int(np.argmin(np.abs(bid_grid - float(action["bids"][position]))))
            expected_reward += float(e_reward[mode, campaign, bid_index])

        instant_regret = float(benchmark[mode] - expected_reward)
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
                "benchmark_reward": float(benchmark[mode]),
                "instant_regret": instant_regret,
                "cumulative_regret": cumulative_regret,
                # Extra columns
                "setting_id": setting_id,
                "frequency_label": frequency_label,
                "skl_label": skl_label,
                "skl": float(target_skl),
                "realization": int(realization),
                "phase_id": mode,
                "detected_changes": int(agent.detected_changes),
                "done": bool(step_result["done"]),
            }
        )
        done = bool(step_result["done"])

    return rows


# PLOTTING: one mean +/- std band (over realizations) per setting.
def plot_metric_for_setting(
    setting_df: pd.DataFrame,
    metric: str,
    horizon: int,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    """Plot one metric over rounds with a single mean +/- std band over the
    realizations of one setting. Realizations that terminate early (budget
    depletion) are forward-filled with their last value up to the horizon."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_index = pd.RangeIndex(1, horizon + 1, name="round")

    pivot = setting_df.pivot(index="round", columns="realization", values=metric)
    pivot = pivot.reindex(full_index).ffill()
    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1)

    plt.figure(figsize=(10, 6))
    line = plt.plot(full_index, mean, label="mean over realizations")[0]
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
    print(f"  saved plot -> {output_path.relative_to(PROJECT_ROOT)}")


# DETECTION DELAY: per-setting, corrected for RECURRING modes.
def compute_detection_delays_for_setting(setting_df: pd.DataFrame) -> pd.DataFrame:
    """For each realization of one setting, measure the delay (in rounds) between
    every true regime change and the first detector alarm that follows it before the
    next change.
    Returns a long-format DataFrame:
        realization, change_index, change_round, detection_round, delay, detected
    """
    records: List[Dict[str, Any]] = []
    for realization, group in setting_df.groupby("realization"):
        group = group.sort_values("round")
        rounds = group["round"].to_numpy()
        phase = group["phase_id"].to_numpy()
        detected = group["detected_changes"].to_numpy()

        # Rounds where an alarm fired (cumulative counter increased).
        alarm_increment = detected - np.concatenate(([0], detected[:-1]))
        alarm_rounds = rounds[alarm_increment > 0]

        # True change rounds: ANY mode switch (recurring modes -> diff != 0).
        change_positions = np.where(np.diff(phase) != 0)[0] + 1
        change_rounds = rounds[change_positions]

        for k, change_round in enumerate(change_rounds):
            next_change = (
                change_rounds[k + 1] if k + 1 < len(change_rounds) else np.inf
            )
            window = alarm_rounds[
                (alarm_rounds >= change_round) & (alarm_rounds < next_change)
            ]
            if window.size > 0:
                detection_round = int(window.min())
                delay = float(detection_round - change_round)
                was_detected = True
            else:
                detection_round = -1
                delay = np.nan
                was_detected = False

            records.append(
                {
                    "realization": int(realization),
                    "change_index": int(k + 1),      # 1-based change point
                    "change_round": int(change_round),
                    "detection_round": detection_round,
                    "delay": delay,
                    "detected": was_detected,
                }
            )
    return pd.DataFrame(records)


def plot_detection_delay_for_setting(
    delays_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Histogram of detection delays over all (change point x realization) pairs of
    one setting, with mean/median lines and the detection rate in the title."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = int(len(delays_df))
    detected = delays_df[delays_df["detected"]]
    n_detected = int(len(detected))
    rate = (n_detected / n_total * 100.0) if n_total else 0.0

    plt.figure(figsize=(10, 6))
    if not detected.empty:
        delays = detected["delay"].to_numpy(dtype=float)
        max_delay = max(1, int(np.ceil(delays.max())))
        # Integer-aligned bins; cap the count to keep the plot readable.
        step = max(1, max_delay // 40)
        bins = np.arange(0, max_delay + step + 1, step) - 0.5
        plt.hist(delays, bins=bins, alpha=0.8, edgecolor="black")
        mean_delay = float(delays.mean())
        median_delay = float(np.median(delays))
        plt.axvline(mean_delay, color="red", linestyle="--",
                    label=f"mean = {mean_delay:.1f}")
        plt.axvline(median_delay, color="green", linestyle=":",
                    label=f"median = {median_delay:.0f}")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "no changes detected", ha="center", va="center",
                 transform=plt.gca().transAxes, fontsize=12)

    plt.xlabel("Detection delay (rounds)")
    plt.ylabel("Count (change point x realization)")
    plt.title(f"{title}\ndetection rate {rate:.0f}% ({n_detected}/{n_total} change points)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  saved plot -> {output_path.relative_to(PROJECT_ROOT)}")


# MAIN
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Requirement 4 change-detector CUCB on the saved shared realizations."
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Run only this setting_id (e.g. high_frequency_high_skl). Default: all 9.",
    )
    parser.add_argument(
        "--detector-type",
        choices=["cusum", "nd_full_feedback"],
        default="cusum",
        help="'cusum' (per-arm CUSUM on the win indicator, pure bandit) or "
             "'nd_full_feedback' (joint Normal Discrepancy on the competing bids -> "
             "FULL FEEDBACK, always global reset). Default: cusum.",
    )
    parser.add_argument(
        "--reset-mode",
        choices=["per_arm", "global"],
        default="global",
        help="What to forget on a detected change: 'per_arm' resets only the firing "
             "arm, 'global' restarts every arm + detector. Ignored for "
             "nd_full_feedback (always global). Default: global.",
    )
    parser.add_argument(
        "--exploration-coefficient",
        type=float,
        default=0.1,
        help="CUCB exploration coefficient. Default: 0.1 (matches the SW experiment).",
    )
    parser.add_argument(
        "--cusum-epsilon",
        type=float,
        default=0.1,
        help="CUSUM per-step slack (drift allowance), in (0, 1). Default: 0.1.",
    )
    parser.add_argument(
        "--nd-window-size",
        type=int,
        default=50,
        help="ND sliding-window length in rounds (only used by nd_full_feedback). "
             "Default: 50.",
    )
    parser.add_argument(
        "--nd-threshold",
        type=float,
        default=3.0,
        help="ND alarm threshold (only used by nd_full_feedback). Default: 3.0.",
    )
    parser.add_argument(
        "--cusum-threshold",
        type=float,
        default=None,
        help="Override the CUSUM detection threshold h directly (decouples it from "
             "the per-setting U_T-derived value). M and alpha stay U_T-derived.",
    )
    parser.add_argument(
        "--cusum-M",
        type=int,
        default=None,
        help="Override the CUSUM warm-up sample count M directly. h and alpha stay "
             "U_T-derived.",
    )
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing shared realizations directory: {INPUT_DIR}")

    # Discover the settings to run (all subfolders, or just the requested one).
    setting_dirs = sorted(path for path in INPUT_DIR.iterdir() if path.is_dir())
    if args.setting is not None:
        setting_dirs = [path for path in setting_dirs if path.name == args.setting]
        if not setting_dirs:
            raise ValueError(f"Unknown setting_id: {args.setting}")

    detector_type = str(args.detector_type)
    reset_mode = str(args.reset_mode)
    exploration_coefficient = float(args.exploration_coefficient)
    cusum_epsilon = float(args.cusum_epsilon)
    nd_window_size = int(args.nd_window_size)
    nd_threshold = float(args.nd_threshold)

    # Stable per-run "case" label used to group plots so different detector configs
    # never overwrite each other. Encodes the detector properties, not the per-setting derived CUSUM values.
    if detector_type == "nd_full_feedback":
        detector_case = f"nd_w{nd_window_size}_thr{nd_threshold:g}_ec{exploration_coefficient:g}"
    else:
        detector_case = (
            f"cusum_{reset_mode}_ec{exploration_coefficient:g}_eps{cusum_epsilon:g}"
        )

    all_rows: List[Dict[str, Any]] = []
    all_delays: List[pd.DataFrame] = []
    summary: List[Dict[str, Any]] = []

    for setting_dir in setting_dirs:
        loaded = load_setting(setting_dir)
        config = loaded["config"]
        modes = loaded["modes"]
        streams = loaded["streams"]

        setting_id = str(config["setting_id"])
        frequency_label = str(config["frequency_label"])
        skl_label = str(config["skl_label"])
        target_skl = float(config["target_skl"])
        horizon = int(config["horizon"])
        number_of_campaigns = int(config["number_of_campaigns"])
        budget = float(config["budget"])
        bid_values = [float(b) for b in config["bid_grid"]]
        conflict_edges = [[int(a), int(b)] for a, b in config["conflict_edges"]]
        # The saved streams use unit campaign values (generator default).
        campaign_values = [1.0] * number_of_campaigns

        number_of_changes = len(config["change_points"])

        # CUSUM parameters derived per setting from U_T = number of change points
        # (floored at 1 so a change-free setting still yields valid M/h/alpha).
        U_T = max(1, number_of_changes)
        cusum_M, cusum_threshold, alpha = derive_cusum_parameters(horizon, U_T)
        if args.cusum_threshold is not None:
            cusum_threshold = float(args.cusum_threshold)
        if args.cusum_M is not None:
            cusum_M = int(args.cusum_M)

        # Feasible super-arms and the per-mode analytic oracle for this setting.
        feasible_sets = enumerate_feasible_sets(number_of_campaigns, conflict_edges)
        candidate_count = count_candidate_super_arms(feasible_sets, len(bid_values))
        benchmark, e_reward = compute_mode_expectations(
            modes=modes,
            campaign_values=campaign_values,
            bid_values=bid_values,
            feasible_sets=feasible_sets,
            clip_low=0.0,
            clip_high=1.0,
        )

        # Per-round mode schedule shared by every realization of this setting.
        phase_ids = build_phase_ids(
            change_points=config["change_points"],
            mode_ids_by_window=config["mode_ids_by_window"],
            horizon=horizon,
        )

        if detector_type == "nd_full_feedback":
            detector_desc = (
                f"nd_full_feedback (window={nd_window_size}, threshold={nd_threshold:g}, "
                f"reset=global [forced]); *** FULL FEEDBACK: reads competing bids, "
                f"NOT directly comparable to bandit SW / CUSUM ***"
            )
        else:
            detector_desc = (
                f"cusum (per-arm, bandit); U_T={U_T} -> M={cusum_M}, "
                f"h={cusum_threshold:.4f}, alpha={alpha:.4f}; epsilon={cusum_epsilon:g}, "
                f"reset={reset_mode}"
            )

        print(
            f"\n[{setting_id}] freq={frequency_label} skl={skl_label} "
            f"(SKL={target_skl:g}), {number_of_changes} change points, "
            f"{len(modes)} modes."
        )
        print(
            f"  N={number_of_campaigns}, budget={budget} (rho={budget / horizon:.3f}), "
            f"horizon={horizon}, ec={exploration_coefficient}, "
            f"{len(feasible_sets)} feasible sets, {candidate_count} candidate super-arms, "
            f"{len(streams)} realizations."
        )
        print(f"  detector: {detector_desc}")

        setting_rows: List[Dict[str, Any]] = []
        for realization, stream in enumerate(streams):
            stream = np.asarray(stream, dtype=float)
            if stream.shape != (horizon, number_of_campaigns):
                raise ValueError(
                    f"{setting_id} realization {realization}: expected "
                    f"{(horizon, number_of_campaigns)}, got {stream.shape}."
                )
            agent_seed = _AGENT_SEED_BASE + realization
            rows = run_single_realization(
                stream=stream,
                phase_ids=phase_ids,
                number_of_campaigns=number_of_campaigns,
                campaign_values=campaign_values,
                bid_values=bid_values,
                budget=budget,
                horizon=horizon,
                conflict_edges=conflict_edges,
                benchmark=benchmark,
                e_reward=e_reward,
                detector_type=detector_type,
                reset_mode=reset_mode,
                cusum_M=cusum_M,
                cusum_epsilon=cusum_epsilon,
                cusum_threshold=cusum_threshold,
                alpha=alpha,
                exploration_coefficient=exploration_coefficient,
                nd_window_size=nd_window_size,
                nd_threshold=nd_threshold,
                agent_seed=agent_seed,
                setting_id=setting_id,
                frequency_label=frequency_label,
                skl_label=skl_label,
                target_skl=target_skl,
                realization=realization,
            )
            setting_rows.extend(rows)
            all_rows.extend(rows)

            final = rows[-1]
            summary.append(
                {
                    "setting_id": setting_id,
                    "frequency_label": frequency_label,
                    "skl_label": skl_label,
                    "skl": target_skl,
                    "realization": realization,
                    "final_cumulative_regret": float(final["cumulative_regret"]),
                    "final_cumulative_reward": float(final["cumulative_reward"]),
                    "final_cumulative_cost": float(final["cumulative_cost"]),
                    "final_detected_changes": int(final["detected_changes"]),
                    "done_round": int(final["round"]),
                }
            )

        # Per-setting plots into the matching req4_comparison subfolder, kept in a
        # per-detector-case sub-subfolder so different detector configs never
        # overwrite.
        setting_df = pd.DataFrame(setting_rows)
        plots_dir = PLOTS_BASE / setting_id / detector_case
        plot_specs = [
            ("cumulative_regret", "cumulative_regret",
             "Cumulative pseudo-regret", "Cumulative pseudo-regret"),
            ("cumulative_reward", "cumulative_reward",
             "Cumulative reward", "Cumulative reward"),
            ("cumulative_cost", "cumulative_cost",
             "Cumulative cost", "Cumulative cost"),
            ("detected_changes", "detected_changes",
             "Detected changes (cumulative)", "Detected changes over time"),
        ]
        for metric, filename, ylabel, title_metric in plot_specs:
            plot_metric_for_setting(
                setting_df=setting_df,
                metric=metric,
                horizon=horizon,
                output_path=plots_dir / f"{setting_id}_{filename}.png",
                ylabel=ylabel,
                title=f"CD-CUCB ({detector_case}, replay) - {setting_id} - {title_metric}",
            )

        # Detection-delay graph (per setting) into the same
        # grouped folder, plus a combined long-format delays CSV.
        delays_df = compute_detection_delays_for_setting(setting_df)
        delays_df.insert(0, "setting_id", setting_id)
        delays_df.insert(1, "frequency_label", frequency_label)
        delays_df.insert(2, "skl_label", skl_label)
        delays_df.insert(3, "skl", target_skl)
        all_delays.append(delays_df)
        plot_detection_delay_for_setting(
            delays_df=delays_df,
            output_path=plots_dir / f"{setting_id}_detection_delay.png",
            title=f"CD-CUCB ({detector_case}, replay) - {setting_id} - detection delay",
        )

        detected_delays = delays_df[delays_df["detected"]]
        delay_rate = float(delays_df["detected"].mean()) * 100.0 if len(delays_df) else 0.0
        delay_mean = float(detected_delays["delay"].mean()) if not detected_delays.empty else float("nan")

        # Per-setting summary line.
        setting_summary = pd.DataFrame(
            [s for s in summary if s["setting_id"] == setting_id]
        )
        print(
            f"  => regret={setting_summary['final_cumulative_regret'].mean():.3f} "
            f"+/- {setting_summary['final_cumulative_regret'].std():.3f}, "
            f"reward={setting_summary['final_cumulative_reward'].mean():.3f} "
            f"+/- {setting_summary['final_cumulative_reward'].std():.3f}, "
            f"detected={setting_summary['final_detected_changes'].mean():.2f} "
            f"(true={number_of_changes}), "
            f"delay_mean={delay_mean:.1f} @ rate={delay_rate:.0f}%"
        )

    # Save combined raw CSV in the official format.
    RAW_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_rows, str(RAW_RESULTS_FILE))
    print(f"\nSaved {len(all_rows)} rows to {RAW_RESULTS_FILE.relative_to(PROJECT_ROOT)}")

    # Save the combined detection-delay records (one row per change point x realization).
    if all_delays:
        delays_all = pd.concat(all_delays, ignore_index=True)
        delays_all.to_csv(DELAYS_RESULTS_FILE, index=False)
        print(
            f"Saved {len(delays_all)} delay records to "
            f"{DELAYS_RESULTS_FILE.relative_to(PROJECT_ROOT)}"
        )

    # Final cross-setting summary table.
    summary_df = pd.DataFrame(summary)
    print("\nFinal cumulative pseudo-regret (mean +/- std over realizations):")
    for setting_id in summary_df["setting_id"].unique():
        sub = summary_df[summary_df["setting_id"] == setting_id]
        print(
            f"  {setting_id:<28}: "
            f"regret={sub['final_cumulative_regret'].mean():8.3f} "
            f"+/- {sub['final_cumulative_regret'].std():7.3f}, "
            f"reward={sub['final_cumulative_reward'].mean():8.3f}, "
            f"detected={sub['final_detected_changes'].mean():5.2f}"
        )

    print("\nReq 4 change-detector shared-realizations experiment completed.")


if __name__ == "__main__":
    main()