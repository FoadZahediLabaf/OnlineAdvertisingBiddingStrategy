"""
Experiment runner for Requirement 4 (sliding-window CUCB) on the PRE-GENERATED
shared realizations.

Owner:
    Alessandro

Support:
    Simone may edit this file when actively helping Alessandro.

Purpose:
    Run the SLIDING-WINDOW Combinatorial-UCB on the saved Requirement 4 shared
    realizations (results/processed/req4_shared_realizations), in REPLAY mode.

Do not:
    - Add Requirement 1, Requirement 2, or Requirement 3 experiment logic here.
    - Regenerate the shared realizations here (that is
      experiments/generate_req4_shared_realizations.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from math import log, sqrt
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
from ola.algorithms.sw_cucb import SlidingWindowCUCB
from ola.environments.non_stationary import NonStationaryMultiCampaignEnvironment
from ola.utils.logging import save_results


INPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req4_shared_realizations"
PLOTS_BASE = PROJECT_ROOT / "results" / "plots" / "req4_comparison"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
RAW_RESULTS_FILE = RAW_DIR / "req4_sw_on_shared.csv"

ALGORITHM_NAME = "sw_cucb"
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


# PER-FREQUENCY (THEORY) WINDOW SIZE.
def theory_window_size(horizon: int, number_of_changes: int, reward_range: float) -> int:
    """Sliding-window-UCB window from the standard formula, per setting.
        W = floor( 2 * B * sqrt( T * log(T) / Upsilon_T ) )
    with B the reward range and Upsilon_T the (known) number of distribution
    changes over the horizon. Here Upsilon_T is the setting's number of change
    points"""
    upsilon = max(1, int(number_of_changes))
    window = int(2.0 * float(reward_range) * sqrt(horizon * log(horizon) / upsilon))
    return max(1, window)


# ANALYTIC ORACLE (same methodology as run_req4.py).
def compute_mode_expectations(
    modes: Sequence[Tuple[np.ndarray, np.ndarray]],
    campaign_values: Sequence[float],
    bid_values: Sequence[float],
    feasible_sets: Sequence[Tuple[int, ...]],
    clip_low: float,
    clip_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute, for each recurring MODE, the per-(campaign, bid) expected reward and
    the benchmark (best feasible super-arm) expected per-round reward.
    The expectation is taken analytically over that mode's true Gaussian.
    Returns:
        benchmark : (number_of_modes) best feasible expected per-round reward
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


# SINGLE REALIZATION: replay one saved stream through SW-CUCB.
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
    window_size: int,
    exploration_coefficient: float,
    agent_seed: int,
    setting_id: str,
    frequency_label: str,
    skl_label: str,
    target_skl: float,
    realization: int,
) -> List[Dict[str, Any]]:
    """Replay one saved competing-bid stream through SW-CUCB and return CSV rows."""
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

        # phase_id returned by the env is the active MODE id (0..number_of_modes-1).
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


# MAIN
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Requirement 4 sliding-window CUCB on the saved shared realizations."
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Run only this setting_id (e.g. high_frequency_high_skl). Default: all 9.",
    )
    parser.add_argument(
        "--window-rule",
        type=str,
        choices=["fixed", "theory"],
        default="fixed",
        help="How to set the window per setting: 'fixed' -> round(sqrt(horizon)) "
             "(same for all settings); 'theory' -> floor(2*B*sqrt(T*logT/U_T)) with "
             "U_T = that setting's number of change points (per-frequency window). "
             "An explicit --window-size overrides this for all settings.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override the sliding window size (rounds) for ALL settings. Takes "
             "precedence over --window-rule. Default: use --window-rule.",
    )
    parser.add_argument(
        "--exploration-coefficient",
        type=float,
        default=0.1,
        help="SW exploration coefficient. Default: 0.1 (matches the main SW experiment).",
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

    all_rows: List[Dict[str, Any]] = []
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

        # window_size: explicit --window-size wins for all settings; otherwise the
        # chosen rule ('fixed' round(sqrt(T)), or per-frequency 'theory').
        if args.window_size is not None:
            window_size = int(args.window_size)
            window_desc = f"window_size={window_size} (override)"
            # Explicit override: label by value only.
            window_case = f"window_w{window_size}"
        elif args.window_rule == "theory":
            window_size = theory_window_size(
                horizon=horizon,
                number_of_changes=number_of_changes,
                reward_range=float(max(campaign_values)),
            )
            window_desc = f"window_size={window_size} (theory, U_T={number_of_changes})"
            window_case = f"theory_w{window_size}"
        else:
            window_size = max(1, round(sqrt(horizon)))
            window_desc = f"window_size={window_size} (fixed sqrt(T))"
            window_case = f"fixed_w{window_size}"
        exploration_coefficient = float(args.exploration_coefficient)

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

        print(
            f"\n[{setting_id}] freq={frequency_label} skl={skl_label} "
            f"(SKL={target_skl:g}), {len(config['change_points'])} change points, "
            f"{len(modes)} modes."
        )
        print(
            f"  N={number_of_campaigns}, budget={budget} (rho={budget / horizon:.3f}), "
            f"horizon={horizon}, {window_desc}, ec={exploration_coefficient}, "
            f"{len(feasible_sets)} feasible sets, {candidate_count} candidate super-arms, "
            f"{len(streams)} realizations."
        )

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
                window_size=window_size,
                exploration_coefficient=exploration_coefficient,
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
                    "done_round": int(final["round"]),
                }
            )

        # Per-setting plots into the matching req4_comparison subfolder, kept in a
        # per-window-case sub-subfolder so different window sizes never overwrite.
        setting_df = pd.DataFrame(setting_rows)
        plots_dir = PLOTS_BASE / setting_id / window_case
        plot_specs = [
            ("cumulative_regret", "cumulative_regret",
             "Cumulative pseudo-regret", "Cumulative pseudo-regret"),
            ("cumulative_reward", "cumulative_reward",
             "Cumulative reward", "Cumulative reward"),
            ("cumulative_cost", "cumulative_cost",
             "Cumulative cost", "Cumulative cost"),
        ]
        for metric, filename, ylabel, title_metric in plot_specs:
            plot_metric_for_setting(
                setting_df=setting_df,
                metric=metric,
                horizon=horizon,
                output_path=plots_dir / f"{setting_id}_{filename}.png",
                ylabel=ylabel,
                title=f"SW-CUCB (replay) - {setting_id} - {title_metric}",
            )

        # Per-setting summary line.
        setting_summary = pd.DataFrame(
            [s for s in summary if s["setting_id"] == setting_id]
        )
        print(
            f"  => regret={setting_summary['final_cumulative_regret'].mean():.3f} "
            f"+/- {setting_summary['final_cumulative_regret'].std():.3f}, "
            f"reward={setting_summary['final_cumulative_reward'].mean():.3f} "
            f"+/- {setting_summary['final_cumulative_reward'].std():.3f}"
        )

    # Save combined raw CSV in the official format.
    RAW_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_rows, str(RAW_RESULTS_FILE))
    print(f"\nSaved {len(all_rows)} rows to {RAW_RESULTS_FILE.relative_to(PROJECT_ROOT)}")

    # Final cross-setting summary table.
    summary_df = pd.DataFrame(summary)
    print("\nFinal cumulative pseudo-regret (mean +/- std over realizations):")
    for setting_id in summary_df["setting_id"].unique():
        sub = summary_df[summary_df["setting_id"] == setting_id]
        print(
            f"  {setting_id:<28}: "
            f"regret={sub['final_cumulative_regret'].mean():8.3f} "
            f"+/- {sub['final_cumulative_regret'].std():7.3f}, "
            f"reward={sub['final_cumulative_reward'].mean():8.3f}"
        )

    print("\nReq 4 sliding-window shared-realizations experiment completed.")


if __name__ == "__main__":
    main()