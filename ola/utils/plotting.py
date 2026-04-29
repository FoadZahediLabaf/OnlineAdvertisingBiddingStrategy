"""
Plotting utilities for experiment results.

Owner:
    Shared utility file.

Purpose:
    Provide reusable plotting functions for CSV result files.

Do not:
    - Run experiments here.
    - Modify CSV result files here.
    - Add requirement-specific algorithm logic here.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PLOTS_DIR = Path("results") / "plots"


def load_result_csvs(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Load and combine multiple result CSV files.

    Args:
        csv_paths: Paths to CSV files with the shared experiment result format.

    Returns:
        A single dataframe containing all rows from the provided CSV files.
    """
    dataframes = []

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing result CSV: {csv_path}")
        dataframes.append(pd.read_csv(csv_path))

    if not dataframes:
        raise ValueError("At least one CSV path is required.")

    return pd.concat(dataframes, ignore_index=True)


def plot_metric_by_algorithm(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Plot one metric over rounds, grouped by algorithm.

    Args:
        dataframe: Experiment results containing at least algorithm, round, and metric.
        metric: Column name to plot on the y-axis.
        output_path: Path where the PNG plot will be saved.
        ylabel: Optional y-axis label. Defaults to the metric name.
        title: Optional plot title. Defaults to a Req-independent title.
    """
    required_columns = {"algorithm", "round", metric}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for plotting: {sorted(missing_columns)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for algorithm_name, group in dataframe.groupby("algorithm"):
        sorted_group = group.sort_values("round")
        plt.plot(sorted_group["round"], sorted_group[metric], label=algorithm_name)

    plt.xlabel("Round")
    plt.ylabel(ylabel or metric)
    plt.title(title or f"{metric} by algorithm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot to {output_path}")


def generate_standard_req1_plots(
    dataframe: pd.DataFrame,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
) -> None:
    """Generate the standard Requirement 1 plots from a combined dataframe."""
    plots_dir = Path(plots_dir)

    plot_metric_by_algorithm(
        dataframe=dataframe,
        metric="cumulative_reward",
        ylabel="Cumulative reward",
        title="Requirement 1 - Cumulative reward",
        output_path=plots_dir / "req1_cumulative_reward.png",
    )

    plot_metric_by_algorithm(
        dataframe=dataframe,
        metric="budget_left",
        ylabel="Budget left",
        title="Requirement 1 - Budget left",
        output_path=plots_dir / "req1_budget_left.png",
    )

    plot_metric_by_algorithm(
        dataframe=dataframe,
        metric="cumulative_cost",
        ylabel="Cumulative cost",
        title="Requirement 1 - Cumulative cost",
        output_path=plots_dir / "req1_cumulative_cost.png",
    )

    plot_metric_by_algorithm(
        dataframe=dataframe,
        metric="action",
        ylabel="Selected bid",
        title="Requirement 1 - Selected bid",
        output_path=plots_dir / "req1_selected_bid.png",
    )
