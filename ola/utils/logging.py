"""
Logging and result-saving utilities for the OLA project.

Owner:
    Shared utility file

Purpose:
    Keep the CSV result format consistent across all requirements.

Do not:
    - Change REQUIRED_RESULT_COLUMNS after the interface freeze without group approval.
    - Remove required CSV columns.
    - Change the meaning of existing columns.
    - Add experiment-specific logic here.
"""

from typing import Dict, List

import pandas as pd


REQUIRED_RESULT_COLUMNS: List[str] = [
    "algorithm",
    "round",
    "action",
    "bids",
    "wins",
    "reward",
    "cost",
    "budget_left",
    "cumulative_reward",
    "cumulative_cost",
    "benchmark_reward",
    "instant_regret",
    "cumulative_regret",
]


def empty_result_row() -> Dict[str, object]:
    """Create an empty result row with the required CSV columns."""
    return {column: None for column in REQUIRED_RESULT_COLUMNS}


def save_results(rows: List[Dict[str, object]], output_path: str) -> None:
    """Save experiment rows using the standard CSV column order.

    Required columns are written first. Any extra columns present in the rows are
    preserved after the required columns.
    """
    dataframe = pd.DataFrame(rows)
    extra_columns = [
        column for column in dataframe.columns if column not in REQUIRED_RESULT_COLUMNS
    ]
    dataframe = dataframe.reindex(columns=REQUIRED_RESULT_COLUMNS + extra_columns)
    dataframe.to_csv(output_path, index=False)
