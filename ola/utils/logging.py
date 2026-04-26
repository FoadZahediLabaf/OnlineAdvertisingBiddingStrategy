"""
Logging and result-saving utilities for the OLA project.

Owner:
    Foad

Purpose:
    Keep the CSV result format consistent across all requirements.

Do not:
    - Change REQUIRED_RESULT_COLUMNS without group approval.
    - Remove required CSV columns.
    - Change the meaning of existing columns.
    - Add experiment-specific logic here.
"""

from typing import Dict, List

import pandas as pd


REQUIRED_RESULT_COLUMNS: List[str] = [
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
    """Save experiment rows using the standard CSV column order."""
    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.reindex(columns=REQUIRED_RESULT_COLUMNS)
    dataframe.to_csv(output_path, index=False)
