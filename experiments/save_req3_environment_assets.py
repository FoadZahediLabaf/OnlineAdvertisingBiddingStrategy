"""
Save reproducible Req 3 Gaussian-Constellation environment assets.

This script stores the final synthetic environments used for the Requirement 3
diagnostics:

- the Gaussian constellation parameters, i.e. one ``mu`` and ``Sigma`` per mode;
- the fixed regime windows and mode sequence;
- repeated generated bid time series for each setting.

The saved files let teammates reuse exactly the same environments without
depending on random regeneration.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.utils.constellation import (  # noqa: E402
    SyntheticDataset,
    symmetric_kullback_leibler,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req3_environment_assets"

T = 1000
N = 5
N_MODES = 5
N_REALIZATIONS = 3
ANCHOR_MEAN = 0.5
ANCHOR_STD = 0.10
CLIP_LO = 0.001
CLIP_HI = 0.999


SETTINGS = [
    {
        "setting_id": "skl_low_freq_1_100",
        "description": "low sKL diagnostics, change frequency 1/100",
        "skl_range": (0.1, 0.5),
        "change_frequency": 1 / 100,
        "oracle_block_length": 100,
        "constellation_seed": 1200,
        "schedule_seed": 12_345,
        "sample_seeds": [51_200, 51_201, 51_202],
    },
    {
        "setting_id": "skl_medium_freq_1_100",
        "description": "medium sKL diagnostics, change frequency 1/100",
        "skl_range": (1.5, 3.0),
        "change_frequency": 1 / 100,
        "oracle_block_length": 100,
        "constellation_seed": 1300,
        "schedule_seed": 12_345,
        "sample_seeds": [52_200, 52_201, 52_202],
    },
    {
        "setting_id": "skl_high_freq_1_100",
        "description": "high sKL diagnostics, change frequency 1/100",
        "skl_range": (10.0, 15.0),
        "change_frequency": 1 / 100,
        "oracle_block_length": 100,
        "constellation_seed": 1400,
        "schedule_seed": 12_345,
        "sample_seeds": [53_200, 53_201, 53_202],
    },
    {
        "setting_id": "skl_medium_freq_1_200",
        "description": "medium sKL frequency diagnostics, slow change frequency 1/200",
        "skl_range": (1.5, 3.0),
        "change_frequency": 1 / 200,
        "oracle_block_length": 200,
        "constellation_seed": 1300,
        "schedule_seed": 22_000,
        "sample_seeds": [22_500, 22_501, 22_502],
    },
    {
        "setting_id": "skl_medium_freq_1_50",
        "description": "medium sKL frequency diagnostics, fast change frequency 1/50",
        "skl_range": (1.5, 3.0),
        "change_frequency": 1 / 50,
        "oracle_block_length": 50,
        "constellation_seed": 1300,
        "schedule_seed": 24_000,
        "sample_seeds": [24_500, 24_501, 24_502],
    },
]


def low_variance_anchor() -> tuple[np.ndarray, np.ndarray]:
    """Return the low-variance first constellation star used in Req 3."""

    mu = np.full(N, ANCHOR_MEAN)
    sigma = (ANCHOR_STD**2) * np.eye(N)
    return mu, sigma


def build_constellation(skl_range: tuple[float, float], seed: int):
    """Build a five-mode low-variance constellation in the target sKL range."""

    dataset = SyntheticDataset(
        n_features=N,
        n_traces=1,
        len_seq=20,
        skl_incontrol=skl_range,
        in_control_modes=N_MODES,
        transition_mode="instant",
        train_change_freq=0.0,
        constellation_method="rotation_range",
        constellation_max_iter=8000,
        constellation_lr=0.01,
        constellation_patience=500,
        constellation_n_restarts=3,
        constellation_seed=seed,
        phi_0=low_variance_anchor(),
        max_std=ANCHOR_STD,
        ondemand=True,
    )
    if len(dataset.constellation) != N_MODES:
        raise RuntimeError(
            f"Expected {N_MODES} modes for sKL range {skl_range}; "
            f"got {len(dataset.constellation)}."
        )
    return dataset.constellation


def pairwise_skl_values(constellation) -> list[float]:
    values = []
    for i in range(len(constellation)):
        for j in range(i + 1, len(constellation)):
            values.append(float(symmetric_kullback_leibler(constellation[i], constellation[j])))
    return values


def make_fixed_schedule(
    *,
    horizon: int,
    n_modes: int,
    change_frequency: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Generate fixed change points and the active mode per window."""

    rng = np.random.default_rng(seed)
    change_points = [t for t in range(1, horizon) if rng.random() < change_frequency]
    mode_ids = [0]
    current = 0
    for _ in change_points:
        choices = [idx for idx in range(n_modes) if idx != current]
        current = int(rng.choice(choices))
        mode_ids.append(current)
    return change_points, mode_ids


def iter_windows(change_points: list[int], mode_ids: list[int], horizon: int) -> Iterable[dict]:
    boundaries = [0] + change_points + [horizon]
    for window_id, (start, end, mode_id) in enumerate(
        zip(boundaries[:-1], boundaries[1:], mode_ids)
    ):
        yield {
            "window_id": window_id,
            "start_round": start,
            "end_round_exclusive": end,
            "length": end - start,
            "mode_id": mode_id,
        }


def sample_stream(
    constellation,
    change_points: list[int],
    mode_ids: list[int],
    sample_seed: int,
) -> np.ndarray:
    """Sample one bid time series with fixed windows and constellation."""

    rng = np.random.default_rng(sample_seed)
    boundaries = [0] + change_points + [T]
    rows = []
    for start, end, mode_id in zip(boundaries[:-1], boundaries[1:], mode_ids):
        mu, sigma = constellation[mode_id]
        samples = rng.multivariate_normal(mu, sigma, end - start)
        rows.append(np.clip(samples, CLIP_LO, CLIP_HI))
    return np.vstack(rows)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_setting_assets(setting: dict) -> dict:
    setting_dir = OUTPUT_DIR / setting["setting_id"]
    setting_dir.mkdir(parents=True, exist_ok=True)

    constellation = build_constellation(setting["skl_range"], setting["constellation_seed"])
    change_points, mode_ids = make_fixed_schedule(
        horizon=T,
        n_modes=N_MODES,
        change_frequency=setting["change_frequency"],
        seed=setting["schedule_seed"],
    )

    mus = np.stack([mode[0] for mode in constellation])
    sigmas = np.stack([mode[1] for mode in constellation])
    np.savez_compressed(
        setting_dir / "constellation.npz",
        mus=mus,
        sigmas=sigmas,
        skl_values=np.array(pairwise_skl_values(constellation)),
    )

    write_csv(
        setting_dir / "windows.csv",
        ["window_id", "start_round", "end_round_exclusive", "length", "mode_id"],
        iter_windows(change_points, mode_ids, T),
    )

    stream_rows = []
    stream_arrays = []
    for realization_index, sample_seed in enumerate(setting["sample_seeds"], start=1):
        stream = sample_stream(constellation, change_points, mode_ids, sample_seed)
        stream_arrays.append(stream)
        for round_index, bids in enumerate(stream):
            stream_rows.append(
                {
                    "realization": realization_index,
                    "sample_seed": sample_seed,
                    "round": round_index,
                    **{f"campaign_{campaign_id}_bid": float(bid) for campaign_id, bid in enumerate(bids)},
                }
            )

    np.savez_compressed(
        setting_dir / "streams.npz",
        streams=np.stack(stream_arrays),
        sample_seeds=np.array(setting["sample_seeds"], dtype=int),
    )
    write_csv(
        setting_dir / "streams.csv",
        [
            "realization",
            "sample_seed",
            "round",
            "campaign_0_bid",
            "campaign_1_bid",
            "campaign_2_bid",
            "campaign_3_bid",
            "campaign_4_bid",
        ],
        stream_rows,
    )

    config = {
        **setting,
        "skl_range": list(setting["skl_range"]),
        "horizon": T,
        "number_of_campaigns": N,
        "number_of_modes": N_MODES,
        "number_of_realizations": N_REALIZATIONS,
        "anchor_mean": ANCHOR_MEAN,
        "anchor_std": ANCHOR_STD,
        "anchor_variance": ANCHOR_STD**2,
        "clip_bounds": [CLIP_LO, CLIP_HI],
        "constellation_method": "rotation_range",
        "change_points": change_points,
        "mode_ids_by_window": mode_ids,
    }
    with (setting_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    skl_values = pairwise_skl_values(constellation)
    return {
        "setting_id": setting["setting_id"],
        "description": setting["description"],
        "skl_low": setting["skl_range"][0],
        "skl_high": setting["skl_range"][1],
        "change_frequency": setting["change_frequency"],
        "oracle_block_length": setting["oracle_block_length"],
        "constellation_seed": setting["constellation_seed"],
        "schedule_seed": setting["schedule_seed"],
        "sample_seeds": json.dumps(setting["sample_seeds"]),
        "n_change_points": len(change_points),
        "n_windows": len(mode_ids),
        "min_pairwise_skl": min(skl_values),
        "max_pairwise_skl": max(skl_values),
        "mean_pairwise_skl": float(np.mean(skl_values)),
    }


def write_index(summary_rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "index.csv",
        [
            "setting_id",
            "description",
            "skl_low",
            "skl_high",
            "change_frequency",
            "oracle_block_length",
            "constellation_seed",
            "schedule_seed",
            "sample_seeds",
            "n_change_points",
            "n_windows",
            "min_pairwise_skl",
            "max_pairwise_skl",
            "mean_pairwise_skl",
        ],
        summary_rows,
    )

    readme = """# Requirement 3 saved Gaussian-Constellation assets

These files save the final environment assets used for the Requirement 3
diagnostic experiments.

Each setting folder contains:

- `config.json`: all seeds, sKL range, change frequency, anchor, and schedule.
- `constellation.npz`: Gaussian mode means (`mus`) and covariances (`sigmas`).
- `windows.csv`: fixed regime windows and active mode per window.
- `streams.csv`: repeated generated competing-bid time series.
- `streams.npz`: compact NumPy copy of all repeated streams.

The first constellation star is fixed to `mu = [0.5] * 5` and
`Sigma = 0.01 I`, so all modes keep low variance while respecting the target
pairwise sKL range.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    summary_rows = [save_setting_assets(setting) for setting in SETTINGS]
    write_index(summary_rows)
    print(f"Saved Req 3 environment assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
