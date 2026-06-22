"""Generate local Req4-style shared realization matrices for Req3 comparison.

This is a stopgap until Ale pushes the final saved matrices. It uses the
Req4 branch environment generator when available, then stores one ``T x N``
highest-competing-bid matrix per realization.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALE_REPO = Path(os.environ.get("TEMP", "")) / "ola-project-req4"
ALE_REPO = Path(os.environ.get("ALE_REQ4_REPO", DEFAULT_ALE_REPO))
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req4_shared_realizations"

if str(ALE_REPO) not in sys.path:
    sys.path.insert(0, str(ALE_REPO))

try:
    from ola.environments.non_stationary import NonStationaryMultiCampaignEnvironment  # noqa: E402
except ModuleNotFoundError:
    NonStationaryMultiCampaignEnvironment = None


T = 3000
N = 10
BUDGET = 10000
NUM_MODES = 3
MAX_STD = 0.1
CONSTELLATION_SEED = 42
CLIP_LO = 0.0
CLIP_HI = 1.0

SKL_SETTINGS = [
    ("low_skl", 20),
    ("medium_skl", 200),
    ("high_skl", 2000),
]

FREQUENCY_SETTINGS = [
    ("low_frequency", 1 / 100, 100, 22_000),
    ("medium_frequency", 1 / 50, 50, 22_000),
    ("high_frequency", 1 / 25, 25, 22_000),
]

N_REALIZATIONS = 10


def build_modes(target_skl: float):
    if NonStationaryMultiCampaignEnvironment is None:
        skl_label = next(label for label, value in SKL_SETTINGS if value == target_skl)
        saved = np.load(OUTPUT_DIR / f"low_frequency_{skl_label}" / "constellation.npz")
        return list(zip(saved["mus"], saved["sigmas"]))
    env = NonStationaryMultiCampaignEnvironment(
        number_of_campaigns=N,
        horizon=T,
        number_of_intervals=NUM_MODES,
        budget=BUDGET,
        target_skl=target_skl,
        max_std=MAX_STD,
        constellation_seed=CONSTELLATION_SEED,
        random_seed=42,
    )
    return env.constellation


def make_schedule(change_frequency: float, schedule_seed: int):
    rng = np.random.default_rng(schedule_seed)
    change_points = [t for t in range(1, T) if rng.random() < change_frequency]
    mode_ids = [0]
    current = 0
    for _ in change_points:
        choices = [idx for idx in range(NUM_MODES) if idx != current]
        current = int(rng.choice(choices))
        mode_ids.append(current)
    return change_points, mode_ids


def sample_stream(modes, change_points, mode_ids, sample_seed: int):
    rng = np.random.default_rng(sample_seed)
    boundaries = [0] + change_points + [T]
    chunks = []
    for start, end, mode_id in zip(boundaries[:-1], boundaries[1:], mode_ids):
        mu, sigma = modes[mode_id]
        samples = rng.multivariate_normal(mu, sigma, size=end - start)
        chunks.append(np.clip(samples, CLIP_LO, CLIP_HI))
    return np.vstack(chunks)


def save_setting(freq_label, change_frequency, oracle_block_length, schedule_seed, skl_label, target_skl):
    setting_id = f"{freq_label}_{skl_label}"
    setting_dir = OUTPUT_DIR / setting_id
    setting_dir.mkdir(parents=True, exist_ok=True)

    modes = build_modes(target_skl)
    mus = np.stack([mode[0] for mode in modes])
    sigmas = np.stack([mode[1] for mode in modes])
    np.savez_compressed(setting_dir / "constellation.npz", mus=mus, sigmas=sigmas)

    sample_seeds = []
    change_points, mode_ids = make_schedule(change_frequency, schedule_seed)
    for realization_idx in range(N_REALIZATIONS):
        sample_seed = 100_000 + realization_idx
        stream = sample_stream(modes, change_points, mode_ids, sample_seed)
        sample_seeds.append(sample_seed)
        np.save(setting_dir / f"realization_{realization_idx:02d}.npy", stream)

    config = {
        "setting_id": setting_id,
        "horizon": T,
        "number_of_campaigns": N,
        "budget": BUDGET,
        "bid_grid": [round(x, 1) for x in np.linspace(0.0, 1.0, 11)],
        "target_skl": target_skl,
        "skl_label": skl_label,
        "frequency_label": freq_label,
        "change_frequency": change_frequency,
        "oracle_block_length": oracle_block_length,
        "number_of_realizations": N_REALIZATIONS,
        "number_of_modes": NUM_MODES,
        "max_std": MAX_STD,
        "constellation_seed": CONSTELLATION_SEED,
        "schedule_seed": schedule_seed,
        "sample_seeds": sample_seeds,
        "change_points": change_points,
        "mode_ids_by_window": mode_ids,
    }
    with (setting_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    return {
        "setting_id": setting_id,
        "frequency_label": freq_label,
        "skl_label": skl_label,
        "target_skl": target_skl,
        "change_frequency": change_frequency,
        "oracle_block_length": oracle_block_length,
        "schedule_seed": schedule_seed,
        "n_change_points": len(change_points),
        "matrix_shape": f"{T}x{N}",
        "n_realizations": N_REALIZATIONS,
    }


def main():
    saved_constellations = [
        OUTPUT_DIR / f"low_frequency_{skl_label}" / "constellation.npz"
        for skl_label, _ in SKL_SETTINGS
    ]
    if NonStationaryMultiCampaignEnvironment is None and not all(path.exists() for path in saved_constellations):
        raise FileNotFoundError(
            f"Ale Req4 repo not found at {ALE_REPO}, and saved constellations are incomplete."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for freq_label, change_frequency, oracle_block_length, schedule_seed in FREQUENCY_SETTINGS:
        for skl_label, target_skl in SKL_SETTINGS:
            row = save_setting(
                freq_label,
                change_frequency,
                oracle_block_length,
                schedule_seed,
                skl_label,
                target_skl,
            )
            rows.append(row)
            print(f"saved {row['setting_id']} with {row['n_change_points']} changes")

    with (OUTPUT_DIR / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved local shared realizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
