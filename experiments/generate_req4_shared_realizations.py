"""Generate local Req4-style shared realization matrices for Req3 comparison.

This is a stopgap until Ale pushes the final saved matrices. It uses the
Req4 branch environment generator when available, then stores one ``T x N``
highest-competing-bid matrix per realization.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "req4_shared_realizations"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ola.utils.constellation.constellation import generate_constellation_gd_rotation  # noqa: E402


T = 3000
N = 5
BUDGET = 2500
NUM_MODES = 3
MAX_STD = 0.1
CONSTELLATION_SEED = 42
CLIP_LO = 0.0
CLIP_HI = 1.0
CONFLICT_EDGES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0],
]

SKL_SETTINGS = [
    ("low_skl", 20),
    ("medium_skl", 200),
    ("high_skl", 500),
]

FREQUENCY_SETTINGS = [
    ("low_frequency", 1 / 1500, 1500, 106),
    ("medium_frequency", 1 / 50, 50, 22_000),
    ("high_frequency", 1 / 25, 25, 22_000),
]

N_REALIZATIONS = 10


def build_modes(target_skl: float):
    result = generate_constellation_gd_rotation(
        dim=N,
        n_modes=NUM_MODES,
        target_skl=target_skl,
        max_iter=5000,
        max_std=MAX_STD,
        seed=CONSTELLATION_SEED,
    )
    return result["constellation"]


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


def phase_ids_by_round(change_points: list[int], mode_ids: list[int]) -> np.ndarray:
    phase_ids = np.empty(T, dtype=int)
    boundaries = [0] + change_points + [T]
    for start, end, mode_id in zip(boundaries[:-1], boundaries[1:], mode_ids):
        phase_ids[start:end] = mode_id
    return phase_ids


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
    np.savez_compressed(
        setting_dir / "constellation.npz",
        mus=mus,
        sigmas=sigmas,
        means=mus,
        covariances=sigmas,
    )

    streams = []
    sample_seeds = []
    change_points, mode_ids = make_schedule(change_frequency, schedule_seed)
    phase_ids = phase_ids_by_round(change_points, mode_ids)
    for realization_idx in range(N_REALIZATIONS):
        sample_seed = 100_000 + realization_idx
        stream = sample_stream(modes, change_points, mode_ids, sample_seed)
        streams.append(stream)
        sample_seeds.append(sample_seed)
        np.save(setting_dir / f"realization_{realization_idx:02d}.npy", stream)

    np.savez_compressed(
        setting_dir / "streams.npz",
        streams=np.stack(streams),
        sample_seeds=np.array(sample_seeds, dtype=int),
        phase_ids=phase_ids,
        change_points=np.array(change_points, dtype=int),
        mode_ids_by_window=np.array(mode_ids, dtype=int),
        means=mus,
        covariances=sigmas,
        mus=mus,
        sigmas=sigmas,
    )

    config = {
        "setting_id": setting_id,
        "horizon": T,
        "number_of_campaigns": N,
        "budget": BUDGET,
        "conflict_edges": CONFLICT_EDGES,
        "allowed_campaign_sets": [
            [0],
            [1],
            [2],
            [3],
            [4],
            [0, 2],
            [0, 3],
            [1, 3],
            [1, 4],
            [2, 4],
        ],
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
