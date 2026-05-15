

import numpy as np

from ola.utils.constellation import SyntheticDataset


def test_synthetic_dataset_imports_from_package():
    assert SyntheticDataset is not None


def test_synthetic_dataset_generates_stream_with_expected_shape():
    dataset = SyntheticDataset(
        n_features=3,
        n_traces=1,
        len_seq=50,
        in_control_modes=2,
        skl_incontrol=(0.1, 0.5),
        transition_mode="instant",
        train_change_freq=0.1,
        constellation_seed=123,
        ondemand=True,
    )

    stream, change_points = dataset.read_datastream("smoke")

    assert stream.shape == (50, 3)
    assert isinstance(change_points, list)
    assert np.isfinite(stream.to_numpy()).all()


def test_synthetic_dataset_values_can_be_clipped_to_bid_domain():
    dataset = SyntheticDataset(
        n_features=2,
        n_traces=1,
        len_seq=25,
        in_control_modes=2,
        skl_incontrol=(0.1, 0.5),
        transition_mode="instant",
        train_change_freq=0.1,
        constellation_seed=321,
        ondemand=True,
    )

    stream, _ = dataset.read_datastream("bid-domain-smoke")
    clipped = np.clip(stream.to_numpy(), 0.0, 1.0)

    assert clipped.shape == (25, 2)
    assert clipped.min() >= 0.0
    assert clipped.max() <= 1.0