"""
ccm_nova.synthetic.handler
=====================
Generates synthetic multivariate time-series data for the multi-campaign
bidding-auction setting.

High-level idea
---------------
The data stream is the per-step maximum bid across ``n_features`` campaigns,
modelled as a low-variance multivariate Gaussian whose mean lives in (0, 1).
The environment changes slowly: every once in a while the active Gaussian
switches to another mode of a pre-computed *constellation* in which every
pair has symmetric KL divergence (sKL) within ``skl_incontrol``.

There is no "out-of-control" regime in this environment — every regime is
in-control, the data simply changes between equally-valid modes.

Three transition styles are available between consecutive modes
(``transition_mode``):

1. ``'instant'``     — abrupt switch (default).
2. ``'incremental'`` — over ``n_drift_samples`` steps, every sample is drawn
   from a Gaussian that linearly interpolates between the two modes
   (parameter-space interpolation of mean and covariance).
3. ``'gradual'``     — over ``n_drift_samples`` steps split into
   ``n_gradual_blocks`` blocks, each block draws from one of the two
   endpoints with probability given by the mixing-weight ramp.

Generated samples are clipped component-wise to ``[clip_lo, clip_hi]``
(default ``[0.001, 0.999]``) so that values stay within (0, 1) even when a
mean of 0.9 plus noise would otherwise overshoot.
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ola.utils.constellation import utils as ccm_utils
from ola.utils.constellation.constellation import (
    MU_HI,
    MU_LO,
    generate_constellation_gd,
    generate_constellation_gd_range,
    generate_constellation_gd_rotation,
    generate_constellation_gd_rotation_range,
    generate_constellation_rej,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Helpers ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_gaussian(g_a, g_b, alpha):
    """Linearly interpolate two Gaussians in (mu, sigma) parameter space.

    ``alpha=0`` returns ``g_a``; ``alpha=1`` returns ``g_b``.  A convex
    combination of two PSD matrices is PSD, so the interpolated covariance
    is always a valid covariance matrix.
    """
    mu_a, sigma_a = g_a
    mu_b, sigma_b = g_b
    mu = (1.0 - alpha) * np.asarray(mu_a) + alpha * np.asarray(mu_b)
    sigma = (1.0 - alpha) * np.asarray(sigma_a) + alpha * np.asarray(sigma_b)
    return mu, sigma


def _sample_clipped(gauss, n, clip_lo, clip_hi):
    """Draw ``n`` samples from a Gaussian and clip them to ``[clip_lo, clip_hi]``."""
    samples = np.random.multivariate_normal(gauss[0], gauss[1], n)
    return np.clip(samples, clip_lo, clip_hi)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main class ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticDataset:
    """Generates synthetic multivariate time-series for the bidding-auction setting.

    Parameters
    ----------
    n_features : int
        Number of campaigns / dimension of the multivariate Gaussian.
    n_traces : int
        How many independent sequences to generate.
    len_seq : int
        Total length of each sequence (number of time steps).
    skl_incontrol : tuple of (float, float)
        Pairwise sKL constraint for every pair of modes in the constellation.
        Use ``(v, v)`` to fix it.  When ``in_control_modes <= 1`` this is
        ignored (no transitions can happen).
    transition_mode : str
        How regime changes are produced when the active mode switches:
        - ``'instant'``     — abrupt change (default).
        - ``'incremental'`` — linear (mu, sigma) interpolation over
          ``n_drift_samples`` steps.
        - ``'gradual'``     — block-wise probabilistic mixing across
          ``n_gradual_blocks`` blocks within the ``n_drift_samples`` window.
    train_change_freq : float
        Per-step probability of switching the active mode.  Set to 0 for a
        stationary distribution.
    n_drift_samples : int
        Length of the transition window for ``'incremental'``/``'gradual'``.
    drift_type : str
        ``'linear'`` or ``'sigmoidal'`` mixing-weight ramp for ``'gradual'``.
    n_gradual_blocks : int
        Number of equal-length blocks the transition window is split into
        for ``'gradual'``.
    in_control_modes : int
        Number of distinct Gaussians in the constellation.  Must be >= 1.
        If 1, the stream is stationary (no switches).
    constellation : list of (mu, sigma) or None
        Pre-computed constellation.  When provided, ``in_control_modes``
        and ``skl_incontrol`` are not used for generation.
    constellation_method : str
        One of ``'rejection'``, ``'gd'``, ``'gd_range'``,
        ``'rotation'`` (default), ``'rotation_range'``.
    constellation_timeout : float
        Wall-clock seconds budget for ``'rejection'``.
    constellation_max_iter, constellation_lr, constellation_patience,
    constellation_n_restarts, constellation_seed :
        Hyper-parameters forwarded to the GD constellation builders.
    phi_0 : (mu, sigma) or None
        Optional anchor for mode 0 of the constellation.  Mean must be
        component-wise in ``[MU_LO, MU_HI]``.
    max_std : float
        Upper bound on the principal-axis std of randomly-sampled covariances.
        Ignored when ``phi_0`` is provided. Default 0.1 keeps samples well
        inside the [0, 1] auction box.
    clip_lo, clip_hi : float
        Per-component clipping bounds applied to every generated sample.
        Defaults: 0.001 and 0.999.
    ondemand : bool
        If True, generate sequences on the fly.  If False, read CSVs from
        ``sinkpath``.
    isTrain : bool
        Selects ``'traindata/'`` or ``'testdata/'`` sub-directory.
    saveData : bool
        If True and ``ondemand=True``, also write each generated sequence
        to a CSV.
    source_path : str
        Root directory for saved Gaussian parameters and CSVs.
    new_train_gaussians : bool
        For the single-mode (``in_control_modes == 1``, no ``phi_0``) path:
        if True always sample a fresh anchor; if False try to load a
        previously saved one.
    """

    def __init__(
        self,
        n_features: int = 3,
        n_traces: int = 5,
        len_seq: int = 1000,
        skl_incontrol: tuple = (1.0, 1.0),
        transition_mode: str = 'instant',
        train_change_freq: float = 1 / 300,
        n_drift_samples: int = 10,
        drift_type: str = 'linear',
        n_gradual_blocks: int = 10,
        in_control_modes: int = 2,
        constellation: Optional[List] = None,
        constellation_method: str = 'rotation',
        constellation_timeout: float = 60.0,
        constellation_max_iter: int = 5000,
        constellation_lr: float = 0.01,
        constellation_patience: int = 200,
        constellation_n_restarts: int = 1,
        constellation_seed: Optional[int] = None,
        phi_0: Optional[Tuple] = None,
        max_std: float = 0.1,
        clip_lo: float = 0.001,
        clip_hi: float = 0.999,
        ondemand: bool = True,
        isTrain: bool = True,
        saveData: bool = False,
        source_path: str = 'datasets/synthetic/',
        new_train_gaussians: bool = False,
    ):
        if transition_mode not in ('instant', 'incremental', 'gradual'):
            raise ValueError(
                f"Unknown transition_mode '{transition_mode}'. "
                "Choose 'instant', 'incremental', or 'gradual'.")
        if in_control_modes < 1:
            raise ValueError("in_control_modes must be >= 1.")

        assert 0 <= train_change_freq < 1, "train_change_freq must be in [0, 1)"
        assert clip_lo < clip_hi, "clip_lo must be < clip_hi"

        self.n_features = n_features
        self.len_seq = len_seq
        self.skl_incontrol = skl_incontrol
        self.transition_mode = transition_mode
        self.train_change_freq = train_change_freq
        self.n_drift_samples = n_drift_samples
        self.drift_type = drift_type
        self.n_gradual_blocks = n_gradual_blocks
        self.in_control_modes = in_control_modes
        self.max_std = max_std
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.pm_names = [f'Feature {i + 1}' for i in range(n_features)]

        self.ondemand = ondemand
        self.saveData = saveData
        self.isTrain = isTrain
        self.new_train_gaussians = new_train_gaussians
        self.source_path = source_path + f'{n_features}variate/'
        self.sinkpath = self.source_path + ('traindata/' if isTrain else 'testdata/')

        if self.saveData:
            os.makedirs(self.sinkpath, exist_ok=True)

        # ── File list ────────────────────────────────────────────────────
        if self.ondemand:
            self.filenames = [f'Sequence {i + 1}' for i in range(n_traces)]
        elif os.path.isdir(self.sinkpath):
            self.filenames = sorted(
                os.path.join(self.sinkpath, f)
                for f in os.listdir(self.sinkpath)
                if f.endswith('.csv')
            )[:n_traces]
        else:
            self.filenames = []

        self.gename = self.getfilename()

        # ── Build the constellation ──────────────────────────────────────
        if constellation is not None:
            self.constellation = list(constellation)
        elif in_control_modes >= 2:
            _target_skl = (skl_incontrol[0] + skl_incontrol[1]) / 2
            if constellation_method == 'rejection':
                self.constellation = generate_constellation_rej(
                    dim=n_features, n_modes=in_control_modes,
                    skl_range=skl_incontrol, timeout=constellation_timeout,
                    seed=constellation_seed, phi_0=phi_0, max_std=max_std,
                )
            elif constellation_method == 'gd':
                self.constellation = generate_constellation_gd(
                    dim=n_features, n_modes=in_control_modes,
                    target_skl=_target_skl,
                    max_iter=constellation_max_iter, lr=constellation_lr,
                    patience=constellation_patience,
                    n_restarts=constellation_n_restarts,
                    seed=constellation_seed, phi_0=phi_0, max_std=max_std,
                )['constellation']
            elif constellation_method == 'gd_range':
                self.constellation = generate_constellation_gd_range(
                    dim=n_features, n_modes=in_control_modes,
                    skl_range=skl_incontrol,
                    max_iter=constellation_max_iter, lr=constellation_lr,
                    patience=constellation_patience,
                    n_restarts=constellation_n_restarts,
                    seed=constellation_seed, phi_0=phi_0, max_std=max_std,
                )['constellation']
            elif constellation_method == 'rotation':
                self.constellation = generate_constellation_gd_rotation(
                    dim=n_features, n_modes=in_control_modes,
                    target_skl=_target_skl,
                    max_iter=constellation_max_iter, lr=constellation_lr,
                    patience=constellation_patience,
                    n_restarts=constellation_n_restarts,
                    seed=constellation_seed, phi_0=phi_0, max_std=max_std,
                )['constellation']
            elif constellation_method == 'rotation_range':
                self.constellation = generate_constellation_gd_rotation_range(
                    dim=n_features, n_modes=in_control_modes,
                    skl_range=skl_incontrol,
                    max_iter=constellation_max_iter, lr=constellation_lr,
                    patience=constellation_patience,
                    n_restarts=constellation_n_restarts,
                    seed=constellation_seed, phi_0=phi_0, max_std=max_std,
                )['constellation']
            else:
                raise ValueError(
                    f"Unknown constellation_method '{constellation_method}'. "
                    "Choose from: 'rejection', 'gd', 'gd_range', 'rotation', "
                    "'rotation_range'."
                )
        else:
            # Single-mode: stationary stream from one Gaussian.
            self.constellation = [self._build_single_mode(phi_0)]

    # ─────────────────────────────────────────────────────────────────────────
    # ── Single-mode initial Gaussian (only used when in_control_modes == 1) ──
    # ─────────────────────────────────────────────────────────────────────────

    def _build_single_mode(self, phi_0):
        if phi_0 is not None:
            mu, sigma = phi_0
            mu = np.asarray(mu)
            sigma = np.asarray(sigma)
            if not np.all((mu >= MU_LO) & (mu <= MU_HI)):
                raise ValueError(
                    f"phi_0 mean must be component-wise in [{MU_LO}, {MU_HI}].")
            return (mu, sigma)

        if not self.new_train_gaussians:
            params_file = self.source_path + 'normal_function_gauss_params'
            if os.path.isfile(params_file):
                with open(params_file, 'rb') as f:
                    return pickle.load(f)

        _, sigma = ccm_utils.random_gaussian(self.n_features, max_std=self.max_std)
        mu = np.random.uniform(MU_LO, MU_HI, size=self.n_features)
        gauss = (mu, sigma)

        os.makedirs(self.source_path, exist_ok=True)
        with open(self.source_path + 'normal_function_gauss_params', 'wb') as f:
            pickle.dump(gauss, f)
        return gauss

    # ─────────────────────────────────────────────────────────────────────────
    # ── Dispatch and I/O methods ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def getfilename(self):
        for name in self.filenames:
            yield name

    def read_datastream(self, file_path: str) -> Tuple[pd.DataFrame, list]:
        """Return one complete data stream as a (DataFrame, change_indices) pair.

        Parameters
        ----------
        file_path : Path to a CSV (only used when ondemand=False).

        Returns
        -------
        df              : DataFrame of shape (len_seq, n_features).
        change_indices  : list of int — time indices where the active mode
                          switches.  For ``transition_mode != 'instant'`` these
                          are the *start* indices of the transition windows.
        """
        if self.ondemand:
            df, change_indices = self._generate_stream()
            if self.saveData:
                df.to_csv(self.sinkpath + f'{next(self.gename)}.csv', index=False)
            return df, change_indices
        df = pd.read_csv(f'{file_path}')
        return df, []

    # ─────────────────────────────────────────────────────────────────────────
    # ── Core generation ───────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_change_points(self):
        """Sample regime-switch indices (Bernoulli per step)."""
        change_points = []
        for i in range(1, self.len_seq):
            if np.random.uniform() < self.train_change_freq:
                change_points.append(i)
        return change_points

    def _pick_constellation_mode(self, current_idx):
        """Pick a random mode index different from ``current_idx``."""
        n = len(self.constellation)
        if n <= 1:
            return current_idx
        candidates = [i for i in range(n) if i != current_idx]
        return candidates[np.random.randint(len(candidates))]

    def _generate_stream(self) -> Tuple[pd.DataFrame, list]:
        """Generate a length-``len_seq`` stream with regime switches.

        Uses ``self.transition_mode`` to decide how each regime change is
        rendered (instant / incremental / gradual).
        """
        change_points = self._generate_change_points()
        # Boundary list: [0, cp_1, cp_2, ..., len_seq]
        boundaries = [0] + change_points + [self.len_seq]

        cur_idx = 0
        gauss_cur = self.constellation[cur_idx]

        rows = []  # accumulating per-step samples; each entry is shape (k, dim)

        for seg_id in range(len(boundaries) - 1):
            start = boundaries[seg_id]
            end = boundaries[seg_id + 1]
            seg_len = end - start

            if seg_id == 0:
                # First segment: pure draws from the starting mode.
                rows.append(_sample_clipped(
                    gauss_cur, seg_len, self.clip_lo, self.clip_hi))
                continue

            # Switch active mode.
            new_idx = self._pick_constellation_mode(cur_idx)
            gauss_next = self.constellation[new_idx]

            if self.transition_mode == 'instant':
                rows.append(_sample_clipped(
                    gauss_next, seg_len, self.clip_lo, self.clip_hi))
            elif self.transition_mode == 'incremental':
                rows.append(self._render_incremental(
                    gauss_cur, gauss_next, seg_len))
            elif self.transition_mode == 'gradual':
                rows.append(self._render_gradual(
                    gauss_cur, gauss_next, seg_len))

            cur_idx = new_idx
            gauss_cur = gauss_next

        self.change_ids_ic = change_points  # backwards-friendly attribute name
        data = np.vstack(rows)
        return pd.DataFrame(data), change_points

    # ── Transition renderers ────────────────────────────────────────────────

    def _render_incremental(self, gauss_cur, gauss_next, seg_len):
        """Render a regime change as an incremental (mu, sigma) interpolation.

        For ``W = min(n_drift_samples, seg_len)`` steps, every sample is drawn
        from a Gaussian linearly interpolated between ``gauss_cur`` and
        ``gauss_next``.  Remaining steps in the segment are drawn from
        ``gauss_next`` directly.
        """
        W = min(self.n_drift_samples, seg_len)
        out = np.empty((seg_len, self.n_features))

        for k in range(W):
            alpha = (k + 1) / W  # ends at 1.0 (gauss_next) on the last step
            g_alpha = _interpolate_gaussian(gauss_cur, gauss_next, alpha)
            out[k] = _sample_clipped(g_alpha, 1, self.clip_lo, self.clip_hi)[0]

        if seg_len > W:
            out[W:] = _sample_clipped(
                gauss_next, seg_len - W, self.clip_lo, self.clip_hi)
        return out

    def _render_gradual(self, gauss_cur, gauss_next, seg_len):
        """Render a regime change as block-wise probabilistic mixing.

        The transition window of length ``W = min(n_drift_samples, seg_len)``
        is split into ``B = min(n_gradual_blocks, W)`` equal blocks; each
        block draws all its samples from one of the two endpoints picked
        with probability ``mixing_weight(b/(B-1))`` of being ``gauss_next``.
        """
        W = min(self.n_drift_samples, seg_len)
        B = min(self.n_gradual_blocks, W)

        block_edges = np.round(np.linspace(0, W, B + 1)).astype(int)
        block_sizes = np.diff(block_edges)

        rows = []
        for b in range(B):
            frac = b / (B - 1) if B > 1 else 1.0
            w = ccm_utils.mixing_weight(frac, self.drift_type)
            gauss_b = gauss_next if np.random.rand() < w else gauss_cur
            rows.append(_sample_clipped(
                gauss_b, int(block_sizes[b]), self.clip_lo, self.clip_hi))

        if seg_len > W:
            rows.append(_sample_clipped(
                gauss_next, seg_len - W, self.clip_lo, self.clip_hi))
        return np.vstack(rows)

    def __str__(self):
        return "SyntheticDataset"
