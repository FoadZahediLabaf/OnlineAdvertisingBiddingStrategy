"""
ccm_nova.synthetic.constellation
============================
Strategies for building a **constellation** of Gaussians where every pair
has a controlled symmetric Kullback-Leibler (sKL) divergence.

- ``generate_constellation`` — alias for
  ``generate_constellation_gd_rotation_range`` (the default: rotation-
  parameterised GD with dead-zone loss).
- ``generate_constellation_rej`` — rejection sampling (fast per attempt, but
  acceptance rate drops with many modes or tight constraints).
- ``generate_constellation_gd`` — gradient-descent optimisation of the
  pairwise MSE loss between each pair's sKL and a scalar target.
- ``generate_constellation_gd_range`` — gradient-descent with a dead-zone
  loss: zero penalty when pairwise sKL falls within a specified range,
  quadratic penalty outside it.
- ``generate_constellation_gd_rotation`` — like ``_gd``, but covariance is
  parameterised by rotation angles (preserving eigenvalues of the anchor).
- ``generate_constellation_gd_rotation_range`` — like ``_gd_range``, with
  rotation parameterisation.

All five accept an optional ``phi_0=(mu, sigma)`` to fix mode 0.
"""

import time
import warnings

import numpy as np

from ola.utils.constellation.utils import (
    random_gaussian,
    init_changeMagnitude,
    convergeChangeMagnitude,
    symmetric_kullback_leibler,
    kullback_leibler,
    generate_canonical_rotation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Mean-domain constraint ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
#
# Means of every Gaussian in the constellation are constrained to live in
# [MU_LO, MU_HI] component-wise.  The motivation comes from the multi-campaign
# bidding-auction environment that consumes this synthetic data: the per-step
# observation is the maximum bid across N campaigns, modelled as a (low-
# variance) multivariate Gaussian with means in (0, 1).  Clamping each mean
# inside [0.1, 0.9] keeps every component well away from the saturation walls.

MU_LO = 0.1
MU_HI = 0.9


def _project_mean(mu):
    """Clip a mean vector component-wise to ``[MU_LO, MU_HI]`` (in place safe)."""
    return np.clip(mu, MU_LO, MU_HI)


def _mean_in_range(mu):
    """Return True iff every component of ``mu`` is inside ``[MU_LO, MU_HI]``."""
    return bool(np.all((mu >= MU_LO) & (mu <= MU_HI)))


def _random_mean_in_range(dim):
    """Sample a random mean vector uniformly in ``[MU_LO, MU_HI]^dim``."""
    return np.random.uniform(MU_LO, MU_HI, size=dim)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Shared helper: validate an externally-supplied anchor ────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_phi_0(phi_0, dim):
    """Validate an externally-supplied anchor ``(mu, sigma)`` against ``dim``.

    Returns ``None`` unchanged when ``phi_0`` is ``None``. Otherwise returns
    ``(mu, sigma)`` as numpy arrays after verifying:
      - structure is a ``(mu, sigma)`` pair
      - ``mu.shape == (dim,)`` and ``sigma.shape == (dim, dim)``
      - ``sigma`` is symmetric and positive definite

    Raises ``ValueError`` on any of the above violations.
    """
    if phi_0 is None:
        return None
    try:
        mu, sigma = phi_0
    except (TypeError, ValueError):
        raise ValueError("phi_0 must be a (mu, sigma) tuple/pair.")
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    if mu.shape != (dim,):
        raise ValueError(
            f"phi_0 mu has shape {mu.shape}, expected ({dim},).")
    if sigma.shape != (dim, dim):
        raise ValueError(
            f"phi_0 sigma has shape {sigma.shape}, expected ({dim}, {dim}).")
    if not np.allclose(sigma, sigma.T, atol=1e-9):
        raise ValueError("phi_0 sigma is not symmetric.")
    eigvals = np.linalg.eigvalsh(sigma)
    if np.any(eigvals <= 0):
        raise ValueError(
            f"phi_0 sigma is not positive definite (min eigval={eigvals.min():.3e}).")
    if not _mean_in_range(mu):
        raise ValueError(
            f"phi_0 mu must be component-wise in [{MU_LO}, {MU_HI}]; got {mu}.")
    return mu, sigma


# ═══════════════════════════════════════════════════════════════════════════════
# ── 1. Rejection-sampling strategy ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def generate_constellation_rej(dim, n_modes, skl_range, timeout=60.0, seed=None,
                               phi_0=None, max_std=0.5):
    """Generate a constellation of Gaussians where every pair has sKL within skl_range.

    Uses rejection sampling: each new candidate is derived from a randomly chosen
    existing Gaussian in the constellation. The candidate is accepted only if its
    sKL to *every* already-accepted Gaussian falls within ``skl_range``.

    Parameters
    ----------
    dim       : int — dimensionality of the Gaussians.
    n_modes   : int — how many Gaussians to generate.
    skl_range : (float, float) — (min, max) for the pairwise sKL constraint.
    timeout   : float — maximum wall-clock seconds before giving up (default 60).
    seed      : int or None — random seed for reproducibility.
    phi_0     : tuple ``(mu, sigma)`` or None (default).
        Optional fixed anchor for mode 0. ``mu`` must have shape ``(dim,)`` and
        ``sigma`` must be a symmetric positive-definite ``(dim, dim)`` matrix.
        When ``None``, the anchor is sampled randomly. Useful for seeding a
        constellation around a Gaussian extracted from real data via
        ``gmm_fit.extract_gaussian``.

    Returns
    -------
    constellation : list of (mu, sigma) tuples, length <= n_modes.

    Warns via warnings.warn if the timeout is reached before all modes are found.
    In that case the partial constellation (with as many modes as were accepted)
    is returned.
    """
    phi_0 = _validate_phi_0(phi_0, dim)

    if seed is not None:
        np.random.seed(seed)

    skl_lo, skl_hi = skl_range

    # First Gaussian — user-supplied anchor or random with mean in [MU_LO, MU_HI].
    if phi_0 is not None:
        g0 = phi_0
    else:
        _, sigma0 = random_gaussian(dim, max_std=max_std)
        g0 = (_random_mean_in_range(dim), sigma0)
    constellation = [g0]

    if n_modes <= 1:
        return constellation

    start = time.monotonic()

    while len(constellation) < n_modes:
        if time.monotonic() - start > timeout:
            warnings.warn(
                f"generate_constellation_rej: timeout after {timeout}s. "
                f"Returning {len(constellation)}/{n_modes} modes.",
                RuntimeWarning,
                stacklevel=2,
            )
            return constellation

        # Pick a random anchor from the existing constellation
        anchor = constellation[np.random.randint(len(constellation))]

        # Derive a candidate at a random sKL within the range (relative to anchor)
        target_skl = np.random.uniform(skl_lo, skl_hi)
        angles, P, rho, dir_shift = init_changeMagnitude(anchor, target_skl)
        candidate = convergeChangeMagnitude(anchor, target_skl, angles, P, rho, dir_shift, eps=0.1)

        # Reject candidates whose mean leaves the admissible box.
        cand_mu = np.asarray(candidate[0])
        if not _mean_in_range(cand_mu):
            continue

        # Check pairwise sKL against ALL existing Gaussians
        ok = True
        for g in constellation:
            skl = symmetric_kullback_leibler(candidate, g)
            if skl < skl_lo or skl > skl_hi:
                ok = False
                break

        if ok:
            constellation.append(candidate)

    return constellation


# ═══════════════════════════════════════════════════════════════════════════════
# ── 2. Gradient-descent strategy ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── KL gradient helpers (analytical, numpy-only) ────────────────────────────

def _kl_gradient_mu(mu_p, mu_q, sigma_q_inv):
    """Gradient of KL(p||q) with respect to mu_p.

    ∂KL(p||q)/∂mu_p = Σ_q⁻¹ (mu_p − mu_q)
    """
    return sigma_q_inv @ (mu_p - mu_q)


def _skl_gradients(mu_i, sigma_i, mu_j, sigma_j):
    """Gradients of sKL(g_i, g_j) with respect to mu_i and sigma_i.

    sKL = KL(i||j) + KL(j||i)

    Returns ∂sKL/∂mu_i and ∂sKL/∂Σ_i (the gradient for mode i only).
    """
    sigma_i_inv = np.linalg.inv(sigma_i)
    sigma_j_inv = np.linalg.inv(sigma_j)

    # ∂sKL/∂mu_i = ∂KL(i||j)/∂mu_i + ∂KL(j||i)/∂mu_i
    #   KL(i||j): p=i, q=j → ∂/∂mu_p = Σ_j⁻¹ (mu_i − mu_j)
    #   KL(j||i): p=j, q=i → ∂/∂mu_q = −Σ_i⁻¹ (mu_j − mu_i) = Σ_i⁻¹ (mu_i − mu_j)
    grad_mu = (sigma_j_inv + sigma_i_inv) @ (mu_i - mu_j)

    # ∂sKL/∂Σ_i = ∂KL(i||j)/∂Σ_i + ∂KL(j||i)/∂Σ_i
    #   = ½(Σ_j⁻¹ − Σ_i⁻¹) + ½(Σ_i⁻¹ − Σ_i⁻¹ Σ_j Σ_i⁻¹ − Σ_i⁻¹ δδᵀ Σ_i⁻¹)
    #   The ±½ Σ_i⁻¹ terms cancel, giving the simplified closed form below.
    #   δ = mu_j − mu_i
    delta = (mu_j - mu_i).reshape(-1, 1)
    grad_sigma = 0.5 * (sigma_j_inv
                        - sigma_i_inv @ (sigma_j + delta @ delta.T) @ sigma_i_inv)

    return grad_mu, grad_sigma


# ── Cholesky parameterisation helpers ────────────────────────────────────────

def _cholesky_from_raw(L_raw):
    """Convert a raw (dim, dim) matrix to a valid lower-triangular Cholesky factor.

    The lower triangle is kept as-is; the diagonal is exponentiated to ensure
    positivity (unconstrained optimisation of log-diagonal).
    """
    L = np.tril(L_raw)
    np.fill_diagonal(L, np.exp(np.diag(L_raw)))
    return L


def _sigma_from_raw(L_raw):
    """Reconstruct Σ = L Lᵀ from raw parameters."""
    L = _cholesky_from_raw(L_raw)
    return L @ L.T


def _grad_L_from_grad_sigma(grad_sigma, L_raw):
    """Chain rule: ∂Loss/∂L_raw from ∂Loss/∂Σ.

    For off-diagonal (lower-triangle) entries:
        ∂Loss/∂L_raw_ij = (2 · ∂Loss/∂Σ @ L)_ij
    For diagonal entries (log-parameterised):
        ∂Loss/∂L_raw_ii = (2 · ∂Loss/∂Σ @ L)_ii · exp(L_raw_ii)
    """
    L = _cholesky_from_raw(L_raw)
    # Symmetrise grad_sigma (should already be symmetric, but be safe)
    G = 0.5 * (grad_sigma + grad_sigma.T)
    raw_grad = 2.0 * G @ L

    # Apply chain rule for the exp-diagonal
    grad_out = np.tril(raw_grad)
    diag_exp = np.exp(np.diag(L_raw))
    np.fill_diagonal(grad_out, np.diag(raw_grad) * diag_exp)
    return grad_out


# ── Adam optimiser (minimal numpy implementation) ────────────────────────────

class _Adam:
    """Minimal Adam optimiser for a list of numpy arrays."""

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Main GD function ─────────────────────────────────────────────────────────

def generate_constellation_gd(dim, n_modes, target_skl, max_iter=30000,
                               lr=0.01, patience=200, n_restarts=1,
                               seed=None, phi_0=None, max_std=0.5):
    """Generate a constellation of Gaussians via gradient descent.

    Minimises  L = Σ_{i<j} (sKL(g_i, g_j) − target_skl)²  over the means
    and Cholesky factors of all modes (except mode 0 which is fixed as anchor).

    Parameters
    ----------
    dim         : int — dimensionality of the Gaussians.
    n_modes     : int — number of Gaussians in the constellation.
    target_skl  : float — desired pairwise sKL between every pair.
    max_iter    : int — maximum gradient-descent iterations per restart (default 5000).
    lr          : float — Adam learning rate (default 0.01).
    patience    : int — early-stopping patience: stop if loss doesn't improve
                  for this many consecutive iterations (default 200).
    n_restarts  : int — number of random restarts; best result is kept (default 1).
    seed        : int or None — random seed for reproducibility.
    phi_0       : tuple ``(mu, sigma)`` or None (default).
        Optional fixed anchor for mode 0. ``mu`` must have shape ``(dim,)`` and
        ``sigma`` must be a symmetric positive-definite ``(dim, dim)`` matrix.
        When ``None``, the anchor is sampled randomly. Useful for seeding a
        constellation around a Gaussian extracted from real data via
        ``gmm_fit.extract_gaussian``.

    Returns
    -------
    result : dict with keys:
        'constellation' : list of (mu, sigma) tuples — the optimised modes.
        'skl_matrix'    : (n_modes, n_modes) ndarray — pairwise sKL values.
        'loss'          : float — final MSE loss of the best restart.
        'iterations'    : int — number of iterations used by the best restart.
    """
    phi_0 = _validate_phi_0(phi_0, dim)

    if seed is not None:
        np.random.seed(seed)

    if n_modes <= 1:
        if phi_0 is not None:
            g0 = phi_0
        else:
            _, sigma0 = random_gaussian(dim, max_std=max_std)
            g0 = (_random_mean_in_range(dim), sigma0)
        return {
            'constellation': [g0],
            'skl_matrix': np.zeros((1, 1)),
            'loss': 0.0,
            'iterations': 0,
        }

    best_result = None

    for restart in range(n_restarts):
        result = _run_single_gd(dim, n_modes, target_skl, max_iter, lr,
                                patience, phi_0=phi_0, max_std=max_std)
        if best_result is None or result['loss'] < best_result['loss']:
            best_result = result

    return best_result


def _run_single_gd(dim, n_modes, target_skl, max_iter, lr, patience,
                   phi_0=None, max_std=0.5):
    """One run of gradient descent (called by generate_constellation_gd)."""

    # ── Initialise parameters ─────────────────────────────────────────────
    # Mode 0: fixed anchor (user-supplied or random with mean in [MU_LO, MU_HI])
    if phi_0 is not None:
        mu0, sigma0 = phi_0
    else:
        _, sigma0 = random_gaussian(dim, max_std=max_std)
        mu0 = _random_mean_in_range(dim)
    L0_raw = np.linalg.cholesky(sigma0)
    # Store log-diagonal for consistency (though we won't optimise this one)
    L0_raw_diag = np.diag(L0_raw).copy()
    L0_raw[np.diag_indices(dim)] = np.log(L0_raw_diag)

    # Modes 1..M-1: free parameters (means initialised inside [MU_LO, MU_HI])
    mus = []       # list of (dim,) arrays
    L_raws = []    # list of (dim, dim) arrays

    for _ in range(n_modes - 1):
        _, sigma = random_gaussian(dim, max_std=max_std)
        mu = _random_mean_in_range(dim)
        L = np.linalg.cholesky(sigma)
        L_raw = L.copy()
        L_raw[np.diag_indices(dim)] = np.log(np.diag(L))
        mus.append(mu)
        L_raws.append(L_raw)

    # Flatten into a single param list for Adam
    params = []
    for mu in mus:
        params.append(mu)
    for L_raw in L_raws:
        params.append(L_raw)
    optimizer = _Adam(params, lr=lr)

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float('inf')
    no_improve = 0
    final_iter = 0
    loss_history = []

    for iteration in range(max_iter):
        # Reconstruct all Gaussians
        all_mus = [mu0] + mus
        all_sigmas = [_sigma_from_raw(L0_raw)] + [_sigma_from_raw(lr_) for lr_ in L_raws]

        # Compute loss and gradients
        loss = 0.0
        grad_mus = [np.zeros(dim) for _ in range(n_modes - 1)]
        grad_L_raws = [np.zeros((dim, dim)) for _ in range(n_modes - 1)]

        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                skl_ij = symmetric_kullback_leibler(
                    (all_mus[i], all_sigmas[i]),
                    (all_mus[j], all_sigmas[j])
                )
                residual = skl_ij - target_skl
                loss += residual ** 2

                # Gradients of sKL w.r.t. mode i
                if i > 0:  # mode 0 is fixed
                    g_mu_i, g_sigma_i = _skl_gradients(
                        all_mus[i], all_sigmas[i],
                        all_mus[j], all_sigmas[j])
                    grad_mus[i - 1] += 2 * residual * g_mu_i
                    grad_L_raws[i - 1] += _grad_L_from_grad_sigma(
                        2 * residual * g_sigma_i, L_raws[i - 1])

                # Gradients of sKL w.r.t. mode j
                if j > 0:  # mode 0 is fixed
                    g_mu_j, g_sigma_j = _skl_gradients(
                        all_mus[j], all_sigmas[j],
                        all_mus[i], all_sigmas[i])
                    grad_mus[j - 1] += 2 * residual * g_mu_j
                    grad_L_raws[j - 1] += _grad_L_from_grad_sigma(
                        2 * residual * g_sigma_j, L_raws[j - 1])

        # Adam step
        grads = list(grad_mus) + list(grad_L_raws)
        optimizer.step(grads)

        # Project means back into the feasible box [MU_LO, MU_HI].
        for k in range(len(mus)):
            np.clip(mus[k], MU_LO, MU_HI, out=mus[k])

        loss_history.append(float(loss))

        # Early stopping (relative improvement < 0.01%)
        final_iter = iteration + 1
        if best_loss > 0 and (best_loss - loss) < best_loss * 1e-4:
            no_improve += 1
            if loss < best_loss:
                best_loss = loss
            if no_improve >= patience:
                break
        else:
            best_loss = loss
            no_improve = 0

    # ── Build result ──────────────────────────────────────────────────────
    all_mus_final = [mu0] + mus
    all_sigmas_final = [_sigma_from_raw(L0_raw)] + [_sigma_from_raw(lr_) for lr_ in L_raws]

    constellation = [(mu.copy(), sigma.copy())
                     for mu, sigma in zip(all_mus_final, all_sigmas_final)]

    skl_matrix = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            s = symmetric_kullback_leibler(constellation[i], constellation[j])
            skl_matrix[i, j] = s
            skl_matrix[j, i] = s

    return {
        'constellation': constellation,
        'skl_matrix': skl_matrix,
        'loss': best_loss,
        'iterations': final_iter,
        'loss_history': loss_history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── 3. Gradient-descent strategy with dead-zone (range) loss ─────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _dead_zone_residual(skl_val, skl_lo, skl_hi):
    """Return the residual for the dead-zone loss.

    - 0            if skl_lo <= skl_val <= skl_hi  (no penalty)
    - skl_val - skl_lo  if skl_val < skl_lo        (too close)
    - skl_val - skl_hi  if skl_val > skl_hi        (too far)
    """
    if skl_val < skl_lo:
        return skl_val - skl_lo
    elif skl_val > skl_hi:
        return skl_val - skl_hi
    else:
        return 0.0


def generate_constellation_gd_range(dim, n_modes, skl_range, max_iter=30000,
                                    lr=0.01, patience=200, n_restarts=1,
                                    seed=None, verbose=0, margin=None,
                                    phi_0=None, max_std=0.5):
    """Generate a constellation of Gaussians via gradient descent with a dead-zone loss.

    The optimiser uses a *tighter* dead-zone ``[skl_lo + margin, skl_hi - margin]``
    so that gradients push pairs away from the boundary.  Training stops early as
    soon as every pairwise sKL falls within the actual ``skl_range``.

    Parameters
    ----------
    dim         : int — dimensionality of the Gaussians.
    n_modes     : int — number of Gaussians in the constellation.
    skl_range   : (float, float) — (min, max) acceptable pairwise sKL.
    max_iter    : int — maximum gradient-descent iterations per restart (default 5000).
    lr          : float — Adam learning rate (default 0.01).
    patience    : int — early-stopping patience. Training stops if the loss does
                  not improve by at least 0.01% for this many consecutive
                  iterations, or when all pairs fall within skl_range (default 200).
    n_restarts  : int — number of random restarts; best result is kept (default 1).
    seed        : int or None — random seed for reproducibility.
    verbose     : int — if > 0, print progress every `verbose` iterations (default 0 = silent).
    margin      : float or None — shrinks the dead-zone by this amount on each side.
                  Dead-zone becomes ``[skl_lo + margin, skl_hi - margin]``.
                  If None, defaults to ``0.1 * (skl_hi - skl_lo)``.
    phi_0       : tuple ``(mu, sigma)`` or None (default).
        Optional fixed anchor for mode 0. ``mu`` must have shape ``(dim,)`` and
        ``sigma`` must be a symmetric positive-definite ``(dim, dim)`` matrix.
        When ``None``, the anchor is sampled randomly. Useful for seeding a
        constellation around a Gaussian extracted from real data via
        ``gmm_fit.extract_gaussian``.

    Returns
    -------
    result : dict with keys:
        'constellation' : list of (mu, sigma) tuples.
        'skl_matrix'    : (n_modes, n_modes) ndarray — pairwise sKL values.
        'loss'          : float — final dead-zone loss of the best restart.
        'iterations'    : int — iterations used by the best restart.
        'loss_history'  : list of float — per-iteration loss.
        'all_in_range'  : bool — True if every pair's sKL is inside skl_range.
    """
    phi_0 = _validate_phi_0(phi_0, dim)

    if seed is not None:
        np.random.seed(seed)

    accept_lo, accept_hi = skl_range
    if accept_lo > accept_hi:
        raise ValueError(f"skl_range[0] must be <= skl_range[1], got {skl_range}")

    if margin is None:
        margin = 0.1 * (accept_hi - accept_lo)
    dz_lo = accept_lo + margin
    dz_hi = accept_hi - margin

    if n_modes <= 1:
        if phi_0 is not None:
            g0 = phi_0
        else:
            _, sigma0 = random_gaussian(dim, max_std=max_std)
            g0 = (_random_mean_in_range(dim), sigma0)
        return {
            'constellation': [g0],
            'skl_matrix': np.zeros((1, 1)),
            'loss': 0.0,
            'iterations': 0,
            'loss_history': [],
            'all_in_range': True,
        }

    best_result = None

    for restart in range(n_restarts):
        result = _run_single_gd_range(dim, n_modes, dz_lo, dz_hi,
                                      accept_lo, accept_hi,
                                      max_iter, lr, patience, verbose=verbose,
                                      phi_0=phi_0, max_std=max_std)
        if best_result is None or result['loss'] < best_result['loss']:
            best_result = result

    return best_result


def _run_single_gd_range(dim, n_modes, skl_lo, skl_hi, accept_lo, accept_hi,
                         max_iter, lr, patience, verbose=0, phi_0=None,
                         max_std=0.5):
    """One run of dead-zone gradient descent."""

    # ── Initialise parameters (same as _run_single_gd) ────────────────────
    if phi_0 is not None:
        mu0, sigma0 = phi_0
    else:
        _, sigma0 = random_gaussian(dim, max_std=max_std)
        mu0 = _random_mean_in_range(dim)
    L0_raw = np.linalg.cholesky(sigma0)
    L0_raw_diag = np.diag(L0_raw).copy()
    L0_raw[np.diag_indices(dim)] = np.log(L0_raw_diag)

    mus = []
    L_raws = []

    for _ in range(n_modes - 1):
        _, sigma = random_gaussian(dim, max_std=max_std)
        mu = _random_mean_in_range(dim)
        L = np.linalg.cholesky(sigma)
        L_raw = L.copy()
        L_raw[np.diag_indices(dim)] = np.log(np.diag(L))
        mus.append(mu)
        L_raws.append(L_raw)

    params = []
    for mu in mus:
        params.append(mu)
    for L_raw in L_raws:
        params.append(L_raw)

    optimizer = _Adam(params, lr=lr)

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float('inf')
    no_improve = 0
    final_iter = 0
    loss_history = []

    for iteration in range(max_iter):
        all_mus = [mu0] + mus
        all_sigmas = [_sigma_from_raw(L0_raw)] + [_sigma_from_raw(lr_) for lr_ in L_raws]

        loss = 0.0
        all_accepted = True
        grad_mus = [np.zeros(dim) for _ in range(n_modes - 1)]
        grad_L_raws = [np.zeros((dim, dim)) for _ in range(n_modes - 1)]

        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                skl_ij = symmetric_kullback_leibler(
                    (all_mus[i], all_sigmas[i]),
                    (all_mus[j], all_sigmas[j])
                )
                if skl_ij < accept_lo or skl_ij > accept_hi:
                    all_accepted = False

                residual = _dead_zone_residual(skl_ij, skl_lo, skl_hi)
                loss += residual ** 2

                if residual == 0.0:
                    continue  # inside the dead zone — no gradient

                # Gradients w.r.t. mode i
                if i > 0:
                    g_mu_i, g_sigma_i = _skl_gradients(
                        all_mus[i], all_sigmas[i],
                        all_mus[j], all_sigmas[j])
                    grad_mus[i - 1] += 2 * residual * g_mu_i
                    grad_L_raws[i - 1] += _grad_L_from_grad_sigma(
                        2 * residual * g_sigma_i, L_raws[i - 1])

                # Gradients w.r.t. mode j
                if j > 0:
                    g_mu_j, g_sigma_j = _skl_gradients(
                        all_mus[j], all_sigmas[j],
                        all_mus[i], all_sigmas[i])
                    grad_mus[j - 1] += 2 * residual * g_mu_j
                    grad_L_raws[j - 1] += _grad_L_from_grad_sigma(
                        2 * residual * g_sigma_j, L_raws[j - 1])

        grads = list(grad_mus) + list(grad_L_raws)
        optimizer.step(grads)

        # Project means back into the feasible box [MU_LO, MU_HI].
        for k in range(len(mus)):
            np.clip(mus[k], MU_LO, MU_HI, out=mus[k])

        loss_history.append(float(loss))

        if verbose > 0 and (iteration + 1) % verbose == 0:
            print(f"      iter {iteration+1:>6}/{max_iter}  "
                  f"loss={loss:>12.6f}  "
                  f"best={best_loss:>12.6f}  "
                  f"lr={optimizer.lr:.6f}  "
                  f"no_improve={no_improve}/{patience}", flush=True)

        final_iter = iteration + 1

        if all_accepted:
            best_loss = min(best_loss, loss)
            if verbose > 0:
                print(f"      iter {iteration+1:>6}/{max_iter}  "
                      f"loss={loss:.2e} — all pairs in accept range, stopping.",
                      flush=True)
            break

        if best_loss > 0 and (best_loss - loss) < best_loss * 1e-4:
            no_improve += 1
            if loss < best_loss:
                best_loss = loss
            if no_improve >= patience:
                if verbose > 0:
                    print(f"      iter {iteration+1:>6}/{max_iter}  "
                          f"patience exhausted (no >0.01% improvement "
                          f"for {patience} iters), stopping.", flush=True)
                break
        else:
            best_loss = loss
            no_improve = 0

    # ── Build result ──────────────────────────────────────────────────────
    all_mus_final = [mu0] + mus
    all_sigmas_final = [_sigma_from_raw(L0_raw)] + [_sigma_from_raw(lr_) for lr_ in L_raws]

    constellation = [(mu.copy(), sigma.copy())
                     for mu, sigma in zip(all_mus_final, all_sigmas_final)]

    skl_matrix = np.zeros((n_modes, n_modes))
    all_in_range = True
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            s = symmetric_kullback_leibler(constellation[i], constellation[j])
            skl_matrix[i, j] = s
            skl_matrix[j, i] = s
            if s < accept_lo or s > accept_hi:
                all_in_range = False

    return {
        'constellation': constellation,
        'skl_matrix': skl_matrix,
        'loss': best_loss,
        'iterations': final_iter,
        'loss_history': loss_history,
        'all_in_range': all_in_range,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── 4. Rotation-parameterised GD (eigenvalue-preserving) ─────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Rotation helpers ─────────────────────────────────────────────────────────

def _dT_dtheta(angles, dim, k):
    """Derivative of the canonical rotation matrix T w.r.t. angle k.

    Returns a (dim, dim) matrix that is zero everywhere except the 2x2 block
    at rows/cols (2k, 2k+1):
        [-sin(theta_k), -cos(theta_k)]
        [ cos(theta_k), -sin(theta_k)]
    """
    dT = np.zeros((dim, dim))
    theta_k = angles[k]
    row = 2 * k
    dT[row,     row]     = -np.sin(theta_k)
    dT[row,     row + 1] = -np.cos(theta_k)
    dT[row + 1, row]     =  np.cos(theta_k)
    dT[row + 1, row + 1] = -np.sin(theta_k)
    return dT


def _Q_from_angles(P, angles, dim):
    """Build Q = P @ T(angles) @ P^T."""
    T = generate_canonical_rotation(angles, dim)
    return P @ T @ P.T


def _sigma_from_rotation(sigma0, P, angles, dim):
    """Compute Sigma_i = Q^T @ sigma0 @ Q using rotation parameterisation."""
    Q = _Q_from_angles(P, angles, dim)
    return Q.T @ sigma0 @ Q


def _grad_angles_from_grad_sigma(grad_sigma, sigma0, P, angles, dim):
    """Chain rule: convert dL/dSigma_i into dL/d(angles_i).

    For each angle k:
        dQ_k = P @ dT/dtheta_k @ P^T
        dSigma_k = dQ_k^T @ sigma0 @ Q + Q^T @ sigma0 @ dQ_k
        grad_angles[k] = sum(grad_sigma * dSigma_k)   (Frobenius inner product)
    """
    m = len(angles)
    Q = _Q_from_angles(P, angles, dim)
    grad_angles = np.zeros(m)
    for k in range(m):
        dT_k = _dT_dtheta(angles, dim, k)
        dQ_k = P @ dT_k @ P.T
        dSigma_k = dQ_k.T @ sigma0 @ Q + Q.T @ sigma0 @ dQ_k
        grad_angles[k] = np.sum(grad_sigma * dSigma_k)
    return grad_angles


# ── Exact-target rotation GD ────────────────────────────────────────────────

def generate_constellation_gd_rotation(dim, n_modes, target_skl, max_iter=30000,
                                        lr=0.01, patience=200, n_restarts=1,
                                        seed=None, phi_0=None, max_std=0.5):
    """Generate a constellation via GD with rotation-parameterised covariances.

    All modes share the eigenvalues of the anchor (mode 0). Each free mode's
    covariance is Sigma_i = Q_i^T @ sigma0 @ Q_i where Q_i = P_i @ T(angles_i) @ P_i^T.
    P_i is a fixed random orthogonal matrix; angles_i are optimised.

    Loss: L = sum_{i<j} (sKL(g_i, g_j) - target_skl)^2

    Parameters
    ----------
    dim, n_modes, target_skl, max_iter, lr, patience, n_restarts, seed :
        Same semantics as generate_constellation_gd.
    phi_0       : tuple ``(mu, sigma)`` or None (default).
        Optional fixed anchor for mode 0. ``mu`` must have shape ``(dim,)`` and
        ``sigma`` must be a symmetric positive-definite ``(dim, dim)`` matrix.
        When ``None``, the anchor is sampled randomly. Useful for seeding a
        constellation around a Gaussian extracted from real data via
        ``gmm_fit.extract_gaussian``. Note: all free modes inherit the
        eigenvalues of ``sigma``.

    Returns
    -------
    result : dict with keys 'constellation', 'skl_matrix', 'loss',
             'iterations', 'loss_history'.
    """
    phi_0 = _validate_phi_0(phi_0, dim)

    if seed is not None:
        np.random.seed(seed)

    if n_modes <= 1:
        if phi_0 is not None:
            g0 = phi_0
        else:
            _, sigma0 = random_gaussian(dim, max_std=max_std)
            g0 = (_random_mean_in_range(dim), sigma0)
        return {
            'constellation': [g0],
            'skl_matrix': np.zeros((1, 1)),
            'loss': 0.0,
            'iterations': 0,
            'loss_history': [],
        }

    best_result = None
    for restart in range(n_restarts):
        result = _run_single_gd_rotation(dim, n_modes, target_skl,
                                          max_iter, lr, patience,
                                          phi_0=phi_0, max_std=max_std)
        if best_result is None or result['loss'] < best_result['loss']:
            best_result = result

    return best_result


def _run_single_gd_rotation(dim, n_modes, target_skl, max_iter, lr, patience,
                            phi_0=None, max_std=0.5):
    """One run of rotation-parameterised GD (exact target)."""

    m = max(1, dim // 2)  # number of rotation angles

    # Mode 0: fixed anchor (user-supplied or random with mean in [MU_LO, MU_HI])
    if phi_0 is not None:
        mu0, sigma0 = phi_0
    else:
        _, sigma0 = random_gaussian(dim, max_std=max_std)
        mu0 = _random_mean_in_range(dim)

    # Modes 1..M-1: free parameters (means + rotation angles)
    mus = []
    P_mats = []       # fixed random orthogonal matrices
    angles_list = []   # optimisable angle vectors

    for _ in range(n_modes - 1):
        mu = _random_mean_in_range(dim)
        mus.append(mu)

        A = np.random.randn(dim, dim)
        P, _ = np.linalg.qr(A)
        P_mats.append(P)

        angles = 0.1 * np.random.randn(m)
        angles_list.append(angles)

    # Flatten params for Adam: [mu_1, ..., mu_{M-1}, angles_1, ..., angles_{M-1}]
    params = list(mus) + list(angles_list)
    optimizer = _Adam(params, lr=lr)

    # Training loop
    best_loss = float('inf')
    no_improve = 0
    final_iter = 0
    loss_history = []

    for iteration in range(max_iter):
        # Reconstruct all Gaussians
        all_mus = [mu0] + mus
        all_sigmas = [sigma0] + [
            _sigma_from_rotation(sigma0, P_mats[i], angles_list[i], dim)
            for i in range(n_modes - 1)
        ]

        # Compute loss and gradients
        loss = 0.0
        grad_mus = [np.zeros(dim) for _ in range(n_modes - 1)]
        grad_angles = [np.zeros(m) for _ in range(n_modes - 1)]

        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                skl_ij = symmetric_kullback_leibler(
                    (all_mus[i], all_sigmas[i]),
                    (all_mus[j], all_sigmas[j])
                )
                residual = skl_ij - target_skl
                loss += residual ** 2

                # Gradients w.r.t. mode i
                if i > 0:
                    g_mu_i, g_sigma_i = _skl_gradients(
                        all_mus[i], all_sigmas[i],
                        all_mus[j], all_sigmas[j])
                    grad_mus[i - 1] += 2 * residual * g_mu_i
                    grad_angles[i - 1] += _grad_angles_from_grad_sigma(
                        2 * residual * g_sigma_i,
                        sigma0, P_mats[i - 1], angles_list[i - 1], dim)

                # Gradients w.r.t. mode j
                if j > 0:
                    g_mu_j, g_sigma_j = _skl_gradients(
                        all_mus[j], all_sigmas[j],
                        all_mus[i], all_sigmas[i])
                    grad_mus[j - 1] += 2 * residual * g_mu_j
                    grad_angles[j - 1] += _grad_angles_from_grad_sigma(
                        2 * residual * g_sigma_j,
                        sigma0, P_mats[j - 1], angles_list[j - 1], dim)

        grads = list(grad_mus) + list(grad_angles)
        optimizer.step(grads)

        # Project means back into the feasible box [MU_LO, MU_HI].
        for k in range(len(mus)):
            np.clip(mus[k], MU_LO, MU_HI, out=mus[k])

        loss_history.append(float(loss))

        final_iter = iteration + 1
        if best_loss > 0 and (best_loss - loss) < best_loss * 1e-4:
            no_improve += 1
            if loss < best_loss:
                best_loss = loss
            if no_improve >= patience:
                break
        else:
            best_loss = loss
            no_improve = 0

    # Build result
    all_mus_final = [mu0] + mus
    all_sigmas_final = [sigma0] + [
        _sigma_from_rotation(sigma0, P_mats[i], angles_list[i], dim)
        for i in range(n_modes - 1)
    ]

    constellation = [(mu.copy(), sigma.copy())
                     for mu, sigma in zip(all_mus_final, all_sigmas_final)]

    skl_matrix = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            s = symmetric_kullback_leibler(constellation[i], constellation[j])
            skl_matrix[i, j] = s
            skl_matrix[j, i] = s

    return {
        'constellation': constellation,
        'skl_matrix': skl_matrix,
        'loss': best_loss,
        'iterations': final_iter,
        'loss_history': loss_history,
    }


# ── Range (dead-zone) rotation GD ───────────────────────────────────────────

def generate_constellation_gd_rotation_range(dim, n_modes, skl_range,
                                              max_iter=30000, lr=0.01,
                                              patience=200, n_restarts=1,
                                              seed=None, verbose=0,
                                              margin=None, phi_0=None,
                                              max_std=0.5):
    """Generate a constellation via GD with rotation parameterisation and dead-zone loss.

    Same rotation parameterisation as generate_constellation_gd_rotation, but
    the optimiser uses a tighter dead-zone ``[skl_lo + margin, skl_hi - margin]``
    and stops early when every pairwise sKL falls within ``skl_range``.

    Parameters
    ----------
    dim, n_modes, skl_range, max_iter, lr, patience, n_restarts, seed, verbose :
        Same semantics as generate_constellation_gd_range.
    margin      : float or None — shrinks the dead-zone by this amount on each side.
                  If None, defaults to ``0.1 * (skl_hi - skl_lo)``.
    phi_0       : tuple ``(mu, sigma)`` or None (default).
        Optional fixed anchor for mode 0. ``mu`` must have shape ``(dim,)`` and
        ``sigma`` must be a symmetric positive-definite ``(dim, dim)`` matrix.
        When ``None``, the anchor is sampled randomly. Useful for seeding a
        constellation around a Gaussian extracted from real data via
        ``gmm_fit.extract_gaussian``. Note: all free modes inherit the
        eigenvalues of ``sigma``.

    Returns
    -------
    result : dict with keys 'constellation', 'skl_matrix', 'loss',
             'iterations', 'loss_history', 'all_in_range'.
    """
    phi_0 = _validate_phi_0(phi_0, dim)

    if seed is not None:
        np.random.seed(seed)

    accept_lo, accept_hi = skl_range
    if accept_lo > accept_hi:
        raise ValueError(f"skl_range[0] must be <= skl_range[1], got {skl_range}")

    if margin is None:
        margin = 0.1 * (accept_hi - accept_lo)
    dz_lo = accept_lo + margin
    dz_hi = accept_hi - margin

    if n_modes <= 1:
        if phi_0 is not None:
            g0 = phi_0
        else:
            _, sigma0 = random_gaussian(dim, max_std=max_std)
            g0 = (_random_mean_in_range(dim), sigma0)
        return {
            'constellation': [g0],
            'skl_matrix': np.zeros((1, 1)),
            'loss': 0.0,
            'iterations': 0,
            'loss_history': [],
            'all_in_range': True,
        }

    best_result = None
    for restart in range(n_restarts):
        result = _run_single_gd_rotation_range(dim, n_modes, dz_lo, dz_hi,
                                                accept_lo, accept_hi,
                                                max_iter, lr, patience,
                                                verbose=verbose,
                                                phi_0=phi_0, max_std=max_std)
        if best_result is None or result['loss'] < best_result['loss']:
            best_result = result

    return best_result


def _run_single_gd_rotation_range(dim, n_modes, skl_lo, skl_hi,
                                   accept_lo, accept_hi,
                                   max_iter, lr, patience, verbose=0,
                                   phi_0=None, max_std=0.5):
    """One run of rotation-parameterised dead-zone GD."""

    m = max(1, dim // 2)

    # Mode 0: fixed anchor (user-supplied or random with mean in [MU_LO, MU_HI])
    if phi_0 is not None:
        mu0, sigma0 = phi_0
    else:
        _, sigma0 = random_gaussian(dim, max_std=max_std)
        mu0 = _random_mean_in_range(dim)

    # Modes 1..M-1
    mus = []
    P_mats = []
    angles_list = []

    for _ in range(n_modes - 1):
        mu = _random_mean_in_range(dim)
        mus.append(mu)

        A = np.random.randn(dim, dim)
        P, _ = np.linalg.qr(A)
        P_mats.append(P)

        angles = 0.1 * np.random.randn(m)
        angles_list.append(angles)

    params = list(mus) + list(angles_list)
    optimizer = _Adam(params, lr=lr)

    best_loss = float('inf')
    no_improve = 0
    final_iter = 0
    loss_history = []

    for iteration in range(max_iter):
        all_mus = [mu0] + mus
        all_sigmas = [sigma0] + [
            _sigma_from_rotation(sigma0, P_mats[i], angles_list[i], dim)
            for i in range(n_modes - 1)
        ]

        loss = 0.0
        all_accepted = True
        grad_mus = [np.zeros(dim) for _ in range(n_modes - 1)]
        grad_angles = [np.zeros(m) for _ in range(n_modes - 1)]

        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                skl_ij = symmetric_kullback_leibler(
                    (all_mus[i], all_sigmas[i]),
                    (all_mus[j], all_sigmas[j])
                )
                if skl_ij < accept_lo or skl_ij > accept_hi:
                    all_accepted = False

                residual = _dead_zone_residual(skl_ij, skl_lo, skl_hi)
                loss += residual ** 2

                if residual == 0.0:
                    continue

                if i > 0:
                    g_mu_i, g_sigma_i = _skl_gradients(
                        all_mus[i], all_sigmas[i],
                        all_mus[j], all_sigmas[j])
                    grad_mus[i - 1] += 2 * residual * g_mu_i
                    grad_angles[i - 1] += _grad_angles_from_grad_sigma(
                        2 * residual * g_sigma_i,
                        sigma0, P_mats[i - 1], angles_list[i - 1], dim)

                if j > 0:
                    g_mu_j, g_sigma_j = _skl_gradients(
                        all_mus[j], all_sigmas[j],
                        all_mus[i], all_sigmas[i])
                    grad_mus[j - 1] += 2 * residual * g_mu_j
                    grad_angles[j - 1] += _grad_angles_from_grad_sigma(
                        2 * residual * g_sigma_j,
                        sigma0, P_mats[j - 1], angles_list[j - 1], dim)

        grads = list(grad_mus) + list(grad_angles)
        optimizer.step(grads)

        # Project means back into the feasible box [MU_LO, MU_HI].
        for k in range(len(mus)):
            np.clip(mus[k], MU_LO, MU_HI, out=mus[k])

        loss_history.append(float(loss))

        if verbose > 0 and (iteration + 1) % verbose == 0:
            print(f"      iter {iteration+1:>6}/{max_iter}  "
                  f"loss={loss:>12.6f}  "
                  f"best={best_loss:>12.6f}  "
                  f"lr={optimizer.lr:.6f}  "
                  f"no_improve={no_improve}/{patience}", flush=True)

        final_iter = iteration + 1

        if all_accepted:
            best_loss = min(best_loss, loss)
            if verbose > 0:
                print(f"      iter {iteration+1:>6}/{max_iter}  "
                      f"loss={loss:.2e} — all pairs in accept range, stopping.",
                      flush=True)
            break

        if best_loss > 0 and (best_loss - loss) < best_loss * 1e-4:
            no_improve += 1
            if loss < best_loss:
                best_loss = loss
            if no_improve >= patience:
                if verbose > 0:
                    print(f"      iter {iteration+1:>6}/{max_iter}  "
                          f"patience exhausted (no >0.01% improvement "
                          f"for {patience} iters), stopping.", flush=True)
                break
        else:
            best_loss = loss
            no_improve = 0

    # Build result
    all_mus_final = [mu0] + mus
    all_sigmas_final = [sigma0] + [
        _sigma_from_rotation(sigma0, P_mats[i], angles_list[i], dim)
        for i in range(n_modes - 1)
    ]

    constellation = [(mu.copy(), sigma.copy())
                     for mu, sigma in zip(all_mus_final, all_sigmas_final)]

    skl_matrix = np.zeros((n_modes, n_modes))
    all_in_range = True
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            s = symmetric_kullback_leibler(constellation[i], constellation[j])
            skl_matrix[i, j] = s
            skl_matrix[j, i] = s
            if s < accept_lo or s > accept_hi:
                all_in_range = False

    return {
        'constellation': constellation,
        'skl_matrix': skl_matrix,
        'loss': best_loss,
        'iterations': final_iter,
        'loss_history': loss_history,
        'all_in_range': all_in_range,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── Default alias ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# The default ``generate_constellation`` is the point-target rotation-OGD: it
# pushes every pair towards a single sKL value, so the resulting constellation
# is close to the target on average (precise, less approximated). On hard
# problems (low dim, many modes) a few outlier pairs may sit slightly above
# the target — use the range variant below if a strict band is required.
generate_constellation = generate_constellation_gd_rotation

# ``generate_constellation_range`` is the rotation-parameterised GD with
# dead-zone loss: it accepts an [lo, hi] band rather than a scalar target, so
# pairs already inside the band are not pushed further. Useful when a strict
# range is needed and small approximation around the midpoint is acceptable.
generate_constellation_range = generate_constellation_gd_rotation_range
