"""
ccm_nova.synthetic.utils
===================
Math/statistics utilities for synthetic data generation.

Core concept: a Gaussian distribution is represented throughout this file as a
2-tuple  (mu, sigma)  where:
  - mu    is a 1-D numpy array of shape (dim,)   — the mean vector
  - sigma is a 2-D numpy array of shape (dim,dim) — the covariance matrix

The main purpose of this module is to construct a "post-change" Gaussian that is
exactly a desired distance (measured in symmetric Kullback-Leibler divergence,
sKL) away from a "pre-change" Gaussian.  This lets SyntheticDataset produce
data streams where the statistical difficulty of detecting a change is controlled
precisely.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# ── Core Gaussian-distribution math ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def random_gaussian(dim, max_std=0.9):
    """Create a random Gaussian distribution with mean 0 and a random PSD covariance.

    The covariance matrix is constructed so that it is:
      - Positive semi-definite (a valid covariance matrix).
      - Non-degenerate (smallest eigenvalue strictly positive).
      - Random orientation (achieved via QR decomposition of a random matrix).
      - Spectrum-bounded: largest eigenvalue equals ``max_std**2``, so the
        principal-axis std along every direction is at most ``max_std``. With
        means living in the auction's [0,1] box, the default keeps samples
        well inside the box and clipping rare.

    Parameters
    ----------
    dim : int
        Dimension of the Gaussian.
    max_std : float
        Upper bound on the principal-axis standard deviation. The eigenvalue
        spectrum is rescaled so its maximum is exactly ``max_std**2``.

    Returns
    -------
    mu0    : zero vector of shape (dim,)
    sigma0 : random PSD covariance matrix of shape (dim, dim)
    """
    mu0 = np.zeros(dim)

    A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
    Q, _ = np.linalg.qr(A)  # random orthogonal matrix

    raw = 2 * np.abs(np.random.normal(0, 1, dim)) + 0.01  # random positive eigenvalues
    eigvals = raw * (max_std ** 2) / raw.max()            # rescale so max == max_std**2
    D = np.diag(eigvals)

    sigma0 = Q @ D @ Q.T
    return mu0, sigma0


def manipulate_gaussian(gauss, Q, shift):
    """Apply a rotation and translation to a Gaussian.

    Given a Gaussian N(mu, Sigma), applying the affine map  x -> R x + t  gives:
      new mean  : R^T (mu - t)          (inverse map applied to the mean)
      new cov   : R^T Sigma R

    Parameters
    ----------
    gauss : (mu, sigma) tuple of the original Gaussian.
    Q     : Rotation matrix R of shape (dim, dim).
    shift : Translation vector t of shape (dim,).

    Returns
    -------
    gauss1 : [mu1, sigma1] of the transformed Gaussian.
    """
    muo = gauss[0] # mean
    sigmao = gauss[1] # cov matrix

    mu1 = np.dot(Q.T, muo - shift) #also mean gets rotated.
    sigma1 = np.dot(np.dot(Q.T, sigmao), Q)

    gauss1 = [mu1, sigma1]
    return gauss1


def init_changeMagnitude(gauss0, target_skl=1):
    """Find rotation/shift parameters whose sKL is >= target_skl.

    This is the *initialization* phase of a two-phase search.  We need to
    find upper bounds for the rotation angles and the shift magnitude so that
    a subsequent binary search (convergeChangeMagnitude) can home in on the
    exact target sKL.

    Rotation parameterisation
    -------------------------
    A d-dimensional rotation is built as   Q = P @ T(angles) @ P^T   where:
      * P  is a random orthogonal matrix (drawn from the Haar measure via QR factorization),
            embedding the rotation into an arbitrary subspace of ℝ^d.
      * T(angles) is a block-diagonal Givens matrix built from m = ⌊d/2⌋
            independent plane rotations with angles θ_1 … θ_m.

    The angle vector ``angles`` is constructed as a **random unit vector on
    the (m-1)-sphere scaled to π/2**:

        raw    ~ N(0, I_m)
        angles = (π/2) * raw / ‖raw‖

    This separates two independent sources of randomness:
      * *Direction* — which combination of planes to rotate in (uniform on
        the unit sphere; any angle vector direction is equally likely).
      * *Magnitude* — always π/2 at initialisation, ensuring the binary
        search always has access to the maximum sKL achievable by rotation
        alone.  The binary search (convergeChangeMagnitude) then scales the
        vector back toward zero until the target sKL is reached.

    Using a fixed upper-bound magnitude (π/2) rather than a random one
    guarantees that the binary search always explores the full monotone
    region [0, π/2] of sKL(θ) = C · sin²θ, independent of the random seed.

    Parameters
    ----------
    gauss0     : (mu, sigma) of the reference (pre-change) Gaussian.
    target_skl : Desired symmetric KL divergence between pre/post distributions.

    Returns
    -------
    angles    : Array of rotation angles (upper bound for binary search).
                Shape (m,) where m = ⌊dim/2⌋ (or 1 when dim=1).
    P         : Orthogonal matrix specifying the planes of rotation of Q. To have more info on how to evaluate
    Q starting from P generated at random and the angles read the paper
    rho       : Shift magnitude that achieves sKL >= target_skl (upper bound).
    dir_shift : Unit vector giving the direction of the shift.
    """
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    dim = len(mu0)

    dir_shift = 2 * np.random.random(dim) - 1 # To have a vector of random number between -1 and 1.
    dir_shift = dir_shift / np.linalg.norm(dir_shift) # To have a unitary vector.

    rho = 1

    m = 1 if dim == 1 else int(np.floor(dim / 2)) # Compute the number of angles we have based on dimension.

    # Random unit vector on the (m-1)-sphere, scaled to π/2.
    # For m=1 this reduces to ±π/2 (random sign); for m>1 the direction is
    # uniform over the sphere so every combination of plane rotations is
    # equally likely.

    raw    = np.random.randn(m)
    angles = (np.pi / 2) * (raw / np.linalg.norm(raw))
    # Cap magnitude at π/2: sKL ∝ sin²θ is monotone on [0, π/2], guaranteeing binary-search convergence.
    # P covers all plane orientations; the sign of each angle covers rotation direction within each plane.

    A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim) # generate random covariance function with mean 0 and sigma^2=1
    P, _ = np.linalg.qr(A) # perform qr factorization on a random matrix to make sure to keep just orthogonal matrices.

    T_ang = generate_canonical_rotation(angles, dim) # to know more on how T is genearted read paper or the function below
    Q = np.dot(np.dot(P, T_ang), P.T)

    shift = rho * dir_shift # this is the dhift of mean evaluated as a multiplicative factor rho times the random unit vector.
    gauss1 = manipulate_gaussian(gauss0, Q, shift)

    skl = 0
    while skl < target_skl:
        skl = symmetric_kullback_leibler(gauss0, gauss1)
        if skl < target_skl:
            rho = 2 * rho
            shift = rho * dir_shift
            gauss1 = manipulate_gaussian(gauss0, Q, shift)

    return angles, P, rho, dir_shift


def convergeChangeMagnitude(gauss0, target_skl, angles, P, rho, dir_shift, eps=1e-6):
    """Binary-search the rotation/shift parameters to hit exactly target_skl.

    Parameters
    ----------
    gauss0     : (mu, sigma) pre-change Gaussian.
    target_skl : Desired sKL value.
    angles     : Upper-bound rotation angles (from init_changeMagnitude).
    P          : Orthogonal embedding matrix (from init_changeMagnitude).
    rho        : Upper-bound shift magnitude (from init_changeMagnitude).
    dir_shift  : Unit direction vector (from init_changeMagnitude).
    eps        : Convergence tolerance on |sKL - target_skl|.

    Returns
    -------
    gauss1 : (mu1, sigma1) of the post-change Gaussian with sKL ~= target_skl.
    """
    dim = len(P)

    theta_lower = np.zeros_like(angles)
    rho_lower   = 0
    theta_upper = angles
    rho_upper   = rho

    skl = 0
    while np.linalg.norm(skl - target_skl) > eps:
        angles_ = (theta_lower + theta_upper) / 2
        rho_    = np.mean([rho_lower, rho_upper])

        T_ang  = generate_canonical_rotation(angles_, dim)
        Q    = np.dot(np.dot(P, T_ang), P.T)
        shift  = rho_ * dir_shift

        gauss1 = manipulate_gaussian(gauss0, Q, shift)
        skl    = symmetric_kullback_leibler(gauss0, gauss1)

        if skl < target_skl:
            theta_lower = angles_
            rho_lower   = rho_
        else:
            theta_upper = angles_
            rho_upper   = rho_

    return gauss1


def generate_canonical_rotation(angles, dim):
    """Build a block-diagonal rotation matrix from a list of 2-D plane angles.

    In dim-D space, a general rotation can be decomposed into at most floor(dim/2)
    independent planar rotations.

    Parameters
    ----------
    angles : 1-D array of rotation angles (radians), length = floor(dim/2).
    dim    : Dimension of the space.

    Returns
    -------
    T : Block-diagonal Givens rotation matrix T(θ) of shape (dim, dim).
    """
    T = np.identity(dim)

    if dim == 1:
        return T

    for i_angles in range(len(angles)):
        theta = angles[i_angles]

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        T[2 * i_angles: 2 * (i_angles + 1),
          2 * i_angles: 2 * (i_angles + 1)] = R # way to set the Rotation angles for each index

    return T


# ═══════════════════════════════════════════════════════════════════════════════
# ── Kullback-Leibler divergence ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def kullback_leibler(gauss0, gauss1):
    """Compute the KL divergence KL(gauss0 || gauss1) for two Gaussians.

    Closed-form formula for two multivariate Gaussians N(mu0, Sigma0) and N(mu1, Sigma1):

        KL = 0.5 * [ tr(Sigma1^-1 Sigma0)
                     + (mu1-mu0)^T Sigma1^-1 (mu1-mu0)
                     - dim
                     + ln(det Sigma1 / det Sigma0) ]

    Parameters
    ----------
    gauss0, gauss1 : (mu, sigma) tuples.

    Returns
    -------
    Scalar KL divergence value (always >= 0).
    """
    mu0    = gauss0[0];  sigma0 = gauss0[1]
    mu1    = gauss1[0];  sigma1 = gauss1[1]
    dim    = len(mu0)

    return 0.5 * (
        np.trace(np.linalg.solve(sigma1, sigma0))
        + np.dot(mu1 - mu0, np.linalg.solve(sigma1, mu1 - mu0))
        - dim
        + np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0))
    )
    # np.solve is industry standard when you want to use np.linalg.inv(sigma1) @ sigma0 for numerical reasons.
    # but the 2 command are the same except for speed and stability.

def symmetric_kullback_leibler(gauss0, gauss1):
    """Compute the *symmetric* KL divergence (sKL) between two Gaussians.

    sKL(p, q) = KL(p||q) + KL(q||p)

    A larger sKL means the two distributions are further apart and therefore
    easier to tell apart.

    Parameters
    ----------
    gauss0, gauss1 : (mu, sigma) tuples.

    Returns
    -------
    Scalar symmetric KL divergence (always >= 0).
    """
    return kullback_leibler(gauss0, gauss1) + kullback_leibler(gauss1, gauss0)


def convergeChangeMagnitude_withParams(gauss0, target_skl, angles, P, rho, dir_shift, eps=1e-6):
    """Same as convergeChangeMagnitude but also returns the converged angles and rho.

    Parameters
    ----------
    gauss0     : (mu, sigma) pre-change Gaussian.
    target_skl : Desired sKL value.
    angles     : Upper-bound rotation angles (from init_changeMagnitude).
    P          : Orthogonal embedding matrix (from init_changeMagnitude).
    rho        : Upper-bound shift magnitude (from init_changeMagnitude).
    dir_shift  : Unit direction vector (from init_changeMagnitude).
    eps        : Convergence tolerance on |sKL - target_skl|.

    Returns
    -------
    gauss1       : (mu1, sigma1) post-change Gaussian with sKL ~= target_skl.
    angles_final : Converged rotation angles (subset of the original upper-bound).
    rho_final    : Converged shift magnitude.
    """
    dim = len(P)

    theta_lower = np.zeros_like(angles)
    rho_lower   = 0
    theta_upper = angles.copy()
    rho_upper   = rho

    angles_ = angles.copy()
    rho_    = rho

    skl = 0
    while np.linalg.norm(skl - target_skl) > eps:
        angles_ = (theta_lower + theta_upper) / 2
        rho_    = np.mean([rho_lower, rho_upper])

        T_ang  = generate_canonical_rotation(angles_, dim)
        Q    = np.dot(np.dot(P, T_ang), P.T)
        shift  = rho_ * dir_shift

        gauss1 = manipulate_gaussian(gauss0, Q, shift)
        skl    = symmetric_kullback_leibler(gauss0, gauss1)

        if skl < target_skl:
            theta_lower = angles_
            rho_lower   = rho_
        else:
            theta_upper = angles_
            rho_upper   = rho_

    return gauss1, angles_, rho_

# Function needed for generating incremental CD
def alpha_for_skl(gauss0, angles_full, P, rho_full, dir_shift, target_skl_k,
                  eps=1e-4):
    """Find the fraction alpha in [0, 1] such that sKL(gauss0, gauss_alpha) = target_skl_k.

    Uses binary search on alpha, which is valid because sKL is monotonically
    non-decreasing as alpha increases from 0 to 1 (the transformation moves the
    distribution further from gauss0 at every step).

    Parameters
    ----------
    gauss0       : (mu, sigma) pre-change Gaussian.
    angles_full  : Converged rotation angles for the full transformation.
    P            : Orthogonal embedding matrix.
    rho_full     : Converged shift magnitude for the full transformation.
    dir_shift    : Unit direction vector of the shift.
    target_skl_k : Desired sKL value for this step (in [0, target_skl]).
    eps          : Convergence tolerance on |sKL - target_skl_k|.

    Returns
    -------
    alpha : float in [0, 1] — the fraction of the rototranslation to apply to get
    targetskl starting from the original Gaussian.
    """
    lo, hi = 0.0, 1.0

    for _ in range(64):  # 64 iterations → precision ~5e-20, well beyond any eps
        mid      = (lo + hi) / 2.0
        gauss_m  = gaussian_at_fraction(gauss0, angles_full, P, rho_full, dir_shift,
                                        mid)
        skl_mid  = symmetric_kullback_leibler(gauss0, gauss_m)

        if abs(skl_mid - target_skl_k) < eps:
            return mid

        if skl_mid < target_skl_k:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0

# Utility function for the gradual CD.
# answers the question: at this point in the transition window,
# how likely is the next sample to come from the new (out-of-control)
# distribution X2?
def mixing_weight(frac, drift_type='linear'):
    """Compute P(X2) at normalized position frac within the transition window.

    Parameters
    ----------
    frac       : Float in [0, 1] — 0 = start of window, 1 = end of window.
    drift_type : 'linear' or 'sigmoidal'.

    Returns
    -------
    w : float in [0, 1] — probability of drawing the next sample from X2.
    """
    if drift_type == 'linear':
        return float(frac)
    elif drift_type == 'sigmoidal':
        # Normalised sigmoid: S-shaped curve that passes through exactly (0,0) and (1,1).
        steepness = 10.0
        x     = steepness * (frac - 0.5)
        x_min = steepness * (0.0  - 0.5)
        x_max = steepness * (1.0  - 0.5)
        raw   = 1.0 / (1.0 + np.exp(-x))
        lo    = 1.0 / (1.0 + np.exp(-x_min))
        hi    = 1.0 / (1.0 + np.exp(-x_max))
        return (raw - lo) / (hi - lo)
    else:
        raise ValueError(
            f"Unknown drift_type '{drift_type}'. Choose 'linear' or 'sigmoidal'.")


def gaussian_at_fraction(gauss0, angles_full, P, rho_full, dir_shift, alpha):
    """Compute the Gaussian obtained by applying a fraction alpha of the full rototranslation.

    Linearly scales both the rotation angles and the shift magnitude by alpha,
    so alpha=0 recovers gauss0 and alpha=1 recovers the fully transformed gauss1.

    Parameters
    ----------
    gauss0      : (mu, sigma) pre-change Gaussian.
    angles_full : Converged rotation angles for the full transformation.
    P           : Orthogonal embedding matrix.
    rho_full    : Converged shift magnitude for the full transformation.
    dir_shift   : Unit direction vector of the shift.
    alpha       : Fraction in [0, 1] — how much of the transformation to apply.

    Returns
    -------
    gauss_alpha : (mu, sigma) Gaussian at fraction alpha of the full rototranslation.
    """
    dim          = len(gauss0[0])
    angles_alpha = alpha * angles_full
    rho_alpha    = alpha * rho_full

    T_ang = generate_canonical_rotation(angles_alpha, dim)
    Q   = np.dot(np.dot(P, T_ang), P.T)
    shift = rho_alpha * dir_shift

    return manipulate_gaussian(gauss0, Q, shift)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Variance-shift variant (CCM with diagonal scaling matrix D) ──────────────
# ═══════════════════════════════════════════════════════════════════════════════

def manipulate_gaussian_variance(gauss, Q, shift, D_diag):
    """Apply rotation, translation, AND diagonal scaling to a Gaussian.

    Given a Gaussian N(mu, Sigma), applying the map  x -> D Q^T (x - v)  gives:
      new mean  : D Q^T (mu - v)
      new cov   : D Q^T Sigma Q D

    Parameters
    ----------
    gauss  : (mu, sigma) tuple of the original Gaussian.
    Q      : Rotation matrix of shape (dim, dim).
    shift  : Translation vector of shape (dim,).
    D_diag : 1-D array of diagonal entries of the scaling matrix D.

    Returns
    -------
    gauss1 : [mu1, sigma1] of the transformed Gaussian.
    """
    mu0 = gauss[0]
    sigma0 = gauss[1]
    D = np.diag(D_diag)

    mu1 = D @ Q.T @ (mu0 - shift)
    sigma1 = D @ Q.T @ sigma0 @ Q @ D

    return [mu1, sigma1]


def init_changeMagnitude_variance(gauss0, target_skl=1, variance_sigma=1.0,
                                  variance_weight=0.5):
    """Find rotation/shift/scaling parameters whose sKL >= target_skl.

    Like init_changeMagnitude, but also introduces a diagonal scaling matrix
    D = diag(exp(lambda * delta)) that changes the variance of the post-change
    distribution, breaking trace invariance.

    Parameters
    ----------
    gauss0          : (mu, sigma) of the reference (pre-change) Gaussian.
    target_skl      : Desired symmetric KL divergence.
    variance_sigma  : Std deviation for the random delta vector (controls how
                      "directional" the variance shift is).
    variance_weight : Float in [0, 1]. Controls the relative contribution of
                      variance shift vs mean shift. 0 = no variance change
                      (degenerates to original CCM), 1 = only variance change.

    Returns
    -------
    angles    : Upper-bound rotation angles.
    P         : Orthogonal embedding matrix.
    rho       : Upper-bound shift magnitude.
    dir_shift : Unit direction vector for the shift.
    lam       : Upper-bound lambda (scaling intensity).
    delta_var : Random direction vector for the variance shift.
    """
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    dim = len(mu0)

    dir_shift = 2 * np.random.random(dim) - 1
    dir_shift = dir_shift / np.linalg.norm(dir_shift)

    rho = 1.0
    lam = 1.0

    # Random variance-shift direction
    delta_var = np.random.normal(0, variance_sigma, dim)

    m = 1 if dim == 1 else int(np.floor(dim / 2))

    raw = np.random.randn(m)
    angles = (np.pi / 2) * (raw / np.linalg.norm(raw))

    A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
    P, _ = np.linalg.qr(A)

    T_ang = generate_canonical_rotation(angles, dim)
    Q = np.dot(np.dot(P, T_ang), P.T)

    # Clamp variance_weight to avoid division by zero at extremes
    w = np.clip(variance_weight, 1e-6, 1 - 1e-6)

    shift = rho * dir_shift
    D_diag = np.exp(lam * delta_var)
    gauss1 = manipulate_gaussian_variance(gauss0, Q, shift, D_diag)

    skl = 0
    while skl < target_skl:
        skl = symmetric_kullback_leibler(gauss0, gauss1)
        if skl < target_skl:
            # Grow rho and lam at rates controlled by variance_weight.
            # w close to 1 -> lam grows fast, rho grows slow (variance-dominated)
            # w close to 0 -> rho grows fast, lam grows slow (mean-dominated)
            rho = rho * (1 + 2 * (1 - w))
            lam = lam * (1 + 2 * w)
            shift = rho * dir_shift
            D_diag = np.exp(lam * delta_var)
            gauss1 = manipulate_gaussian_variance(gauss0, Q, shift, D_diag)

    return angles, P, rho, dir_shift, lam, delta_var


def convergeChangeMagnitude_variance(gauss0, target_skl, angles, P, rho,
                                     dir_shift, lam, delta_var,
                                     variance_weight=0.5, eps=1e-6):
    """Binary-search rotation/shift/scaling parameters to hit exactly target_skl.

    Searches over a single scalar t in [0, 1] that parameterises the path from
    gauss0 (t=0) to the overshooting transformation (t=1).  The three physical
    parameters are derived from t using variance_weight (w):

        angles(t) = t * angles_upper
        rho(t)    = t^(1/max(1-w, eps)) * rho_upper   (grows slowly when w~1)
        lam(t)    = t^(1/max(w, eps))   * lam_upper   (grows slowly when w~0)

    At w=0.5 all three grow linearly (standard bisection).
    At w->1  rho barely moves while lambda reaches its upper bound quickly.
    At w->0  lambda barely moves while rho reaches its upper bound quickly.

    Parameters
    ----------
    gauss0          : (mu, sigma) pre-change Gaussian.
    target_skl      : Desired sKL value.
    angles          : Upper-bound rotation angles.
    P               : Orthogonal embedding matrix.
    rho             : Upper-bound shift magnitude.
    dir_shift       : Unit direction vector.
    lam             : Upper-bound lambda (scaling intensity).
    delta_var       : Direction vector for variance shift.
    variance_weight : Float in [0, 1].
    eps             : Convergence tolerance.

    Returns
    -------
    gauss1 : (mu1, sigma1) post-change Gaussian with sKL ~= target_skl.
    """
    dim = len(P)
    w = np.clip(variance_weight, 1e-6, 1 - 1e-6)
    rho_exp = 1.0 / (1.0 - w)   # exponent for rho:  large when w->1
    lam_exp = 1.0 / w            # exponent for lam:  large when w->0

    t_lo = 0.0
    t_hi = 1.0

    skl = 0
    while np.abs(skl - target_skl) > eps:
        t = (t_lo + t_hi) / 2

        angles_ = t * angles
        rho_ = (t ** rho_exp) * rho
        lam_ = (t ** lam_exp) * lam

        T_ang = generate_canonical_rotation(angles_, dim)
        Q = np.dot(np.dot(P, T_ang), P.T)
        shift = rho_ * dir_shift
        D_diag = np.exp(lam_ * delta_var)

        gauss1 = manipulate_gaussian_variance(gauss0, Q, shift, D_diag)
        skl = symmetric_kullback_leibler(gauss0, gauss1)

        if skl < target_skl:
            t_lo = t
        else:
            t_hi = t

    return gauss1


def convergeChangeMagnitude_variance_withParams(gauss0, target_skl, angles, P,
                                                rho, dir_shift, lam, delta_var,
                                                variance_weight=0.5, eps=1e-6):
    """Same as convergeChangeMagnitude_variance but also returns converged params.

    Returns
    -------
    gauss1     : (mu1, sigma1) post-change Gaussian.
    angles_    : Converged rotation angles.
    rho_       : Converged shift magnitude.
    lam_       : Converged lambda (scaling intensity).
    """
    dim = len(P)
    w = np.clip(variance_weight, 1e-6, 1 - 1e-6)
    rho_exp = 1.0 / (1.0 - w)
    lam_exp = 1.0 / w

    t_lo = 0.0
    t_hi = 1.0

    angles_ = angles.copy()
    rho_ = rho
    lam_ = lam

    skl = 0
    while np.abs(skl - target_skl) > eps:
        t = (t_lo + t_hi) / 2

        angles_ = t * angles
        rho_ = (t ** rho_exp) * rho
        lam_ = (t ** lam_exp) * lam

        T_ang = generate_canonical_rotation(angles_, dim)
        Q = np.dot(np.dot(P, T_ang), P.T)
        shift = rho_ * dir_shift
        D_diag = np.exp(lam_ * delta_var)

        gauss1 = manipulate_gaussian_variance(gauss0, Q, shift, D_diag)
        skl = symmetric_kullback_leibler(gauss0, gauss1)

        if skl < target_skl:
            t_lo = t
        else:
            t_hi = t

    return gauss1, angles_, rho_, lam_


def gaussian_at_fraction_variance(gauss0, angles_full, P, rho_full, dir_shift,
                                  lam_full, delta_var, alpha):
    """Compute the Gaussian at fraction alpha of the full rototranslation+scaling.

    Linearly scales rotation angles, shift magnitude, AND lambda by alpha,
    so alpha=0 recovers gauss0 and alpha=1 recovers the fully transformed gauss1.

    Parameters
    ----------
    gauss0      : (mu, sigma) pre-change Gaussian.
    angles_full : Converged rotation angles for the full transformation.
    P           : Orthogonal embedding matrix.
    rho_full    : Converged shift magnitude.
    dir_shift   : Unit direction vector of the shift.
    lam_full    : Converged lambda (scaling intensity).
    delta_var   : Direction vector for variance shift.
    alpha       : Fraction in [0, 1].

    Returns
    -------
    gauss_alpha : (mu, sigma) Gaussian at fraction alpha.
    """
    dim = len(gauss0[0])
    angles_alpha = alpha * angles_full
    rho_alpha = alpha * rho_full
    lam_alpha = alpha * lam_full

    T_ang = generate_canonical_rotation(angles_alpha, dim)
    Q = np.dot(np.dot(P, T_ang), P.T)
    shift = rho_alpha * dir_shift
    D_diag = np.exp(lam_alpha * delta_var)

    return manipulate_gaussian_variance(gauss0, Q, shift, D_diag)


def alpha_for_skl_variance(gauss0, angles_full, P, rho_full, dir_shift,
                           lam_full, delta_var, target_skl_k, eps=1e-4):
    """Find alpha in [0, 1] such that sKL(gauss0, gauss_alpha) = target_skl_k.

    Variance-shift variant of alpha_for_skl.

    Parameters
    ----------
    gauss0       : (mu, sigma) pre-change Gaussian.
    angles_full  : Converged rotation angles.
    P            : Orthogonal embedding matrix.
    rho_full     : Converged shift magnitude.
    dir_shift    : Unit direction vector.
    lam_full     : Converged lambda.
    delta_var    : Direction vector for variance shift.
    target_skl_k : Desired sKL value.
    eps          : Convergence tolerance.

    Returns
    -------
    alpha : float in [0, 1].
    """
    lo, hi = 0.0, 1.0

    for _ in range(64):
        mid = (lo + hi) / 2.0
        gauss_m = gaussian_at_fraction_variance(
            gauss0, angles_full, P, rho_full, dir_shift, lam_full, delta_var, mid)
        skl_mid = symmetric_kullback_leibler(gauss0, gauss_m)

        if abs(skl_mid - target_skl_k) < eps:
            return mid

        if skl_mid < target_skl_k:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0
