"""
mp_theory.py — Marčenko-Pastur law and BBP phase transition.

Theoretical foundations
-----------------------
All functions in this module implement results from the following papers.
Equation numbers and theorem references are given in each docstring.

  [MP67]  Marčenko, Pastur (1967). "Distribution of eigenvalues for some
          sets of random matrices." Math. USSR-Sbornik 1(4), 457–483.
          The foundational theorem: empirical spectral distribution of
          (1/T)X^T X converges to a deterministic law as d,T→∞ with
          d/T→γ. See Theorem 1 and Example 1 (pp. 461–462).

  [L99]   Laloux, Cizeau, Bouchaud, Potters (1999). "Noise Dressing of
          Financial Correlation Matrices." PRL 83(7), 1467–1470.
          First financial application. Equation (3) gives the density
          in the form used here.

  [P02]   Plerou, Gopikrishnan, Rosenow, Amaral, Guhr, Stanley (2002).
          "Random matrix approach to cross correlations in financial data."
          Phys. Rev. E 65, 066126. Rolling window methodology.

  [BBP05] Baik, Ben Arous, Péché (2005). "Phase transition of the largest
          eigenvalue for nonnull complex sample covariance matrices."
          Ann. Prob. 33(5), 1643–1697. arXiv: math/0403022.
          Theorem 1.1 and Corollary 1.1.

  [LW04]  Ledoit, Wolf (2004). "A well-conditioned estimator for
          large-dimensional covariance matrices."
          J. Multivariate Analysis 88(2), 365–411.

  [BBP17] Bun, Bouchaud, Potters (2017). "Cleaning large correlation
          matrices: Tools from Random Matrix Theory."
          Physics Reports 666, 1–109.

Mathematical flag 1 (BBP threshold vs. bulk edge)
-------------------------------------------------
These are DISTINCT quantities and must not be confused:

  BBP detection threshold:  θ_c = σ²(1 + √γ)
  MP bulk upper edge:        λ+  = σ²(1 + √γ)²

The threshold lives below the bulk edge. A population spike at exactly
θ_c sits *below* λ+ and is provably absorbed into the bulk. A population
spike at θ > θ_c produces a sample eigenvalue strictly *above* λ+. The
two quantities coincide only in the degenerate limit γ → 0. See the
function verify_bbp_distinction() for a numerical demonstration, and
Corollary 1.1 of [BBP05] for the proof.

Mathematical flag 2 (BBP is proven for complex Gaussian samples)
----------------------------------------------------------------
Theorem 1.1 of [BBP05] is proven for complex Gaussian samples (GUE-type
Wishart matrices). For real sample covariance matrices (the financially
relevant case), the same critical value 1 + √γ and the same spike location
formula are conjectured in Section 1.3 of [BBP05] and supported by
subsequent empirical and theoretical work, but the full proof for the real
case was not established in the 2005 paper. Results that use the BBP spike
location formula on real financial data should be understood as applying a
result proven for complex samples.

Mathematical flag 3 (CUSUM recursive form vs. Page's original)
--------------------------------------------------------------
Page (1954) Rule 1 (eq. 5, p. 103) states:
  "Take action if S_n - min_{0 ≤ i ≤ n} S_i > h"
The recursive form C(t) = max(0, C(t-1) + (x(t) - k)) that appears in
changepoint.py is algebraically equivalent but was not stated in this form
in the original paper. The recursive formulation comes from subsequent
statistical process control literature. Both forms are cited in changepoint.py.
"""

import warnings
import numpy as np
from scipy import integrate, stats


# ---------------------------------------------------------------------------
# Bulk edge formulas
# ---------------------------------------------------------------------------

def bulk_edges(gamma: float, sigma2: float) -> tuple[float, float]:
    """
    Compute the lower and upper edges of the Marčenko-Pastur bulk spectrum.

    Theorem [MP67, Example 1, p. 462]:
      λ± = σ²(1 ± √γ)²

    where γ = d/T is the aspect ratio (dimensions / observations) and σ² is
    the variance of the population noise floor.

    Assumptions
    -----------
    - d, T → ∞ with d/T → γ ∈ (0, ∞) fixed.
    - Matrix entries are i.i.d. with mean 0, variance σ², finite fourth moment.
    - The population covariance equals σ²·I (pure noise; no spikes).
    - For γ > 1, a point mass of weight (1 - 1/γ) sits at 0; the formula
      still gives the correct continuous-spectrum support.

    Limitations
    -----------
    - Financial returns are NOT i.i.d.: they exhibit serial correlation,
      fat tails, and volatility clustering. Each violation introduces a bias
      in the estimated bulk edges of unknown sign and magnitude.
    - For γ close to 1, λ- ≈ 0; numerical noise in eigenvalue computation
      can produce spurious sub-threshold eigenvalues.
    - σ² must be estimated self-consistently (see estimate_sigma2_self_consistent);
      using the naive mean of all eigenvalues inflates σ² when spikes are present.

    Parameters
    ----------
    gamma : float
        Aspect ratio γ = d/T. Must be > 0.
    sigma2 : float
        Noise floor variance σ². Must be > 0.

    Returns
    -------
    (lambda_minus, lambda_plus) : tuple of float
        Lower and upper bulk edges.

    Examples
    --------
    >>> bulk_edges(0.25, 1.0)   # λ- = (1-0.5)² = 0.25, λ+ = (1+0.5)² = 2.25
    (0.25, 2.25)
    >>> bulk_edges(1.0, 1.0)    # λ- = 0, λ+ = 4
    (0.0, 4.0)
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")
    sqrt_gamma = np.sqrt(gamma)
    lambda_minus = sigma2 * (1.0 - sqrt_gamma) ** 2
    lambda_plus  = sigma2 * (1.0 + sqrt_gamma) ** 2
    return float(lambda_minus), float(lambda_plus)


# ---------------------------------------------------------------------------
# MP density and CDF
# ---------------------------------------------------------------------------

def mp_density(x: float, gamma: float, sigma2: float) -> float:
    """
    Marčenko-Pastur probability density at point x.

    Continuous part [MP67, Example 1, p. 462; L99, eq. 3]:

      f_γ(x) = (1 / (2π γ σ² x)) · √((λ+ − x)(x − λ−))
               for x ∈ [λ−, λ+]

    Point mass [MP67, p. 462]:

      If γ > 1, there is an additional point mass of weight (1 − 1/γ) at x = 0.
      This function returns only the continuous density; callers must account
      for the point mass separately when normalizing or computing CDFs.

    Assumptions
    -----------
    Same as bulk_edges(). The density formula is exact only in the limit
    d, T → ∞ with d/T → γ. For finite d, T, the edges are blurred.

    Parameters
    ----------
    x : float
        Point at which to evaluate the density. Returns 0 outside [λ−, λ+].
    gamma : float
        Aspect ratio γ = d/T. Must be > 0.
    sigma2 : float
        Noise floor σ². Must be > 0.

    Returns
    -------
    float
        Density value f_γ(x) ≥ 0.
    """
    lm, lp = bulk_edges(gamma, sigma2)
    if x <= lm or x >= lp:
        return 0.0
    inner = (lp - x) * (x - lm)
    if inner < 0:
        return 0.0
    return np.sqrt(inner) / (2.0 * np.pi * gamma * sigma2 * x)


def mp_cdf(x: float, gamma: float, sigma2: float) -> float:
    """
    Cumulative distribution function of the Marčenko-Pastur law.

    Defined as the integral of the continuous density from λ− to x, plus
    the point mass at 0 when γ > 1:

      F_γ(x) = max(0, 1 − 1/γ) · 1_{x ≥ 0}
              + ∫_{λ−}^{min(x, λ+)} f_γ(t) dt

    The integral is computed via scipy.integrate.quad. This function is
    used to compute the KS distance between the empirical bulk eigenvalue
    distribution and the theoretical MP law.

    Assumptions
    -----------
    Same as bulk_edges(). Numerical integration accuracy is limited by
    integrand singularities at the edges (square-root vanishing); quad
    handles these well in practice but tolerance is not guaranteed to 1e-12.

    Parameters
    ----------
    x : float
        Point at which to evaluate the CDF.
    gamma : float
        Aspect ratio.
    sigma2 : float
        Noise floor.

    Returns
    -------
    float
        CDF value in [0, 1].
    """
    lm, lp = bulk_edges(gamma, sigma2)
    point_mass = max(0.0, 1.0 - 1.0 / gamma)  # weight at 0 when gamma > 1

    if x < 0.0:
        return 0.0
    if 0.0 <= x < lm:
        return point_mass
    if x >= lp:
        return 1.0

    integral, _ = integrate.quad(
        lambda t: mp_density(t, gamma, sigma2),
        lm, x,
        limit=200,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return point_mass + integral


# ---------------------------------------------------------------------------
# BBP phase transition
# ---------------------------------------------------------------------------

def bbp_threshold(gamma: float, sigma2: float) -> float:
    """
    Baik-Ben Arous-Péché phase transition threshold for population eigenvalues.

    A population eigenvalue θ produces a detectable sample eigenvalue
    (separated from the MP bulk) if and only if:

      θ > θ_c  where  θ_c = σ²(1 + √γ)

    [BBP05, Corollary 1.1, p. 1651-1652]:
      "?1 is separated from the rest of the eigenvalues if and only if at
       least one eigenvalue of Σ is greater than 1 + ρ^{-1}"
    In their notation ρ = √(M/N) = 1/√γ, so 1 + ρ^{-1} = 1 + √γ; with
    noise variance σ² the threshold scales as σ²(1 + √γ).

    IMPORTANT — FLAG 1: θ_c ≠ λ+
    --------------------------------
    The BBP detection threshold θ_c = σ²(1 + √γ) is STRICTLY LESS than
    the MP bulk upper edge λ+ = σ²(1 + √γ)². They are equal only at γ = 0.
    A population spike at exactly θ_c is NOT detectable (absorbed into the
    bulk). A detectable spike θ > θ_c produces a sample eigenvalue that
    appears ABOVE λ+. See verify_bbp_distinction() for a numerical example.

    IMPORTANT — FLAG 2: Complex vs. real
    -------------------------------------
    This result is proven for complex Gaussian samples in [BBP05, Theorem 1.1].
    For real sample covariance matrices, the same critical value is conjectured
    [BBP05, Section 1.3, Conjecture, p. 1653] but not proven in that paper.

    Parameters
    ----------
    gamma : float
        Aspect ratio γ = d/T. Must be > 0.
    sigma2 : float
        Noise floor σ². Must be > 0.

    Returns
    -------
    float
        BBP threshold θ_c = σ²(1 + √γ).
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")
    return float(sigma2 * (1.0 + np.sqrt(gamma)))


def bbp_sample_eigenvalue(theta: float, gamma: float, sigma2: float) -> float:
    """
    Predicted sample eigenvalue location for a detectable population spike.

    When a population eigenvalue θ exceeds the BBP threshold θ_c = σ²(1+√γ),
    it produces a sample eigenvalue at:

      λ_spike = θ · (1 + γσ² / (θ − σ²))

    [BBP05, Corollary 1.1(b), eq. (46), p. 1651]:
      In BBP notation with ρ = 1/√γ and unit noise:
        λ_spike → θ(1 + ρ^{-2} / (θ − 1)) = θ(1 + γ/(θ − 1))
      Rescaling for noise variance σ²: replace θ → θ/σ² and multiply by σ².

    This formula is the large-sample limit. For finite d, T the actual spike
    eigenvalue fluctuates around this prediction; the fluctuation scale is
    O(1/√T) [BBP05, Theorem 1.1].

    IMPORTANT — FLAG 2: This formula applies to the conjectured real case.
    See bbp_threshold() docstring.

    Parameters
    ----------
    theta : float
        Population eigenvalue. Must exceed bbp_threshold(gamma, sigma2).
    gamma : float
        Aspect ratio.
    sigma2 : float
        Noise floor.

    Returns
    -------
    float
        Predicted sample eigenvalue λ_spike > λ+.

    Raises
    ------
    ValueError
        If theta ≤ bbp_threshold(gamma, sigma2). Below the threshold the spike
        is absorbed into the bulk and this formula does not apply.
    """
    theta_c = bbp_threshold(gamma, sigma2)
    if theta <= theta_c:
        raise ValueError(
            f"theta={theta:.6f} must exceed the BBP threshold "
            f"theta_c=sigma2*(1+sqrt(gamma))={theta_c:.6f}. "
            f"Below this threshold the spike is absorbed into the bulk "
            f"and is not a detectable separate eigenvalue."
        )
    return float(theta * (1.0 + gamma * sigma2 / (theta - sigma2)))


# ---------------------------------------------------------------------------
# Self-consistent σ² estimator
# ---------------------------------------------------------------------------

def estimate_sigma2_self_consistent(
    eigenvalues: np.ndarray,
    gamma: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple[float, int, bool]:
    """
    Self-consistent iterative estimator of the MP noise floor σ².

    Rationale
    ---------
    The naive estimator σ² = mean(all eigenvalues) is biased upward when
    signal spikes are present: the spike eigenvalues inflate the mean,
    which inflates the estimated bulk edge λ+, which causes the bulk to
    be underestimated. The self-consistent procedure iterates until the
    estimated λ+ is consistent with the eigenvalues used to estimate σ².

    Procedure [BBP17, Section 2, discussion of bulk σ² estimation]:
      Step 0: σ²_0 = mean(all eigenvalues)
      Step k: λ+(σ²_k) = σ²_k · (1 + √γ)²
              σ²_{k+1} = mean of eigenvalues strictly below λ+(σ²_k)
      Repeat until |σ²_{k+1} − σ²_k| < tol.

    This is an approximation, not a closed-form estimator. It converges
    to the true noise floor σ² when the bulk eigenvalues follow MP exactly.
    It can fail (diverge or converge to a wrong value) in the following cases:

      - γ close to 1: λ- ≈ 0, bulk and spike eigenvalues overlap
      - Very many spikes (k ≫ 1): too few eigenvalues remain in the bulk
        after filtering to estimate σ² reliably
      - Very large spikes: the initial σ²_0 is severely inflated, causing
        the first λ+ to exclude most of the bulk

    See [BBP17, Section 2] for theoretical justification and [L99, eq. 3]
    for the standard parameterization.

    Parameters
    ----------
    eigenvalues : np.ndarray
        1-D array of sample eigenvalues (all of them, including spikes).
    gamma : float
        Aspect ratio γ = d/T. Must be > 0.
    tol : float
        Convergence tolerance on σ². Default 1e-6.
    max_iter : int
        Maximum iterations before declaring non-convergence.

    Returns
    -------
    sigma2 : float
        Estimated noise floor.
    n_iter : int
        Number of iterations performed.
    converged : bool
        True if |σ²_new − σ²_old| < tol was achieved within max_iter.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or len(eigenvalues) == 0:
        raise ValueError("eigenvalues must be a non-empty 1-D array")

    sigma2 = float(np.mean(eigenvalues))

    for i in range(max_iter):
        _, lp = bulk_edges(gamma, sigma2)
        bulk = eigenvalues[eigenvalues < lp]
        if len(bulk) == 0:
            warnings.warn(
                "Self-consistent σ² estimator: no eigenvalues below λ+(σ²). "
                "Returning σ² from previous iteration. This can occur when γ "
                "is close to 1 or when spikes dominate. Result is unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            return sigma2, i + 1, False
        sigma2_new = float(np.mean(bulk))
        if abs(sigma2_new - sigma2) < tol:
            return sigma2_new, i + 1, True
        sigma2 = sigma2_new

    warnings.warn(
        f"Self-consistent σ² estimator did not converge in {max_iter} "
        f"iterations (final |Δσ²| may exceed {tol}). Consider increasing "
        f"max_iter or checking that gamma={gamma:.4f} is not near 1.",
        RuntimeWarning,
        stacklevel=2,
    )
    return sigma2, max_iter, False


# ---------------------------------------------------------------------------
# KS distance
# ---------------------------------------------------------------------------

def ks_distance_from_mp(
    bulk_eigenvalues: np.ndarray,
    gamma: float,
    sigma2: float,
) -> float:
    """
    Kolmogorov-Smirnov distance between the empirical CDF of bulk eigenvalues
    and the theoretical MP CDF with parameters (gamma, sigma2).

      KS = sup_x |F_empirical(x) − F_MP(x; σ², γ)|

    This is the primary spectral test statistic used in the rolling estimator.
    Under the null hypothesis of a stationary, pure-noise covariance structure
    with the fitted σ² and γ, KS should fluctuate near a stable baseline.
    A persistent increase indicates the bulk is no longer well-described by
    the MP law — the covariance structure has changed.

    Note on stationarity: the KS statistic is a scalar summary of the bulk
    fit at a single time window. Its time series is NOT i.i.d. under the null
    because consecutive windows overlap and because financial returns exhibit
    volatility clustering. The CUSUM ARL guarantee (see changepoint.py) requires
    stationarity of the input series, which is violated in practice. See the
    README Limitations section.

    Parameters
    ----------
    bulk_eigenvalues : np.ndarray
        1-D array of eigenvalues already filtered to those below λ+(t).
        Caller is responsible for filtering; this function does not re-filter.
    gamma : float
        Aspect ratio γ at the current window.
    sigma2 : float
        Self-consistently estimated noise floor σ² at the current window.

    Returns
    -------
    float
        KS distance in [0, 1].
    """
    bulk_eigenvalues = np.asarray(bulk_eigenvalues, dtype=float)
    if len(bulk_eigenvalues) == 0:
        warnings.warn(
            "ks_distance_from_mp called with empty bulk_eigenvalues. "
            "Returning 1.0 (maximum distance) as a conservative signal.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0

    bulk_sorted = np.sort(bulk_eigenvalues)
    n = len(bulk_sorted)
    empirical_cdf = np.arange(1, n + 1) / n

    # Theoretical CDF evaluated at each sorted bulk eigenvalue
    # We vectorize mp_cdf over the sorted array
    theoretical_cdf = np.array([mp_cdf(x, gamma, sigma2) for x in bulk_sorted])

    # Also check left-of-each-point for the supremum
    empirical_cdf_left = np.arange(0, n) / n
    ks = max(
        np.max(np.abs(empirical_cdf - theoretical_cdf)),
        np.max(np.abs(empirical_cdf_left - theoretical_cdf)),
    )
    return float(ks)


def full_ks_distance_from_mp(
    all_eigenvalues: np.ndarray,
    gamma: float,
    sigma2: float,
) -> float:
    """
    KS distance between the FULL empirical CDF (all d eigenvalues) and the
    theoretical MP CDF with parameters (gamma, sigma2).

      KS_full = sup_x | F_empirical(x; all eigs) - F_MP(x; sigma2, gamma) |

    Unlike ks_distance_from_mp() which operates only on bulk eigenvalues,
    this function includes all eigenvalues — spike and bulk.  The MP CDF
    equals 1 for x >= lambda+, so any eigenvalue above lambda+ creates a
    deficit in the empirical CDF relative to the theoretical CDF.  At
    x = lambda+ this deficit is exactly k/d (the spike fraction).

    Consequently, KS_full ~ k/d during crisis periods (when k/d is large)
    and is dominated by the bulk fit during calm periods.  This statistic
    correctly identifies regime transitions where the spike count surges.

    This is the statistic shown in Figure 4 (ESD vs MP at four dates) and
    printed in the cross-date KS comparison table.

    Parameters
    ----------
    all_eigenvalues : np.ndarray
        All d eigenvalues (spike and bulk), in any order.
    gamma, sigma2 : float
        As for ks_distance_from_mp.

    Returns
    -------
    float
        KS distance in [0, 1].
    """
    eigs = np.asarray(all_eigenvalues, dtype=float)
    if len(eigs) == 0:
        return 1.0

    eigs_sorted = np.sort(eigs)
    n = len(eigs_sorted)

    empirical_cdf      = np.arange(1, n + 1) / n   # right-continuous
    empirical_cdf_left = np.arange(0, n)     / n   # left limit

    theoretical_cdf = np.array([mp_cdf(x, gamma, sigma2) for x in eigs_sorted])

    ks = max(
        float(np.max(np.abs(empirical_cdf      - theoretical_cdf))),
        float(np.max(np.abs(empirical_cdf_left - theoretical_cdf))),
    )
    return ks


# ---------------------------------------------------------------------------
# Living documentation check
# ---------------------------------------------------------------------------

def verify_bbp_distinction(gamma: float, sigma2: float) -> None:
    """
    Print a numerical demonstration of the distinction between the BBP
    detection threshold and the MP bulk upper edge.

    This function serves as living documentation. Anyone who runs it
    immediately sees why θ_c ≠ λ+ and what each quantity means.

    [BBP05, Corollary 1.1; Mathematical Flag 1 in module docstring]

    For a population spike placed exactly at θ = θ_c + ε (just above
    the detection threshold), this function shows:
      - The BBP threshold θ_c = σ²(1 + √γ)
      - The MP bulk upper edge λ+ = σ²(1 + √γ)²
      - Their ratio λ+/θ_c = (1 + √γ) > 1
      - The sample eigenvalue that a spike at θ_c + ε would produce
        (it lands above λ+, confirming detectability)

    Parameters
    ----------
    gamma : float
        Aspect ratio γ = d/T.
    sigma2 : float
        Noise floor σ².
    """
    _, lp = bulk_edges(gamma, sigma2)
    theta_c = bbp_threshold(gamma, sigma2)

    # Use a spike noticeably above the threshold (0.5*sigma2 above theta_c)
    # to produce a clearly separated sample eigenvalue in the printout.
    epsilon = 0.5 * sigma2
    theta_demo = theta_c + epsilon
    try:
        lambda_spike = bbp_sample_eigenvalue(theta_demo, gamma, sigma2)
    except ValueError:
        lambda_spike = float("nan")

    print("\nBBP vs. MP bulk edge verification  (gamma=%g, sigma2=%g)" % (gamma, sigma2))
    print("-" * 62)
    print("  BBP detection threshold:  theta_c = sigma2*(1+sqrt(gamma))   = %.6f" % theta_c)
    print("  MP bulk upper edge:        lambda+ = sigma2*(1+sqrt(gamma))^2 = %.6f" % lp)
    print("  Ratio lambda+/theta_c:                                         = %.6f" % (lp / theta_c))
    print("  (Note: ratio = 1+sqrt(gamma) = %.6f, always > 1)" % (1 + np.sqrt(gamma)))
    print()
    print("  Interpretation: a population spike placed just above theta_c")
    print("  at theta = %.6f produces a sample eigenvalue at" % theta_demo)
    print("  lambda_spike = %.6f, which is above lambda+ by %.6f." % (lambda_spike, lambda_spike - lp))
    print()
    print("  A spike at exactly theta_c is absorbed into the bulk and is")
    print("  NOT separately detectable. This is a fundamental")
    print("  information-theoretic limit, not an estimation failure.")
    print("  [BBP05, Corollary 1.1; see also module docstring Flag 1]")
    print("-" * 62)


# ---------------------------------------------------------------------------
# Test-support helper (used by test_estimator.py parametrized tests)
# ---------------------------------------------------------------------------

def _sample_mp_eigenvalues_for_test(
    gamma: float,
    sigma2: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate n_samples eigenvalues from the MP distribution via rejection
    sampling.  Intended only for use in unit tests; not part of the public
    research API.

    Uses the continuous density only (no point mass at 0 for gamma > 1).

    Parameters
    ----------
    gamma : float
    sigma2 : float
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    lm, lp = bulk_edges(gamma, sigma2)
    # Find an upper bound for the density on [lm, lp]
    xs_probe = np.linspace(lm + 1e-8, lp - 1e-8, 2000)
    max_density = max(mp_density(x, gamma, sigma2) for x in xs_probe)

    samples: list[float] = []
    while len(samples) < n_samples:
        batch = n_samples * 5
        x_cand = rng.uniform(lm, lp, size=batch)
        u_cand = rng.uniform(0.0, max_density, size=batch)
        densities = np.array([mp_density(x, gamma, sigma2) for x in x_cand])
        accepted = x_cand[u_cand <= densities]
        samples.extend(accepted.tolist())

    return np.array(samples[:n_samples])
