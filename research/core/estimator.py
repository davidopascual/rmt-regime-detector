"""
estimator.py — Rolling spectral estimator for time-varying covariance structure.

This module implements the RollingSpectralEstimator, which applies the
Marčenko-Pastur framework from mp_theory.py to a sliding window over a
panel of returns, computing a full spectral summary at each step.

Theoretical basis
-----------------
  [MP67]  Marčenko, Pastur (1967) — bulk edges, MP density
  [L99]   Laloux et al. (1999) — noise dressing interpretation
  [P02]   Plerou et al. (2002) — rolling window methodology
  [BBP05] Baik, Ben Arous, Péché (2005) — BBP spike detection threshold
  [BBP17] Bun, Bouchaud, Potters (2017) — cleaning, self-consistent σ²

Design decisions
----------------
Within each window the data matrix X (T×d) is standardized to zero mean
and unit variance per asset column before computing the sample covariance.
This is REQUIRED for MP law applicability: the law holds for matrices with
unit-variance entries. Standardizing removes cross-sectional volatility
differences; we are therefore analyzing correlation structure, not
covariance structure. This must be borne in mind when interpreting σ²:
after standardization, σ² = 1 under the pure-noise null.

The aspect ratio γ(t) = d(t)/T is recomputed at each step because the
number of active assets d(t) can vary if assets are dropped due to
missing data.

Spike detection uses the self-consistent σ² estimate and the BBP threshold
θ_c = σ²(1 + √γ) to count eigenvalues above λ+(t). Note carefully:

  A spike eigenvalue in the SAMPLE spectrum above λ+(t) implies a
  POPULATION eigenvalue above θ_c = σ²(1+√γ), i.e., above the BBP
  detection threshold. Below θ_c, population spikes are provably
  undetectable; they appear inside the bulk.

  [Mathematical Flag 1]: θ_c ≠ λ+. See mp_theory module docstring.
  [Mathematical Flag 2]: BBP proven for complex Gaussian; applied here
                          to real financial data as a heuristic.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.core.mp_theory import (
    bulk_edges,
    bbp_threshold,
    estimate_sigma2_self_consistent,
    ks_distance_from_mp,
    full_ks_distance_from_mp,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SpectralSnapshot:
    """
    All spectral statistics at a single rolling window.

    Attributes
    ----------
    t_index : int
        Integer index of this window in the rolling sequence (0-based).
    center_pos : int
        Row index (in the original returns array) of the last observation
        in this window.
    d : int
        Number of assets active in this window (after dropping missing data).
    T : int
        Window length (number of observations).
    gamma : float
        Aspect ratio γ = d/T at this window.
    eigenvalues : np.ndarray
        All d sample eigenvalues of (1/T)X^T X, sorted descending.
    sigma2 : float
        Self-consistently estimated noise floor.
    sigma2_converged : bool
        Whether the self-consistent iteration converged.
    lambda_minus : float
        MP lower bulk edge σ²(1 − √γ)².
    lambda_plus : float
        MP upper bulk edge σ²(1 + √γ)².
    lambda1 : float
        Largest sample eigenvalue (market mode strength).
    k : int
        Number of eigenvalues above λ+ (estimated signal factor count).
    rho : float
        Detachment ratio λ₁/λ+. Values > 1 indicate market-mode strength
        relative to the noise ceiling; spikes during crises.
    ks : float
        KS distance between empirical BULK CDF and fitted MP CDF.
        Measures goodness-of-fit of the identified noise floor; NOT a
        direct crisis indicator (see ks_full for that role).
    ks_full : float
        KS distance between the FULL empirical CDF (all d eigenvalues)
        and the fitted MP CDF.  Approximately equals k/d (spike fraction)
        at x=lambda+ and is the primary statistic for the ESD comparison
        figure.  Correctly elevated during crises when spike count surges.
    r_eff : float
        Effective rank = (Σλᵢ)² / Σλᵢ² (participation ratio / inverse
        Herfindahl index of eigenvalue mass). Low r_eff means concentration
        in few dominant modes; high r_eff means diffuse, near-identity.
    kappa : float
        Condition number λ₁/λ_min (smallest non-zero eigenvalue).
        Tracks numerical ill-conditioning of the covariance estimate.
    eigvecs_top : np.ndarray or None
        Top-k eigenvectors, shape (d, k_stored), columns in descending
        eigenvalue order.  Only populated when the estimator is run with
        store_top_k_eigvecs > 0.  Required for subspace overlap Extension 2.
    active_cols : np.ndarray or None
        Integer indices (into the original d_total columns) of the assets
        that were active (no NaN) in this window.  Shape (d,).  Only
        populated when store_top_k_eigvecs > 0.
    """
    t_index: int
    center_pos: int
    d: int
    T: int
    gamma: float
    eigenvalues: np.ndarray
    sigma2: float
    sigma2_converged: bool
    lambda_minus: float
    lambda_plus: float
    lambda1: float
    k: int
    rho: float
    ks: float
    ks_full: float
    r_eff: float
    kappa: float
    eigvecs_top: Optional[np.ndarray] = None
    active_cols: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class RollingSpectralEstimator:
    """
    Compute Marčenko-Pastur spectral statistics over a rolling window.

    Parameters
    ----------
    window : int
        Length T of each rolling window (number of observations).
        Default 252 (approximately one trading year).
        Results are qualitatively robust to T in {126, 189, 252} but
        the choice of T materially affects γ = d/T and therefore λ+.
        Sensitivity to T must be documented in any published results.
    step : int
        Number of observations between consecutive windows.
        Default 21 (approximately one trading month).
    min_assets : int
        Minimum number of active assets d required to compute a window.
        Windows with fewer active assets are skipped and logged.
    sigma2_tol : float
        Convergence tolerance for the self-consistent σ² estimator.
    sigma2_max_iter : int
        Maximum iterations for the self-consistent estimator.
    store_top_k_eigvecs : int
        If > 0, compute and store the top-k eigenvectors in each
        SpectralSnapshot.eigvecs_top (shape d × k).  Also stores
        SpectralSnapshot.active_cols (the column indices of active assets
        in the original returns array, needed to compute cross-window
        overlap when d varies).  Uses np.linalg.eigh (slower than
        eigvalsh); set to 0 (default) to skip.

    Usage
    -----
    est = RollingSpectralEstimator(window=252, step=21)
    results = est.fit(returns)   # returns: T_total × d array, NaN for missing
    """

    def __init__(
        self,
        window: int = 252,
        step: int = 21,
        min_assets: int = 10,
        sigma2_tol: float = 1e-6,
        sigma2_max_iter: int = 50,
        store_top_k_eigvecs: int = 0,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        if step < 1:
            raise ValueError("step must be >= 1")
        if min_assets < 2:
            raise ValueError("min_assets must be >= 2")

        self.window = window
        self.step = step
        self.min_assets = min_assets
        self.sigma2_tol = sigma2_tol
        self.sigma2_max_iter = sigma2_max_iter
        self.store_top_k_eigvecs = int(store_top_k_eigvecs)

        self.snapshots: list[SpectralSnapshot] = []
        self.skipped_windows: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> list[SpectralSnapshot]:
        """
        Run the rolling spectral estimation over a panel of returns.

        Parameters
        ----------
        returns : np.ndarray, shape (T_total, d_total)
            Panel of returns. NaN values are treated as missing.
            Rows are time steps; columns are assets.

        Returns
        -------
        list of SpectralSnapshot
            One entry per window that had enough active assets.
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2:
            raise ValueError("returns must be a 2-D array (T_total × d_total)")

        T_total, d_total = returns.shape
        self.snapshots = []
        self.skipped_windows = []

        t_index = 0
        for end in range(self.window, T_total + 1, self.step):
            start = end - self.window
            window_data = returns[start:end, :]   # shape (T, d_total)

            snap = self._process_window(window_data, t_index, end - 1)
            if snap is not None:
                self.snapshots.append(snap)
            t_index += 1

        return self.snapshots

    def fit_single_window(self, window_data: np.ndarray) -> Optional[SpectralSnapshot]:
        """
        Compute spectral statistics for a single pre-sliced window.

        Parameters
        ----------
        window_data : np.ndarray, shape (T, d)
            Returns for a single window. May contain NaN (missing assets
            will be dropped for this window).

        Returns
        -------
        SpectralSnapshot or None
            None if fewer than min_assets assets are active.
        """
        window_data = np.asarray(window_data, dtype=float)
        if window_data.ndim != 2:
            raise ValueError("window_data must be 2-D (T × d)")
        return self._process_window(window_data, t_index=0, center_pos=len(window_data) - 1)

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _process_window(
        self,
        window_data: np.ndarray,
        t_index: int,
        center_pos: int,
    ) -> Optional[SpectralSnapshot]:
        """
        Full spectral computation for one window.

        Steps
        -----
        1. Drop assets with any NaN in this window.
        2. Standardize each remaining asset to zero mean, unit variance.
           [Required for MP law; see module docstring.]
        3. Compute sample covariance (1/T) X^T X.
        4. Compute eigenvalues.
        5. Self-consistent σ² estimation.
        6. Compute all derived statistics.
        """
        T, d_total = window_data.shape

        # Step 1: drop assets with missing data in this window
        active_mask = ~np.any(np.isnan(window_data), axis=0)
        active_cols_idx = np.where(active_mask)[0]   # integer indices into original columns
        X = window_data[:, active_mask]
        d = X.shape[1]

        if d < self.min_assets:
            self.skipped_windows.append({
                "t_index": t_index,
                "center_pos": center_pos,
                "reason": f"only {d} active assets (min={self.min_assets})",
            })
            return None

        # Step 2: standardize within window (zero mean, unit variance).
        # We use ddof=0 (divide by T, not T-1) so that the resulting
        # X_std has EXACT sample variance 1 column-wise.  This is required
        # for Tr(S) = d, which is the normalisation assumed by the MP law
        # (each diagonal entry of the population covariance = sigma^2).
        # Using ddof=1 would give Tr(S) = d*(T-1)/T, a ~0.4% bias for
        # T=252 that violates the exact MP normalisation used in all tests.
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=0)
        zero_var = stds < 1e-12
        if np.any(zero_var):
            n_zero = zero_var.sum()
            warnings.warn(
                f"Window t={t_index}: {n_zero} asset(s) have near-zero variance "
                f"and will be dropped before standardization.",
                RuntimeWarning,
                stacklevel=3,
            )
            active_cols_idx = active_cols_idx[~zero_var]
            X = X[:, ~zero_var]
            means = means[~zero_var]
            stds = stds[~zero_var]
            d = X.shape[1]
            if d < self.min_assets:
                self.skipped_windows.append({
                    "t_index": t_index,
                    "center_pos": center_pos,
                    "reason": f"only {d} active assets after dropping zero-variance",
                })
                return None

        X_std = (X - means) / stds   # shape (T, d), each column ~ N(0,1)

        # Step 3: sample covariance
        # (1/T) X^T X — note: X is already demeaned, no need to demean again
        # since we subtracted the window mean in step 2.
        # Shape: (d, d)
        cov = X_std.T @ X_std / T

        # Step 4: eigenvalues (and optionally eigenvectors).
        # eigvalsh is faster when eigenvectors aren't needed.
        if self.store_top_k_eigvecs > 0:
            eigs_asc, evecs_asc = np.linalg.eigh(cov)
            # Top-k eigenvectors (descending eigenvalue order)
            k_store = min(self.store_top_k_eigvecs, d)
            # evecs_asc columns are in ascending order; reverse to get descending
            eigvecs_top = evecs_asc[:, ::-1][:, :k_store].copy()  # shape (d, k_store)
            stored_active_cols = active_cols_idx.copy()
        else:
            eigs_asc = np.linalg.eigvalsh(cov)
            eigvecs_top = None
            stored_active_cols = None

        eigenvalues = eigs_asc[::-1].copy()   # descending
        # Clamp tiny negative eigenvalues from floating-point noise
        eigenvalues = np.maximum(eigenvalues, 0.0)

        gamma = d / T

        # Step 5: self-consistent σ²
        sigma2, _, converged = estimate_sigma2_self_consistent(
            eigenvalues, gamma,
            tol=self.sigma2_tol,
            max_iter=self.sigma2_max_iter,
        )

        # Step 6: derived statistics
        lm, lp = bulk_edges(gamma, sigma2)

        lambda1 = float(eigenvalues[0])
        lambda_min_nonzero = float(eigenvalues[eigenvalues > 1e-12][-1]) \
            if np.any(eigenvalues > 1e-12) else 1e-12

        # k: eigenvalues strictly above the MP bulk upper edge λ+
        # These are the empirically detectable signal components.
        k = int(np.sum(eigenvalues > lp))

        # ρ: detachment ratio — how far the market mode has moved above noise
        rho = lambda1 / lp if lp > 0 else float("inf")

        # KS distance — only bulk eigenvalues (those below λ+)
        bulk_eigs = eigenvalues[eigenvalues <= lp]
        ks = ks_distance_from_mp(bulk_eigs, gamma, sigma2)

        # Full KS distance — all eigenvalues vs. MP CDF
        # Captures the spike mass deficit (~ k/d at x=λ+).
        # Correctly elevated during crises when spike count surges.
        ks_full = full_ks_distance_from_mp(eigenvalues, gamma, sigma2)

        # Effective rank (participation ratio)
        sum_eigs = float(np.sum(eigenvalues))
        sum_eigs2 = float(np.sum(eigenvalues ** 2))
        r_eff = (sum_eigs ** 2) / sum_eigs2 if sum_eigs2 > 0 else float("nan")

        # Condition number
        kappa = lambda1 / lambda_min_nonzero if lambda_min_nonzero > 0 else float("inf")

        return SpectralSnapshot(
            t_index=t_index,
            center_pos=center_pos,
            d=d,
            T=T,
            gamma=gamma,
            eigenvalues=eigenvalues,
            sigma2=sigma2,
            sigma2_converged=converged,
            lambda_minus=float(lm),
            lambda_plus=float(lp),
            lambda1=lambda1,
            k=k,
            rho=rho,
            ks=ks,
            ks_full=ks_full,
            r_eff=r_eff,
            kappa=kappa,
            eigvecs_top=eigvecs_top,
            active_cols=stored_active_cols,
        )
