"""
test_mp_theory.py — Unit tests for all theoretical functions in mp_theory.py.

Each test is self-contained and documents the mathematical result it verifies.
Tests are designed to fail loudly and informatively if the implementation
drifts from the underlying theory.
"""

import warnings
import numpy as np
import pytest
from scipy import integrate, stats

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from research.core.mp_theory import (
    bulk_edges,
    mp_density,
    mp_cdf,
    bbp_threshold,
    bbp_sample_eigenvalue,
    estimate_sigma2_self_consistent,
    ks_distance_from_mp,
    verify_bbp_distinction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_mp_eigenvalues(
    gamma: float,
    sigma2: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate eigenvalues that follow the MP distribution via inverse-CDF
    sampling (accept-reject on the continuous part).

    This is used in tests that need ground-truth MP-distributed samples.
    The point mass at 0 (when gamma > 1) is ignored here because we focus
    on the continuous bulk.
    """
    lm, lp = bulk_edges(gamma, sigma2)
    # Sample from continuous part via rejection sampling
    samples = []
    max_density = max(
        mp_density(x, gamma, sigma2)
        for x in np.linspace(lm + 1e-8, lp - 1e-8, 500)
    )
    while len(samples) < n_samples:
        x = rng.uniform(lm, lp, size=n_samples * 5)
        u = rng.uniform(0, max_density, size=len(x))
        accepted = x[u <= np.array([mp_density(xi, gamma, sigma2) for xi in x])]
        samples.extend(accepted.tolist())
    return np.array(samples[:n_samples])


def _sample_spiked_covariance_eigenvalues(
    d: int,
    T: int,
    sigma2: float,
    spike_strengths: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate eigenvalues of a sample covariance matrix from a spiked model.

    Population covariance: Σ = σ²·I + Σ_spikes where spikes are rank-1.
    Data matrix X is T×d Gaussian with covariance Σ.

    Returns the d eigenvalues of (1/T)·X^T·X sorted descending.
    """
    # Build spiked population covariance in diagonal form
    pop_eigs = np.full(d, sigma2)
    for i, s in enumerate(spike_strengths):
        pop_eigs[i] = s  # replace first k diagonal entries

    # Draw data: T observations from N(0, Σ) where Σ = diag(pop_eigs)
    X = rng.standard_normal((T, d)) * np.sqrt(pop_eigs)  # shape T×d
    S = X.T @ X / T  # d×d sample covariance
    eigs = np.linalg.eigvalsh(S)
    return eigs[::-1]  # descending order


# ---------------------------------------------------------------------------
# Test 1: MP density integrates to the correct total mass
# ---------------------------------------------------------------------------

class TestMpDensityIntegration:
    """
    The continuous part of the MP density integrates to (1/γ) when γ > 1
    (remaining mass 1 − 1/γ sits at the point mass at 0), and to 1 when γ ≤ 1.
    [MP67, Example 1; L99, eq. 3]
    """

    @pytest.mark.parametrize("gamma,sigma2", [
        (0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.8, 1.0), (1.0, 1.0),
        (0.1, 0.5), (0.5, 2.0),
        (2.0, 1.0),  # gamma > 1: point mass exists
    ])
    def test_density_integrates_to_one(self, gamma, sigma2):
        lm, lp = bulk_edges(gamma, sigma2)
        expected_continuous_mass = min(1.0, 1.0 / gamma)  # = 1 - max(0, 1-1/γ)

        integral, err = integrate.quad(
            lambda x: mp_density(x, gamma, sigma2),
            lm + 1e-10, lp - 1e-10,
            limit=500,
            epsabs=1e-7,
            epsrel=1e-7,
        )
        assert abs(integral - expected_continuous_mass) < 1e-4, (
            f"gamma={gamma}, sigma2={sigma2}: "
            f"integral={integral:.6f}, expected={expected_continuous_mass:.6f}, "
            f"error={abs(integral - expected_continuous_mass):.2e}"
        )

    def test_density_zero_outside_support(self):
        gamma, sigma2 = 0.5, 1.0
        lm, lp = bulk_edges(gamma, sigma2)
        assert mp_density(lm - 0.1, gamma, sigma2) == 0.0
        assert mp_density(lp + 0.1, gamma, sigma2) == 0.0
        assert mp_density(-1.0, gamma, sigma2) == 0.0
        assert mp_density(100.0, gamma, sigma2) == 0.0

    def test_density_nonnegative_on_support(self):
        gamma, sigma2 = 0.4, 1.5
        lm, lp = bulk_edges(gamma, sigma2)
        xs = np.linspace(lm + 1e-8, lp - 1e-8, 1000)
        densities = np.array([mp_density(x, gamma, sigma2) for x in xs])
        assert np.all(densities >= 0)


# ---------------------------------------------------------------------------
# Test 2: Bulk edges — known analytical values
# ---------------------------------------------------------------------------

class TestBulkEdgesKnownValues:
    """
    Verify λ± = σ²(1 ± √γ)² against hand-calculated values.
    [MP67, p. 462; L99, eq. 3]
    """

    def test_gamma_quarter_sigma2_one(self):
        # γ=0.25, σ²=1.0: √γ=0.5
        # λ- = (1-0.5)² = 0.25, λ+ = (1+0.5)² = 2.25
        lm, lp = bulk_edges(0.25, 1.0)
        assert abs(lm - 0.25) < 1e-12
        assert abs(lp - 2.25) < 1e-12

    def test_gamma_one_sigma2_one(self):
        # γ=1.0, σ²=1.0: √γ=1.0
        # λ- = 0, λ+ = 4.0
        lm, lp = bulk_edges(1.0, 1.0)
        assert abs(lm - 0.0) < 1e-12
        assert abs(lp - 4.0) < 1e-12

    def test_gamma_half_sigma2_two(self):
        # γ=0.5, σ²=2.0: √γ=√0.5≈0.70711
        # λ- = 2*(1-√0.5)² = 2*(0.29289)² ≈ 2*0.08579 ≈ 0.17157
        # λ+ = 2*(1+√0.5)² = 2*(1.70711)² ≈ 2*2.91421 ≈ 5.82843
        lm, lp = bulk_edges(0.5, 2.0)
        expected_lm = 2.0 * (1 - np.sqrt(0.5)) ** 2
        expected_lp = 2.0 * (1 + np.sqrt(0.5)) ** 2
        assert abs(lm - expected_lm) < 1e-12
        assert abs(lp - expected_lp) < 1e-12

    def test_sigma2_scaling(self):
        # λ± should scale linearly with σ²
        lm1, lp1 = bulk_edges(0.3, 1.0)
        lm3, lp3 = bulk_edges(0.3, 3.0)
        assert abs(lm3 - 3.0 * lm1) < 1e-12
        assert abs(lp3 - 3.0 * lp1) < 1e-12

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            bulk_edges(-0.1, 1.0)
        with pytest.raises(ValueError):
            bulk_edges(0.5, -1.0)
        with pytest.raises(ValueError):
            bulk_edges(0.0, 1.0)


# ---------------------------------------------------------------------------
# Test 3: BBP threshold and its distinction from λ+
# ---------------------------------------------------------------------------

class TestBbpThreshold:
    """
    Verify θ_c = σ²(1 + √γ) and document its distinction from λ+.

    Mathematical Flag 1: θ_c ≠ λ+
    θ_c = σ²(1 + √γ)     [BBP05, Corollary 1.1]
    λ+  = σ²(1 + √γ)²    [MP67, Example 1]

    They are related by λ+ = θ_c · (1 + √γ). Since 1 + √γ > 1 for all
    γ > 0, we always have λ+ > θ_c. They are equal only at γ = 0 (trivial
    limit where both → σ²).
    """

    def test_threshold_formula_gamma_quarter(self):
        # γ=0.25, σ²=1.0: θ_c = 1*(1+0.5) = 1.5
        theta_c = bbp_threshold(0.25, 1.0)
        assert abs(theta_c - 1.5) < 1e-12

    def test_threshold_formula_gamma_one(self):
        # γ=1.0, σ²=1.0: θ_c = 1*(1+1) = 2.0
        theta_c = bbp_threshold(1.0, 1.0)
        assert abs(theta_c - 2.0) < 1e-12

    def test_threshold_formula_sigma2_scaling(self):
        theta_c = bbp_threshold(0.3, 2.5)
        assert abs(theta_c - 2.5 * (1 + np.sqrt(0.3))) < 1e-12

    def test_threshold_strictly_less_than_bulk_edge(self):
        """
        θ_c < λ+ for all γ > 0, σ² > 0.
        Ratio λ+/θ_c = (1+√γ) > 1 always.
        """
        for gamma in [0.1, 0.25, 0.5, 0.8, 1.0, 2.0]:
            _, lp = bulk_edges(gamma, 1.0)
            theta_c = bbp_threshold(gamma, 1.0)
            ratio = lp / theta_c
            expected_ratio = 1.0 + np.sqrt(gamma)
            assert lp > theta_c, (
                f"gamma={gamma}: λ+={lp:.4f} should exceed θ_c={theta_c:.4f}"
            )
            assert abs(ratio - expected_ratio) < 1e-12, (
                f"gamma={gamma}: ratio={ratio:.6f}, expected={expected_ratio:.6f}"
            )

    def test_bbp_sample_eigenvalue_above_bulk_edge(self):
        """
        A detectable spike (θ > θ_c) produces a sample eigenvalue above λ+.
        [BBP05, Corollary 1.1(b), eq. 46]
        """
        gamma, sigma2 = 0.25, 1.0
        _, lp = bulk_edges(gamma, sigma2)
        theta_c = bbp_threshold(gamma, sigma2)
        # Test several values above the threshold
        for theta in [theta_c + 0.1, theta_c + 0.5, theta_c + 1.0, 5.0]:
            lam_spike = bbp_sample_eigenvalue(theta, gamma, sigma2)
            assert lam_spike > lp, (
                f"theta={theta:.4f}: λ_spike={lam_spike:.4f} should exceed "
                f"λ+={lp:.4f}"
            )

    def test_bbp_spike_below_threshold_raises(self):
        """
        Spikes at or below θ_c are absorbed into the bulk.
        Calling bbp_sample_eigenvalue() must raise ValueError.
        """
        gamma, sigma2 = 0.25, 1.0
        theta_c = bbp_threshold(gamma, sigma2)
        with pytest.raises(ValueError, match="BBP threshold"):
            bbp_sample_eigenvalue(theta_c, gamma, sigma2)
        with pytest.raises(ValueError, match="BBP threshold"):
            bbp_sample_eigenvalue(theta_c - 0.1, gamma, sigma2)

    def test_bbp_spike_formula_gamma_quarter(self):
        """
        γ=0.25, σ²=1.0, θ=3.0 (well above θ_c=1.5):
          λ_spike = 3*(1 + 0.25/(3-1)) = 3*(1 + 0.125) = 3*1.125 = 3.375
        """
        lam = bbp_sample_eigenvalue(3.0, 0.25, 1.0)
        expected = 3.0 * (1.0 + 0.25 * 1.0 / (3.0 - 1.0))
        assert abs(lam - expected) < 1e-12

    def test_verify_bbp_distinction_runs_without_error(self, capsys):
        """verify_bbp_distinction() runs and prints the expected fields."""
        verify_bbp_distinction(0.25, 1.0)
        out = capsys.readouterr().out
        assert "BBP detection threshold" in out
        assert "MP bulk upper edge" in out
        assert "Ratio" in out
        assert "1.500000" in out   # θ_c = 1.5 for γ=0.25, σ²=1.0
        assert "2.250000" in out   # λ+ = 2.25

    def test_verify_bbp_distinction_analytical_values(self, capsys):
        """
        Hand-calculated values for γ=0.25, σ²=1.0:
          θ_c = 1*(1+0.5)    = 1.5
          λ+  = 1*(1+0.5)²   = 2.25
          ratio               = 1.5
          spike at θ_c+ε produces λ_spike slightly above 2.25
        """
        gamma, sigma2 = 0.25, 1.0
        theta_c_expected = 1.5
        lp_expected = 2.25
        ratio_expected = lp_expected / theta_c_expected  # = 1.5

        verify_bbp_distinction(gamma, sigma2)
        out = capsys.readouterr().out

        # Check the printed values match hand-calculation
        assert "1.500000" in out, f"Expected θ_c=1.5 in output:\n{out}"
        assert "2.250000" in out, f"Expected λ+=2.25 in output:\n{out}"
        assert "1.500000" in out, f"Expected ratio=1.5 in output:\n{out}"


# ---------------------------------------------------------------------------
# Test 4: Self-consistent σ² estimator — no spikes
# ---------------------------------------------------------------------------

class TestSelfConsistentSigma2NoSpikes:
    """
    When eigenvalues are drawn from an MP distribution with known σ², the
    self-consistent estimator should recover σ² within 10%.

    [BBP17, Section 2; estimate_sigma2_self_consistent docstring]
    """

    @pytest.mark.parametrize("gamma,sigma2", [
        (0.1, 1.0), (0.3, 1.0), (0.5, 1.0),
        (0.1, 2.0), (0.3, 0.5),
    ])
    def test_recovery_no_spikes(self, gamma, sigma2):
        rng = np.random.default_rng(42)
        n_samples = 800
        eigs = _sample_mp_eigenvalues(gamma, sigma2, n_samples, rng)

        estimated, n_iter, converged = estimate_sigma2_self_consistent(eigs, gamma)
        assert converged, (
            f"gamma={gamma}, sigma2={sigma2}: estimator did not converge "
            f"in {n_iter} iterations"
        )
        rel_err = abs(estimated - sigma2) / sigma2
        assert rel_err < 0.10, (
            f"gamma={gamma}, sigma2={sigma2}: estimated={estimated:.4f}, "
            f"true={sigma2:.4f}, relative error={rel_err:.3f} > 10%"
        )

    def test_converges_in_few_iterations(self):
        rng = np.random.default_rng(0)
        eigs = _sample_mp_eigenvalues(0.3, 1.0, 500, rng)
        _, n_iter, converged = estimate_sigma2_self_consistent(eigs, 0.3)
        assert converged
        assert n_iter < 20, f"Expected convergence within 20 iters, got {n_iter}"


# ---------------------------------------------------------------------------
# Test 5: Self-consistent σ² estimator — with spikes
# ---------------------------------------------------------------------------

class TestSelfConsistentSigma2WithSpikes:
    """
    When k=3 spike eigenvalues are present well above λ+, the self-consistent
    estimator should recover the bulk σ² without spike contamination.
    The naive mean of all eigenvalues should be biased upward.

    [BBP17, Section 2; estimate_sigma2_self_consistent docstring]
    """

    def test_self_consistent_beats_naive_mean(self):
        rng = np.random.default_rng(99)
        gamma = 0.2
        sigma2_true = 1.0
        d = 200
        T = int(d / gamma)

        # k=3 spikes well above BBP threshold
        theta_c = bbp_threshold(gamma, sigma2_true)
        spike_strengths = [theta_c * 3.0, theta_c * 2.5, theta_c * 2.0]

        eigs = _sample_spiked_covariance_eigenvalues(
            d, T, sigma2_true, spike_strengths, rng
        )

        # Naive mean
        naive_mean = np.mean(eigs)

        # Self-consistent
        estimated, _, converged = estimate_sigma2_self_consistent(eigs, gamma)

        # Naive mean should be biased upward
        assert naive_mean > sigma2_true, (
            "Naive mean should be inflated by spike eigenvalues"
        )

        # Self-consistent should be closer to truth
        err_sc = abs(estimated - sigma2_true)
        err_naive = abs(naive_mean - sigma2_true)
        assert err_sc < err_naive, (
            f"Self-consistent error ({err_sc:.4f}) should be less than "
            f"naive-mean error ({err_naive:.4f})"
        )

    def test_spike_eigenvalues_excluded_from_estimate(self):
        """
        Verify that after convergence, the estimated λ+ excludes spike eigenvalues.
        """
        rng = np.random.default_rng(7)
        gamma = 0.2
        sigma2_true = 1.0
        d = 150
        T = int(d / gamma)

        theta_c = bbp_threshold(gamma, sigma2_true)
        spike_strengths = [theta_c * 3.0, theta_c * 2.5, theta_c * 2.2]

        eigs = _sample_spiked_covariance_eigenvalues(
            d, T, sigma2_true, spike_strengths, rng
        )

        estimated, _, converged = estimate_sigma2_self_consistent(eigs, gamma)
        _, lp_estimated = bulk_edges(gamma, estimated)

        # All three spikes should be above the estimated λ+
        top_3 = np.sort(eigs)[-3:]
        for spike_sample in top_3:
            assert spike_sample > lp_estimated, (
                f"Spike sample eig {spike_sample:.4f} should exceed "
                f"estimated λ+ {lp_estimated:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 6: KS distance under null and alternative
# ---------------------------------------------------------------------------

class TestKsDistanceFromMp:
    """
    Under the null (eigenvalues from MP), KS distance should be small.
    Under the alternative (different distribution), it should be large.

    [ks_distance_from_mp docstring]
    """

    def test_small_ks_under_null(self):
        """
        With N=1000 samples from the exact MP distribution, KS should be < 0.05.
        """
        rng = np.random.default_rng(13)
        gamma, sigma2 = 0.3, 1.0
        lm, lp = bulk_edges(gamma, sigma2)
        eigs = _sample_mp_eigenvalues(gamma, sigma2, 1000, rng)

        # Filter to bulk (they're all bulk since no spikes were added)
        bulk = eigs[eigs < lp]
        ks = ks_distance_from_mp(bulk, gamma, sigma2)
        assert ks < 0.05, (
            f"KS distance under null should be < 0.05 for N=1000, got {ks:.4f}"
        )

    def test_large_ks_under_alternative(self):
        """
        Uniform distribution on [λ-, λ+] will produce large KS vs MP CDF.
        """
        rng = np.random.default_rng(21)
        gamma, sigma2 = 0.3, 1.0
        lm, lp = bulk_edges(gamma, sigma2)
        uniform_eigs = rng.uniform(lm, lp, size=500)
        ks = ks_distance_from_mp(uniform_eigs, gamma, sigma2)
        assert ks > 0.10, (
            f"KS distance under wrong distribution should be > 0.10, got {ks:.4f}"
        )

    def test_ks_decreases_with_sample_size(self):
        """
        KS distance under the null should decrease (on average) as N grows.
        """
        rng = np.random.default_rng(42)
        gamma, sigma2 = 0.3, 1.0
        lm, lp = bulk_edges(gamma, sigma2)

        ks_small = np.mean([
            ks_distance_from_mp(
                _sample_mp_eigenvalues(gamma, sigma2, 100, rng),
                gamma, sigma2
            )
            for _ in range(20)
        ])
        ks_large = np.mean([
            ks_distance_from_mp(
                _sample_mp_eigenvalues(gamma, sigma2, 1000, rng),
                gamma, sigma2
            )
            for _ in range(20)
        ])
        assert ks_small > ks_large, (
            f"KS should decrease with N: small-N avg={ks_small:.4f}, "
            f"large-N avg={ks_large:.4f}"
        )

    def test_empty_bulk_returns_one(self):
        """Empty bulk array returns 1.0 as a conservative signal."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ks = ks_distance_from_mp(np.array([]), 0.3, 1.0)
        assert ks == 1.0


# ---------------------------------------------------------------------------
# Test 7: MP CDF properties
# ---------------------------------------------------------------------------

class TestMpCdf:
    """Verify basic CDF properties: monotonicity, limits, and values."""

    def test_cdf_at_lower_edge(self):
        gamma, sigma2 = 0.5, 1.0
        lm, lp = bulk_edges(gamma, sigma2)
        # CDF just below lm should equal point mass (0 for gamma ≤ 1)
        cdf_below = mp_cdf(lm - 1e-6, gamma, sigma2)
        assert abs(cdf_below - max(0.0, 1.0 - 1.0 / gamma)) < 1e-6

    def test_cdf_at_upper_edge(self):
        for gamma in [0.2, 0.5, 1.0]:
            sigma2 = 1.0
            _, lp = bulk_edges(gamma, sigma2)
            cdf_at_top = mp_cdf(lp + 1e-6, gamma, sigma2)
            assert abs(cdf_at_top - 1.0) < 1e-5, (
                f"gamma={gamma}: CDF at λ+={lp:.4f} should be 1.0, "
                f"got {cdf_at_top:.6f}"
            )

    def test_cdf_monotone(self):
        gamma, sigma2 = 0.4, 1.2
        lm, lp = bulk_edges(gamma, sigma2)
        xs = np.linspace(lm, lp, 100)
        cdfs = np.array([mp_cdf(x, gamma, sigma2) for x in xs])
        diffs = np.diff(cdfs)
        assert np.all(diffs >= -1e-8), "CDF must be non-decreasing"

    def test_point_mass_gamma_greater_than_one(self):
        """For γ > 1, CDF should jump by (1 - 1/γ) at x=0."""
        gamma, sigma2 = 2.0, 1.0
        expected_pm = 1.0 - 1.0 / gamma  # = 0.5
        cdf_at_zero = mp_cdf(0.0, gamma, sigma2)
        assert abs(cdf_at_zero - expected_pm) < 1e-6


# ---------------------------------------------------------------------------
# Test 8: Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_bulk_edges_invalid_gamma(self):
        with pytest.raises(ValueError):
            bulk_edges(0.0, 1.0)
        with pytest.raises(ValueError):
            bulk_edges(-1.0, 1.0)

    def test_bulk_edges_invalid_sigma2(self):
        with pytest.raises(ValueError):
            bulk_edges(0.5, 0.0)
        with pytest.raises(ValueError):
            bulk_edges(0.5, -2.0)

    def test_bbp_threshold_invalid(self):
        with pytest.raises(ValueError):
            bbp_threshold(0.0, 1.0)
        with pytest.raises(ValueError):
            bbp_threshold(0.5, 0.0)

    def test_estimate_sigma2_empty(self):
        with pytest.raises(ValueError):
            estimate_sigma2_self_consistent(np.array([]), 0.3)

    def test_estimate_sigma2_invalid_gamma(self):
        with pytest.raises(ValueError):
            estimate_sigma2_self_consistent(np.array([1.0, 2.0]), 0.0)
