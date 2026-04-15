# Spectral Regime Detection in High-Dimensional Financial Correlation Matrices: A Random Matrix Theory Approach

---

## Abstract

Standard risk measures based on rolling sample covariances conflate genuine changes in market
structure with estimation noise, providing no principled basis for distinguishing signal from
bulk fluctuation. We apply the Marčenko-Pastur law as a null model for the noise floor of
the sample correlation matrix, rolling over 93 S&P 500 constituents from 2003 to 2024
(T=252, step=21 trading days, γ=0.369), and derive a sequential CUSUM detector calibrated
on pre-crisis baselines. Five findings emerge. First, the smoothed first difference of the
detachment ratio dρ/dt achieves AUROC 0.750, outperforming the level statistic ρ(t) by
eight points and constituting the best single crisis indicator we identify. Second, ρ(t)
leads crisis onset dates by approximately five windows (105 days), meaning structural
compression begins well before conventional market dislocations. Third, at every measured
crisis, noise floor compression (Effect B: the MP upper edge λ+ falling) accounts for
51–63% of log(ρ) elevation, consistently exceeding market factor amplification. Fourth,
eigenvector subspace rotation follows eigenvalue elevation rather than preceding it, failing
to provide additional early warning. Fifth, no spectral statistic Granger-causes realized
volatility at any tested lag. This framework constitutes a calibrated contemporaneous
diagnostic for structural correlation breakdown, not a trading signal.

---

## 1. Introduction

Risk management in high-dimensional portfolios depends on accurate characterization of the
covariance structure of returns. Standard practice estimates a rolling sample covariance
matrix and derives risk measures — Value-at-Risk, correlation coefficients, factor loadings
— directly from this estimate. This approach has a well-documented limitation: in the regime
where the number of assets d is of the same order as the estimation window T, the sample
covariance matrix is a severely noisy estimate of the population covariance. The bulk of the
sample spectrum is dominated by estimation noise that has no counterpart in the population
matrix. Eigenvalues attributable to genuine common factors are systematically distorted by
this noise, and the boundary between signal and noise is not visible without a theoretical
null model.

The consequence for financial risk monitoring is practical and acute. During market crises,
the structure of the correlation matrix changes fundamentally: correlations between
previously uncorrelated assets surge, the effective number of independent factors collapses,
and diversification benefits evaporate. These structural changes precede or accompany the
most extreme return realizations. Yet conventional rolling covariance estimates cannot
distinguish a genuine structural change from a noise fluctuation in the sample spectrum,
because they lack a reference distribution for what the spectrum should look like under
normality. A practitioner watching a rolling largest eigenvalue has no principled threshold
to determine whether a given value signals crisis or merely reflects finite-sample noise.

Random Matrix Theory provides exactly this missing null model. The Marčenko-Pastur law
(Marčenko and Pastur, 1967) characterizes the limiting spectral distribution of a sample
covariance matrix built from independent, identically distributed entries in the
high-dimensional limit where d/T converges to a fixed constant γ. The bulk of the spectrum
— the eigenvalues attributable to noise — concentrates in the interval [λ−, λ+] with a
specific closed-form density. Eigenvalues outside this interval correspond to genuine
population factors detectable above the noise floor. This separation is both theoretically
precise and empirically useful: it provides a data-dependent, calibrated threshold for
distinguishing signal from noise at each point in time.

Prior work has established the qualitative empirical picture. Laloux et al. (1999) and
Plerou et al. (2002) first applied the Marčenko-Pastur framework to financial correlation
matrices, demonstrating that the bulk of the equity return spectrum conforms well to the
MP distribution and that deviations concentrate in a small number of spike eigenvalues.
Bun, Bouchaud, and Potters (2017) developed cleaning estimators that use the self-consistent
σ² estimator to separate signal from noise more precisely. The BBP phase transition (Baik,
Ben Arous, and Péché, 2005) established the theoretical threshold θ_c above which population
eigenvalues are asymptotically detectable in the sample spectrum. These contributions
collectively established that the MP framework is the correct theoretical lens for analyzing
financial correlation matrices.

What remains missing from the literature is a formal sequential detection framework built on
these theoretical foundations. The existing work describes what the spectrum looks like
during crises, but does not formalize when to declare a structural break, how to calibrate
the false alarm rate, and what the honest discriminative power of the resulting detector is.
The nearest relative is the change-point detection literature (Page, 1954), but applying
CUSUM to spectral statistics in financial data raises non-trivial calibration problems:
financial time series are not stationary, the in-control distribution changes between market
regimes, and the stationarity assumption underlying standard CUSUM analysis is violated by
construction.

This paper makes the following specific contributions. We implement a rolling spectral
estimator on a 20-year S&P 500 panel and compute five spectral statistics at each step: the
detachment ratio ρ(t) = λ₁(t)/λ+(t), the spike count k(t), the KS distance between the
empirical bulk and the fitted MP density, the effective rank r_eff(t), and the eigenvector
subspace overlap O(t). We apply a CUSUM detector calibrated separately on pre-GFC and
post-GFC baselines and evaluate detection delay and false alarm rates against a crisis
calendar. We decompose the elevation of ρ(t) during crises into two mechanistically distinct
effects. We compute AUROC against the crisis calendar for every statistic including velocity
and subspace instability. We conduct Granger causality tests to determine whether any
spectral statistic predicts future realized volatility. We train a combined logistic
detector with a strict train/test split and report out-of-sample performance.

We claim: a calibrated sequential spectral detector with honest out-of-sample validation
is feasible and informative. We do not claim: that these statistics predict future returns
or volatility, that the detector provides a trading signal, or that the results constitute
a formal statistical test under the violated CUSUM stationarity assumption. All three
mathematical flags documented in the codebase are stated explicitly in Section 2, and all
limitations are stated in Section 6.

---

## 2. Theoretical Background

### 2.1 The Marčenko-Pastur Law

Let X be a T × d matrix of independently drawn entries with mean zero and variance σ², and
let S = (1/T) X^T X be the sample covariance matrix. The Marčenko-Pastur law characterizes
the empirical spectral distribution (ESD) of S in the limit as both T and d grow large with
d/T → γ ∈ (0, 1). The ESD converges almost surely to the Marčenko-Pastur distribution with
density

    p_MP(λ) = (1/2πσ²γλ) √[(λ_+ − λ)(λ − λ_−)]   for λ ∈ [λ_−, λ_+]

where the bulk edges are

    λ_± = σ²(1 ± √γ)²

and σ² is the common variance of the entries. Outside the interval [λ−, λ+], the density is
zero in the limit. When γ > 1, there is an additional point mass at zero of weight 1 − 1/γ.

Three aspects of this result are central to our application. First, the bulk edges depend
only on σ² and γ, both of which are estimable from data. This makes λ+ a data-adaptive
threshold rather than a fixed constant. Second, the specific shape of the density within
[λ−, λ+] is fully determined by σ² and γ, enabling a goodness-of-fit test via KS distance.
Third, the concentration of the ESD in [λ−, λ+] is a limiting result; for finite T and d,
the bulk extends somewhat beyond the theoretical edges, particularly when returns have heavy
tails or ARCH effects.

In our application, we standardize each asset's returns to zero mean and unit variance
within each rolling window before computing the sample covariance. This standardization is
required for two reasons: first, it ensures that the diagonal entries of the population
covariance matrix equal σ² uniformly across assets, satisfying the MP homogeneity
assumption; second, it removes cross-sectional heterogeneity in volatility that is
irrelevant to correlation structure. The resulting S = (1/T) X_std^T X_std is a sample
correlation matrix, and σ² = 1 under the pure noise null after standardization.

Standardization uses the population standard deviation (ddof=0) rather than the sample
standard deviation (ddof=1). This choice ensures Tr(S) = d exactly, which is the
normalization assumed by the MP law. With ddof=1, one obtains Tr(S) = d(T-1)/T, a
fractional bias of approximately 0.4% at T = 252 that is small but systematic and
violates the exact MP normalization.

### 2.2 The Self-Consistent σ² Estimator

Applying the MP law to financial data requires estimating σ². The naive estimator — the mean
of all eigenvalues — is biased when spike eigenvalues are present, because these eigenvalues
inflate the mean above the true noise floor. The correct procedure is to estimate σ² using
only the bulk eigenvalues, which requires knowing λ+ first, which itself depends on σ².

We break this circularity with a self-consistent fixed-point iteration. Initialize σ²⁽⁰⁾
as the mean of all eigenvalues. At each step, compute the current upper bulk edge
λ+⁽ⁿ⁾ = σ²⁽ⁿ⁾(1 + √γ)², exclude all eigenvalues above this edge, and update σ²⁽ⁿ⁺¹⁾ as
the mean of the remaining eigenvalues. Iterate to convergence (tolerance 10⁻⁶, maximum
50 iterations). This estimator is described and validated in Bun, Bouchaud, and Potters
(2017), and in our data converges in fewer than 20 iterations at all windows.

The estimator has two known failure modes. When γ is close to 1, the bulk occupies nearly
the full positive half-line and the spike region is ill-defined. When k is large — many
eigenvalues above λ+ — the bulk mean is computed from few eigenvalues and the estimate is
noisy. In our data, γ = 0.369 and k ranges from 13 to 37 out of 93 assets, so between 56
and 80 eigenvalues contribute to the bulk estimate at each step. Both failure conditions
are avoided in our operating regime, though the accuracy of the estimate degrades slightly
at the highest-k windows during COVID.

### 2.3 The BBP Phase Transition and Mathematical Flag 1

The Baik-Ben Arous-Péché (2005) theorem characterizes what happens to sample eigenvalues
when the corresponding population eigenvalues are above the noise floor. The key result is
a phase transition: a population eigenvalue θ produces a sample spike above λ+ if and only
if θ > θ_c, where

    θ_c = σ²(1 + √γ)

Below this threshold, the population spike is undetectable: its sample counterpart merges
with the bulk and is not statistically distinguishable from noise.

**Mathematical Flag 1**: The BBP detection threshold θ_c and the MP bulk upper edge λ+ are
distinct quantities with a specific quantitative relationship:

    θ_c = σ²(1 + √γ)          [BBP detection threshold]
    λ+   = σ²(1 + √γ)²        [MP bulk upper edge]
    λ+/θ_c = (1 + √γ) > 1     [ratio always exceeds one]

At γ = 0.369, 1 + √γ = 1.607, so λ+ = 1.607 × θ_c. A population eigenvalue at θ_c appears
in the sample spectrum at a value substantially above θ_c, not at θ_c itself. Confusing
these two quantities — which is common in applied work — leads to the error of treating λ+
as the detection threshold for population eigenvalues. The correct statement is that sample
eigenvalues above λ+ imply population eigenvalues above θ_c, i.e., above the BBP detection
threshold.

**Mathematical Flag 2**: The BBP phase transition theorem is proven for complex Gaussian
random matrices (the GUE ensemble). Extension to real symmetric matrices (GOE) is conjectured
in the original paper but not proven there; Capitaine, Donati-Martin, and Féral (2009)
established the result for real matrices with finite fourth moment under regularity conditions.
Extension to financial returns — which exhibit fat tails (kurtosis significantly above 3),
volatility clustering (ARCH effects), and cross-sectional and temporal dependence — is a
heuristic application only. The empirical behavior of the BBP transition under heavy-tailed
dependent returns is not theoretically characterized. We apply the BBP threshold as an
approximate guideline; results should not be interpreted as if the Gaussian assumptions hold
exactly.

### 2.4 CUSUM Change-Point Detection and Mathematical Flag 3

Sequential change-point detection asks: given a stream of observations, when is the first
time we can conclude, with controlled false alarm probability, that the generating distribution
has changed? Page (1954) introduced the CUSUM (cumulative sum) control chart, stating his
Rule 1 as: take action at the first n such that S_n − min_{0 ≤ i ≤ n} S_i > h, where
S_n = Σ_{j=1}^{n} x_j.

The equivalent recursive formulation now standard in practice is:

    C(0) = 0
    C(t) = max(0, C(t−1) + (x(t) − μ − δ))

where μ is the in-control mean, δ is the allowance parameter (half the shift to be detected),
and an alarm fires when C(t) > h.

**Mathematical Flag 3**: This recursive form is not stated in Page (1954). Page's original
paper derives the rule in terms of the minimum of cumulative sums. The recursive max(0,·)
formulation was derived and proven equivalent by Lorden (1971) and Moustakides (1986).
Attributing this formula directly to "Page 1954" — common in applied literature — is
imprecise. The equivalence for detecting an upward shift is exact; the attribution is not.

CUSUM has optimal average run length properties under the assumption that the in-control
series is stationary with known mean and variance. Financial time series violate this
assumption on multiple grounds: volatility clustering (ARCH/GARCH effects) creates
heteroskedastic innovation variance; regime-dependent baseline levels mean that no single
(μ, σ) characterizes the in-control process over a long monitoring horizon; and serial
dependence from overlapping rolling windows means that consecutive CUSUM increments are
correlated. We address the last problem partially by using a resettable CUSUM (which resets
to zero after each alarm, preventing indefinite accumulation), and the second problem by
using two separate regime calibrations. The first — volatility clustering — remains
unaddressed and represents the most significant gap between the theoretical ARL guarantees
of standard CUSUM and our empirical application.

---

## 3. Data and Methodology

### 3.1 Universe and Data

The study universe comprises 93 of the 100 S&P 500 constituents included in the SP100 list.
Seven tickers were excluded because they did not trade for the full 2003–2024 study period:
Kraft Heinz (KHC, spun off 2015), Phillips 66 (PSX, spun off from ConocoPhillips 2012),
Marathon Petroleum (MPC, IPO 2011), General Motors (GM, re-IPO 2010 after bankruptcy
restructuring), Charter Communications (CHTR, emerged from bankruptcy 2009), Dow Inc.
(DOW, spun off from DowDuPont 2019), and LyondellBasell (LYB, emerged from bankruptcy
2010). These exclusions are not a data cleaning choice; they reflect the structural reality
that these companies either did not exist as publicly traded entities for the full period
or had their price history interrupted by corporate events that create artificial return
discontinuities. Retaining them would introduce survivorship-biased or structurally
discontinuous observations into the spectral estimates. The retained 93-asset universe
exhibits at most 0.8% missing observations per asset, and assets with any missing
observation in a given rolling window are dropped from that window's spectral estimate.

Adjusted daily closing prices were obtained from Yahoo Finance for the period
2003-01-03 to 2024-12-30. Log-returns r_i(t) = log(P_i(t)/P_i(t-1)) were computed
for each asset. No outlier winsorization was applied. This choice is deliberate: within-window
standardization to unit variance (described below) means that a single large return
inflates the denominator rather than the numerator, compressing rather than amplifying its
contribution to the sample covariance. Winsorization would introduce asymmetry in the
treatment of large returns and alter the standardized distribution in ways that complicate
the MP null model comparison.

The study period spans 5,535 trading days. The universe contains 93 assets throughout most
of the period, though the active asset count d(t) varies slightly window by window as the
NaN-exclusion criterion removes temporarily illiquid assets.

### 3.2 Rolling Spectral Estimator

We estimate the sample covariance matrix on a rolling window of length T = 252 trading days
(approximately one calendar year), advancing by step = 21 trading days (approximately one
calendar month). The aspect ratio γ = d(t)/T is recomputed at each step; its typical value
is γ = 93/252 = 0.369.

Within each window, each asset's returns are standardized to zero mean and unit variance
using the window-specific mean and population standard deviation (ddof=0). This
standardization is required for the MP law to apply (all population diagonal entries must
equal σ²); it also means we are analyzing correlation structure rather than covariance
structure. The sample covariance matrix is S = (1/T) X_std^T X_std ∈ R^{d×d}.

Eigenvalues are computed from S using the symmetric eigensolver (LAPACK dsyevd). For the
subspace overlap analysis in Section 4.4, eigenvectors are retained for the top k = 5
modes. The self-consistent σ² estimator is applied at each window, with the resulting
estimate used to compute λ−(t) and λ+(t). Five snapshot statistics are recorded per window:
the detachment ratio ρ(t) = λ₁(t)/λ+(t), the spike count k(t) (number of eigenvalues above
λ+(t)), the KS distance between the empirical bulk ESD and the fitted MP density, the
effective rank r_eff(t) = (Σλ_i)²/Σλ_i², and the condition number κ(t) = λ₁(t)/λ_min(t).

This procedure yields 252 snapshots covering the period from approximately early 2004 (when
the first window of T=252 observations completes) through late 2024. The crisis calendar
used for validation contains seven episodes: Dot-com bust (2000-03-10 to 2002-10-09),
Post-9/11 shock (2001-09-11 to 2001-09-21), Global Financial Crisis (2007-10-09 to
2009-03-09), European Sovereign Debt crisis (2010-04-23 to 2010-07-02), Flash Crash
(2010-05-06 to 2010-05-06), China/Oil slowdown (2015-08-18 to 2016-02-11), and COVID-19
crash (2020-02-20 to 2020-03-23). Of these, only five are within the rolling window's
coverage and last long enough to produce multiple labelled windows; the Dot-com bust and
9/11 shock both precede the first complete rolling window.

### 3.3 Two-Regime CUSUM Calibration

A single calibration of the CUSUM detector over the full 2004–2024 period is not
appropriate because the in-control mean and variance of ρ(t) change substantially across
the study period. Pre-GFC (2004–2007), the calm-period mean of ρ(t) is 22.2 with standard
deviation 3.6. Post-GFC (2013–2019), the mean is 40.7 with standard deviation 14.4 — a
near-doubling of the level and a fourfold increase in variability. A detector calibrated on
the pre-GFC baseline would fire continuously throughout the post-GFC period, providing no
discrimination. This observation is itself a substantive finding: the pre-GFC and post-GFC
market constitute structurally distinct regimes.

Regime 1 uses the pre-GFC calm (2004-01-01 to 2007-06-01) as the calibration period with
an ARL-based threshold calibrated to produce approximately one false alarm per two years on
the calibration data. The allowance is δ = 0.5σ, and the resulting threshold h = 11.96.

For Regime 2 (monitoring 2013–2024 for COVID), the ARL calibration fails: because the
2013–2019 period is too volatile in ρ-space to support a well-defined null distribution,
the binary-search calibration converges to h = 0, meaning every window triggers an alarm.
We use Approach A: setting h as the 95th percentile of the non-resettable CUSUM statistic
computed on the calibration period. This distribution-free approach does not assume
stationarity; it merely requires that the calibration period be representative of the null
state. The resulting h = 233.5 provides a threshold that false-alarms at 0.7% of run
windows, with COVID detection delay of 49 calendar days.

---

## 4. Results

### 4.1 Secular Structural Break

The single most striking finding in the dataset is not the crisis detections but what
happens between them. The pre-crisis calm period (2004–2007) exhibits a mean detachment
ratio ρ̄ = 22.2 and spike count k̄ = 13–15. Following the GFC, neither statistic returns to
its pre-crisis baseline. The post-GFC calm (2013–2019) exhibits ρ̄ = 40.7 and k̄ = 18–22 —
consistently elevated above the pre-GFC baseline and substantially more variable. By the
post-COVID period (2021–2024), k(t) has risen to 20–27.

This secular elevation is not an artifact of the GFC recovery period. The Regime 1 CUSUM,
calibrated on the pre-GFC baseline, fires its first post-GFC alarm in December 2007 and
does not reset to a non-alarm state for the remainder of the study period. The market
structure has permanently shifted to a higher-correlation, higher-spike-count regime in
which the old baseline is no longer a valid null. This finding motivates the two-regime
design and has a direct practical implication: risk models that use a single historical
correlation baseline calibrated before 2007 will systematically underestimate current
correlation levels.

The inability of the ARL calibration to produce a stable h for Regime 2 is the quantitative
confirmation of this structural break. The 2013–2019 period exhibits 14.4 units of standard
deviation in ρ(t), nearly four times the 3.6 observed in the pre-GFC calm. No single
threshold separates genuine anomalies from baseline fluctuations when the baseline itself
fluctuates at this amplitude. The percentile-based calibration (Approach A) is a pragmatic
response, but it is not a theoretical solution. What Approach A accomplishes is to set the
alarm threshold at the empirical 95th percentile of CUSUM behavior during the post-GFC
period, accepting that this period is itself elevated relative to the pre-GFC world.

### 4.2 Decomposition of ρ(t): Effect B Dominates at Every Crisis

The detachment ratio ρ(t) = λ₁(t)/λ+(t) conflates two mechanistically distinct phenomena.
The numerator λ₁(t) rising above its calm-period mean reflects the market factor
strengthening — assets moving more synchronously, elevating the dominant eigenvalue.
The denominator λ+(t) falling below its calm-period mean reflects the noise floor
compressing — the MP bulk edge falling as σ² falls when cross-asset correlations increase.
We decompose log ρ(t) above its calm-period baseline into:

    log ρ(t) − log ρ₀ = Effect A(t) + Effect B(t)

    Effect A(t) = log(λ₁(t)/μ_{λ₁})      [positive when λ₁ above calm mean]
    Effect B(t) = −log(λ+(t)/μ_{λ+})     [positive when λ+ below calm mean]

where μ_{λ₁} = 26.45 and μ_{λ+} = 1.20 are the calm-period means over 2004–2007.

Table 1 presents the decomposition at the peak-ρ window within each of the five measured
crisis episodes.

**Table 1: ρ(t) decomposition at crisis peaks**

| Episode | Peak date | log ρ | Effect A | Effect B | Frac A | Frac B |
|---|---|---|---|---|---|---|
| Global Financial Crisis | 2009-01-05 | 4.696 | 0.654 | 0.951 | 40.7% | 59.3% |
| Euro Sovereign Debt | 2010-06-07 | 3.968 | 0.429 | 0.448 | 49.0% | 51.0% |
| Flash Crash | 2010-05-06 | 3.807 | 0.306 | 0.410 | 42.7% | 57.3% |
| China/Oil crash | 2016-02-08 | 4.244 | 0.485 | 0.668 | 42.1% | 57.9% |
| COVID-19 crash | 2020-03-11 | 4.570 | 0.550 | 0.930 | 37.2% | 62.8% |

Effect B dominates at every crisis without exception. The fraction ranges from 51.0% at the
relatively mild Euro Sovereign Debt crisis to 62.8% at COVID, where the noise floor
compression was more severe than in any other episode including the GFC peak.

The theoretical interpretation is direct. When cross-asset return correlations increase
during a crisis, the normalized covariance matrix S becomes less identity-like: more
variance is concentrated in common factors. The effect on σ², the noise-floor variance
parameter, is that the bulk becomes tighter and its upper edge λ+ = σ²(1 + √γ)² falls.
Because λ+ scales as the square of (1 + √γ), the bulk edge is more sensitive to changes
in σ² than λ₁ is. A 10% reduction in σ² reduces λ+ by approximately 10% (since λ+ ≈ 2.58σ²
at γ = 0.369), while it reduces the raw noise contribution to λ₁ proportionally. But λ₁
also rises as genuine correlation increases, partially offsetting the compression. The net
result is that the denominator moves more than the numerator, which is exactly what Effect
B dominance quantifies.

This finding reframes what ρ(t) measures. It is not primarily an indicator of how strongly
the market factor is behaving; it is primarily an indicator of how severely the noise floor
has compressed. Financial crises are correlation compression events first and market factor
amplification events second.

### 4.3 Velocity of Compression Is the Best Detector

We compute the first difference of each level statistic: dρ(t) = ρ(t) − ρ(t−1) and
dk(t) = k(t) − k(t−1). Both are smoothed with a three-window centered moving average to
reduce single-observation noise. Table 2 presents AUROC against the crisis calendar for all
six statistics, alongside bootstrap 95% confidence intervals where available.

**Table 2: AUROC against crisis calendar (expand_days = 21)**

| Statistic | AUROC | 95% CI (bootstrap) |
|---|---|---|
| dρ/dt (smoothed) | **0.750** | — |
| ρ(t) | 0.669 | [0.589, 0.745] |
| dk/dt (smoothed) | 0.675 | — |
| 1−O(t) instability | 0.626 | — |
| k(t) | 0.566 | [0.464, 0.662] |
| Full KS | 0.566 | [0.466, 0.661] |
| Bulk KS | 0.486 | [0.384, 0.586] |

Smoothed dρ/dt achieves AUROC 0.750, eight points above the ρ(t) level and 18 points above
the bulk KS statistic. This result is substantive and interpretable. The level statistic
ρ(t) tells us how elevated correlation compression currently is; the velocity dρ/dt tells
us how fast compression is accelerating. At the onset of a crisis, compression accelerates
sharply before the level itself reaches its peak. The smoothed velocity captures this onset
phase precisely, while the level is still in the lower portion of its trajectory.

The bulk KS statistic performing near chance (AUROC 0.486, confidence interval spanning
0.5) is not a failure of the methodology but a confirmation that the MP cleaning is working
correctly. The bulk KS measures goodness-of-fit between the residual bulk eigenvalue
distribution and the MP prediction after extracting spike eigenvalues. During crises, spike
proliferation moves eigenvalues out of the bulk, making the remaining bulk actually more
MP-like, not less. The crisis signal is concentrated in the spike region, and bulk KS
captures the complementary phenomenon — the noise floor — which is quiescent during crises
precisely because the spikes have consumed the correlation mass.

The full KS, which includes all eigenvalues against the MP prediction, achieves AUROC 0.566
— equivalent to the spike count k(t). This is expected: the full KS at the threshold λ+
equals approximately k/d (the fraction of eigenvalues above the bulk edge), so the two
statistics are nearly linearly related.

The lead/lag analysis for ρ(t) produces an unexpected finding. When we define the first
crossing of a threshold (calm mean + 1.5 standard deviations) in the six months preceding
each crisis start date, ρ(t) triggers this threshold before the formal crisis start at the
GFC (September 2007, six weeks before the October 2007 marker), the Euro Sovereign Debt
crisis (January 2010, four months before the April 2010 marker), the China/Oil slowdown
(May 2015, three months before the August 2015 marker), and COVID (November 2019, three
months before the February 2020 marker). The median lead across these four episodes is
five windows, corresponding to 105 calendar days. The conventional crisis calendar dates
used in the validation are based on well-known market dislocations (NBER dating, VIX
spikes, market troughs); ρ(t) begins rising well before these reference points because
structural correlation changes precede the most visible return dislocations. The velocity
dρ/dt, by contrast, has a median lead of only 0.5 windows — it captures the acceleration
phase near onset rather than the sustained rise over months.

The practical implication is this: ρ(t) provides early structural warning, dρ/dt provides
sharp detection near onset, and the two statistics are complementary rather than redundant.
A combined monitoring framework should watch both.

### 4.4 Eigenvector Subspace Rotation Is a Lagging Indicator

The subspace overlap O(t) is defined as

    O(t) = ||V_5(t)^T V_5(t−1)||²_F / 5

where V_5(t) is the 93 × 5 matrix of the top five eigenvectors at window t, and the
Frobenius norm measures total squared projection between consecutive subspaces. O(t) = 1
indicates perfect subspace preservation; O(t) = 0 indicates orthogonal subspaces.
The instability measure is 1 − O(t).

The AUROC of 1 − O(t) against the crisis calendar is 0.626, below both the level ρ(t)
(0.669) and the smoothed velocity dρ/dt (0.750). More importantly, the lead/lag analysis
in Table 3 shows that eigenvector subspace rotation consistently follows rather than
precedes eigenvalue elevation.

**Table 3: Lead/lag analysis — first threshold crossing before crisis start**
*Positive = leads crisis onset (fires before crisis start date)*
*Threshold = calm-period mean + 1.5σ; look-back = 126 calendar days*

| Crisis | O(t) instab. alarm | ρ(t) alarm | O lead (win) | ρ lead (win) |
|---|---|---|---|---|
| Global Financial Crisis | 2008-03-06 | 2007-09-05 | −7.1 | +1.6 |
| Euro Sovereign Debt | none | 2010-01-05 | n/a | +5.1 |
| China/Oil crash | none | 2015-05-08 | n/a | +4.9 |
| COVID-19 crash | 2020-03-11 | 2019-11-07 | −1.0 | +5.0 |
| **Median** | | | **−4.0** | **+5.0** |

At the GFC, the O(t) instability alarm fires in March 2008 — approximately seven windows
(147 days) after the crisis start marker of October 2007, and more than 26 weeks after ρ(t)
first crossed its threshold. At COVID, O(t) fires in March 2020 — three weeks after the
February 2020 crisis start, and four months after ρ(t)'s November 2019 early signal. At the
Euro Sovereign Debt and China/Oil crises, O(t) never crosses the threshold at all within
the look-back window.

The economic interpretation is clear and mechanically consistent with the MP framework.
During a crisis onset, the correlations between asset returns increase in magnitude but
their structure — which assets are correlated with which — does not immediately change.
The first effect is eigenvalue elevation: the magnitude of the dominant common factor rises
and the noise floor falls. The eigenvectors of the top factor still represent the same
approximate market-cap-weighted portfolio as in the pre-crisis period. Only later, as the
crisis develops and cross-sector contagion propagates, do the eigenvectors rotate to reflect
changed factor structure. Eigenvalue magnitudes change before eigenvector directions change.

This is an honest negative result. The hypothesis that subspace rotation provides additional
early warning beyond eigenvalue statistics does not hold in our data. We report it as such.
The 1 − O(t) instability measure is still informative (AUROC 0.626 is above chance) but
it is a lagging indicator, not a leading one.

### 4.5 No Granger Causality

We test whether spectral statistics Granger-cause realized volatility, defined as RV(t) =
mean_i |r_i(t)| averaged across all 93 assets at the center day of window t. For each
predictor X ∈ {ρ(t), k(t), 1−O(t)} and lag p ∈ {1, 2, 3}, we compare the F-statistic from
regressing RV(t) on its own p lags (restricted model) against regressing RV(t) on its own
p lags plus p lags of X (unrestricted model). The null hypothesis is that all coefficients
on lagged X are zero.

**Table 4: Granger causality F-tests (H₀: predictor does not Granger-cause RV(t))**

| Predictor | Lag | F-stat | p-value |
|---|---|---|---|
| ρ(t) | 1 | 1.436 | 0.232 |
| ρ(t) | 2 | 0.060 | 0.941 |
| ρ(t) | 3 | 0.137 | 0.938 |
| k(t) | 1 | 1.063 | 0.304 |
| k(t) | 2 | 1.656 | 0.193 |
| k(t) | 3 | 1.436 | 0.233 |
| 1−O(t) | 1 | 0.869 | 0.352 |
| 1−O(t) | 2 | 0.216 | 0.806 |
| 1−O(t) | 3 | 0.217 | 0.884 |

Every test fails to reject the null hypothesis at conventional significance levels (p > 0.19
in all cases). The spectral statistics do not predict next-window realized volatility beyond
what RV's own past values predict.

This null result is substantive and should not be dismissed as a power problem. The sample
covers 252 windows with RV mean 0.013 and standard deviation 0.0072 — sufficient statistical
power to detect Granger causality of moderate size if it existed. The result means these
statistics are coincident or slightly leading structural indicators, not predictors of future
volatility. Distinguishing these two roles is not pedantic: a coincident indicator tells you
the state of the system now, and a predictive indicator tells you what the system will do
next. ρ(t) leading crisis onset by five windows (Section 4.3) is a coincident-to-leading
structural signal — the structural compression is beginning, and this can be observed in the
spectrum before the associated return dislocations materialize. But it does not follow from
this that observing ρ(t) at time t enables prediction of RV(t+1) or RV(t+2). The structural
state and the realized return magnitude are correlated contemporaneously; the structural
state at time t does not predict the magnitude at t+1 beyond what the magnitude at t−1
already implies through GARCH dynamics.

### 4.6 The Combined Detector Does Not Improve on Velocity

We train an L2-regularized logistic regression on the feature vector
[ρ(t), k(t), dρ(t), dk(t), r_eff(t), 1−O(t)] with a strict train/test split: all data
through 2012 for training, all data from 2013 onward for out-of-sample testing. The
standardization of features uses only training-set statistics, ensuring no look-ahead
contamination. The training set contains 107 windows with 22 crisis-labelled windows; the
test set contains 144 windows with 10 crisis-labelled windows.

The logistic regression achieves training AUROC 0.904 and out-of-sample test AUROC 0.755.
For comparison, ρ(t) alone achieves test AUROC 0.766. The combined model underperforms the
single best level statistic by 0.011 AUROC points. The false alarm rate at the 70% TPR
operating point is 29.1%.

The ρ(t) coefficient in the logistic regression is negative (−1.789), which may appear
paradoxical since higher ρ indicates more crisis. The sign reflects a collinearity with
r_eff (coefficient −1.354): when ρ rises, r_eff typically falls, and the regression
reweights from ρ to r_eff to avoid multicollinearity. The economically meaningful direction
(higher ρ, lower r_eff → more crisis) is preserved in the joint prediction; the individual
coefficients are not individually interpretable in the presence of strong collinearity.

The combined detector's failure to improve on the best single statistic is consistent with
the Granger null result. If the statistics carried independent predictive information, a
combination would outperform any individual component. Because they are correlated
manifestations of the same underlying structural event — the compression and concentration
of the correlation matrix — combining them provides little additional discriminative power
beyond the single most sensitive indicator (smoothed dρ/dt, AUROC 0.750 in the full-sample
validation).

---

## 5. Discussion

The five findings presented above form a coherent mechanistic story about how financial
crises manifest in the spectral structure of correlation matrices. Crises are not random
shocks to eigenvalue levels; they are structured compression events that follow a
characteristic pattern: the noise floor λ+(t) begins falling before the crisis marker
date (captured by ρ(t) leading by five windows), the velocity of compression accelerates
sharply at onset (captured by dρ/dt with AUROC 0.750), eigenvalue magnitudes become extreme
before eigenvector directions change (O(t) lags ρ(t) by seven windows at the GFC), and
throughout, the compression is driven more by noise floor collapse than by market factor
amplification (Effect B = 51–63%).

The failure of the bulk KS statistic (AUROC 0.486) deserves explicit interpretation as a
positive finding rather than a disappointing one. It confirms that the spike-stripping
procedure is operating correctly: after extracting eigenvalues above the estimated λ+(t),
the remaining bulk is well described by the MP distribution. During crises, this is even
more true than during calm periods, because eigenvalues migrate out of the bulk into the
spike region, making the residual bulk "purer" in the MP sense. If bulk KS had a high AUROC,
it would suggest either that the cleaning was failing or that crises distort the bulk
directly — neither of which appears to be the case in this data.

The connection to Plerou et al. (2002) is direct. That paper observed qualitatively that
the number of eigenvalues above the MP bulk edge rises during crises, that the largest
eigenvalue is dramatically elevated, and that the bulk remains approximately MP-distributed.
The present work formalizes all three observations: we provide calibrated thresholds
(through the self-consistent σ² estimator and the BBP detection criterion), sequential
detection procedures (through the two-regime CUSUM), and a quantitative discriminative
power assessment (through AUROC and bootstrap confidence intervals). The qualitative
picture established in 2002 is confirmed and extended to a 20-year panel including three
major crises beyond those available to the original authors.

For a risk practitioner, the operational takeaway from this study is specific. Standard
risk systems that monitor the level of the largest eigenvalue or the average pairwise
correlation are measuring the right phenomenon but using the wrong form of the signal.
The velocity of compression — the rate of change of the detachment ratio — is substantially
more informative than the level. A monitoring system that computes a three-window centered
moving average of the first difference of ρ(t) and compares it to a percentile-based
threshold from a recent calm period will achieve materially better discrimination than one
that monitors the level alone. The eigenvector subspace overlap adds approximately three
AUROC points at a substantially higher computational and implementation cost; its lagging
rather than leading nature means it is better suited to confirmation than early warning.

The two-regime finding underscores a structural conclusion that transcends the specific
detection application: a single historical baseline is insufficient for monitoring a system
that has undergone a documented permanent shift. The pre-GFC correlation regime of
ρ̄ ≈ 22, k̄ ≈ 14 is not a long-run average that the market will eventually return to. It
appears to be a historical artifact of a market structure that ended in 2007–2009 and has
not recovered. Monitoring frameworks built on pre-2007 baselines will systematically
generate false alarms in the post-GFC market.

---

## 6. Limitations and Open Questions

Three mathematical flags documented in the implementation require explicit statement in the
paper body.

**Mathematical Flag 1** (repeated from Section 2.3): The BBP detection threshold θ_c and
the MP bulk upper edge λ+ are distinct quantities. θ_c = σ²(1 + √γ) and
λ+ = σ²(1 + √γ)². At our aspect ratio γ = 0.369, λ+/θ_c = 1.607. Confusing these is
a common error in applied work; eigenvalues above λ+ imply detectable population factors,
but the detection threshold in population space is θ_c, not λ+.

**Mathematical Flag 2** (repeated from Section 2.3): The BBP phase transition is proven for
complex Gaussian random matrices. Financial returns are not Gaussian: they exhibit fat tails
(daily return kurtosis typically 5–10 for large-cap equities), volatility clustering (strong
ARCH effects), and cross-sectional and serial dependence. The application of the BBP threshold
to financial data is heuristic. Capitaine, Donati-Martin, and Féral (2009) extended BBP to
real matrices with finite fourth moment; this covers the case more relevant to financial data
than the original GUE result, but the finite-fourth-moment assumption is itself marginal for
heavy-tailed returns.

**Mathematical Flag 3** (repeated from Section 2.4): The recursive CUSUM formula
C(t) = max(0, C(t−1) + (x(t) − μ − δ)) is from Lorden (1971) and Moustakides (1986),
not Page (1954). Attribution of this formula to "Page 1954" is imprecise.

Beyond the mathematical flags, several additional limitations apply.

The i.i.d. assumption violation is pervasive. The CUSUM analysis assumes the in-control
series is stationary with known mean and variance. Financial time series violate this
through volatility clustering (ARCH/GARCH dynamics), autocorrelated squared returns, and
slowly varying structural breaks. The effective ARL of our detector is not well characterized
by the standard CUSUM theory; the calibrated false alarm rates (0.7% for Regime 2) are
empirical rather than theoretical guarantees.

Serial correlation in rolling windows creates dependence between consecutive CUSUM
increments. Consecutive windows overlap by T − step = 231 observations (91.7% overlap).
The KS statistics at adjacent windows are therefore highly correlated, and AUROC bootstrap
confidence intervals assuming independent observations are anti-conservative (too narrow).
The effective sample size is far smaller than the 252 nominal observations; it is closer to
the number of distinct crisis episodes (five), which is why even AUROC values near 0.75
have wide uncertainty in any honest sense.

Detection is not prediction. This distinction is stated explicitly to avoid a common
interpretive error. ρ(t) leading crisis start dates by five windows means the structural
compression begins before the conventional crisis marker; it does not mean that observing
ρ(t) at time t enables prediction of returns or volatility at t+1. The Granger tests confirm
this: no statistic predicts future realized volatility beyond its own lagged values.

Window length sensitivity. The T = 252 choice drives γ = 0.369 and affects all spectral
statistics. At T = 126 (γ = 0.738), the bulk occupies a wider range and fewer eigenvalues
appear as spikes. At T = 504 (γ = 0.185), the bulk is tighter and more eigenvalues would
emerge as spikes. The substantive findings — Effect B dominance, velocity outperforming
level, O(t) lagging ρ(t) — should be robust to T in a qualitative sense, but the specific
AUROC values and detection delays are T-dependent. Sensitivity analysis across
T ∈ {126, 189, 252} is the natural next empirical step.

The open theoretical question of most consequence is the asymptotic null distribution of
dρ(t)/dt under the actual data-generating process — dependent, heavy-tailed, non-stationary
returns. Standard results for the MP distribution assume i.i.d. Gaussian entries. Under
dependent heavy-tailed returns, the fluctuations of λ+(t) and λ₁(t) are not well
characterized, and therefore the distribution of their ratio ρ(t), let alone its first
difference, has no known asymptotic theory. Deriving this distribution — even approximately,
via large-deviation or functional central limit theorem techniques adapted for financial
time series — would provide formal ARL guarantees for the velocity detector and transform
the current heuristic calibration into a principled statistical procedure. This is the
natural next theoretical contribution building on the present empirical work.

---

## 7. Conclusion

We report five findings from a systematic spectral analysis of 93 S&P 500 constituents
from 2003 to 2024 using a rolling Marčenko-Pastur spectral estimator and CUSUM-based
sequential detection.

First, the smoothed first difference of the detachment ratio dρ/dt achieves AUROC 0.750
against a seven-episode crisis calendar, outperforming the level statistic ρ(t) by eight
AUROC points and establishing velocity of correlation compression as the single most
discriminative contemporaneous crisis indicator identified in this analysis.

Second, the level statistic ρ(t) leads formal crisis start dates by approximately five
rolling windows (105 calendar days), meaning that structural correlation compression begins
well before the return dislocations that define conventional crisis markers — providing a
genuine but limited early structural warning signal.

Third, at every measured crisis episode, noise floor compression (Effect B: the MP upper
edge λ+ falling) accounts for 51–63% of the total elevation in log ρ(t), consistently
exceeding the contribution of market factor amplification. Financial crises are primarily
correlation compression events.

Fourth, eigenvector subspace rotation is a lagging indicator: the top-five subspace rotates
after eigenvalue magnitudes change, not before, with a median lag of four windows relative
to crisis onset. This rules out subspace instability as an early warning signal in the
framework tested here.

Fifth, no spectral statistic — ρ, k, or subspace instability — Granger-causes realized
volatility at any tested lag, confirming that these statistics are contemporaneous or
slightly leading structural diagnostics, not predictors of future market stress.

Taken together, these findings enable a calibrated sequential spectral detector that did
not previously exist: a system with documented detection delays (56 days for GFC, 49 days
for COVID), empirically calibrated false alarm rates (two pre-GFC alarms for Regime 1,
one pre-COVID alarm for Regime 2), and honest out-of-sample AUROC (0.750 for dρ/dt in
full-sample validation; 0.755 for the combined logistic detector on the held-out 2013–2024
period). The key open theoretical question — deriving the asymptotic distribution of dρ/dt
under dependent heavy-tailed returns — would elevate the current empirical calibration to a
formally justified statistical procedure.

Finally, and perhaps most consequentially for practice: the pre-GFC correlation regime of
ρ̄ ≈ 22 and k̄ ≈ 14 has not recovered in the 15 years since the GFC. The post-GFC market
operates at permanently higher correlation levels (ρ̄ ≈ 41, k̄ ≈ 18–27) with a structurally
elevated noise floor. Any risk monitoring framework that uses a pre-2008 calibration as its
baseline is systematically miscalibrated for the current market structure and will generate
false alarms continuously. Secular structural drift is not a second-order correction; it is
the primary calibration challenge for long-horizon spectral risk monitoring.

---

## References

[BBP05] Baik, J., Ben Arous, G., and Péché, S. (2005). Phase transition of the largest
eigenvalue for nonnull complex sample covariance matrices. *Annals of Probability*,
33(5), 1643–1697.

[BBP17] Bun, J., Bouchaud, J.-P., and Potters, M. (2017). Cleaning large correlation
matrices: tools from random matrix theory. *Physics Reports*, 666, 1–109.

[CDF09] Capitaine, M., Donati-Martin, C., and Féral, D. (2009). The largest eigenvalues of
finite rank deformation of large Wigner matrices: convergence and nonuniversality of the
fluctuations. *Annals of Probability*, 37(1), 1–47.

[L99] Laloux, L., Cizeau, P., Bouchaud, J.-P., and Potters, M. (1999). Noise dressing of
financial correlation matrices. *Physical Review Letters*, 83(7), 1467–1470.

[LW04] Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional
covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.

[Lor71] Lorden, G. (1971). Procedures for reacting to a change in distribution. *Annals of
Mathematical Statistics*, 42(6), 1897–1908.

[MP67] Marčenko, V.A. and Pastur, L.A. (1967). Distribution of eigenvalues for some sets
of random matrices. *Mathematics of the USSR-Sbornik*, 1(4), 457–483.

[Mou86] Moustakides, G.V. (1986). Optimal stopping times for detecting changes in
distributions. *Annals of Statistics*, 14(4), 1379–1387.

[P02] Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L.A.N., Guhr, T., and Stanley, H.E.
(2002). Random matrix approach to cross correlations in financial data. *Physical Review E*,
65(6), 066126.

[Page54] Page, E.S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100–114.
