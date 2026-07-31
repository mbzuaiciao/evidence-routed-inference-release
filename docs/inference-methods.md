# Inference Methods

All methods are deterministic typed Python functions. No orchestration,
network service, agent framework, or global random state is involved.

## IID Gaussian approximation

For sample size $n$, Phase 1 uses

$$
\widehat{\operatorname{Var}}(\widehat S)
= \frac{1 + \widehat S^2/2}{n},
\qquad
\widehat{SE}(\widehat S)
= \sqrt{\widehat{\operatorname{Var}}(\widehat S)}.
$$

A two-sided plug-in confidence interval evaluates the variance at the observed
$\widehat S$ and uses
$\widehat S \pm z_{(1+c)/2}\widehat{SE}(\widehat S)$. Benchmark tests instead
evaluate the null standard error at the unannualized benchmark $S_0$:

$$
Z_0=\frac{\widehat S-S_0}{\sqrt{(1+S_0^2/2)/n}}.
$$

The `greater` alternative uses $1-\Phi(Z_0)$, `less` uses $\Phi(Z_0)$, and
`two-sided` uses $2[1-\Phi(|Z_0|)]$. These are first-order asymptotic
approximations, not exact finite-sample identities, and may be inaccurate in
small samples, under non-Gaussianity, or under dependence. Tests, intervals,
estimates, and benchmarks all remain on the unannualized per-period scale.

## Mertens variance and PSR

Using uncorrected plug-in sample skewness $\widehat\gamma_3$ and **raw**
kurtosis $\widehat\gamma_4$, computed from central moments with denominator
$n$, the non-Gaussian IID variance approximation is

$$
\widehat{\operatorname{Var}}_M(\widehat S)
= \frac{1-\widehat\gamma_3\widehat S
+(\widehat\gamma_4-1)\widehat S^2/4}{n}.
$$

When skewness is zero and raw kurtosis is three, this reduces exactly to the
Gaussian expression. Against benchmark $S_0$, the Probabilistic Sharpe Ratio
is

$$
PSR(S_0)=\Phi\!\left(\frac{\widehat S-S_0}
{\sqrt{\widehat{\operatorname{Var}}_M(\widehat S)}}\right).
$$

Despite its name, PSR is a frequentist normal-approximation probability. It is
not a Bayesian posterior probability. The variance is evaluated at the
observed `ddof=1` sample Sharpe and uses denominator $n$; there is no
finite-sample moment bias correction.

## HAC influence-function uncertainty

Let $e_t=r_t-\mu$, $\sigma^2=E[e_t^2]$, and $S=\mu/\sigma$. The Sharpe
influence function identity is

$$
IF_t = \frac{e_t}{\sigma}
-\frac{S}{2\sigma^2}(e_t^2-\sigma^2).
$$

The second term makes dependence in squared returns part of the uncertainty;
the method is not merely a correction to the mean. With explicit bandwidth
$L$, lag covariance
$\widehat\gamma_k=n^{-1}\sum_{t=k+1}^n IF_t IF_{t-k}$, and Bartlett weights,

$$
\widehat\Omega
=\widehat\gamma_0
+2\sum_{k=1}^L\left(1-\frac{k}{L+1}\right)\widehat\gamma_k,
\qquad
\widehat{SE}_{HAC}=\sqrt{\widehat\Omega/n}.
$$

`influence_function_series` plugs in the sample mean and population-form
variance with denominator $n$, then centers the resulting series.
`bartlett_long_run_variance` divides every lag product sum by $n$, including
positive lags; it does not divide the LRV by $n$. Only
`hac_sharpe_standard_error` applies the final estimator scaling
$\widehat\Omega/n$. Bandwidth zero is therefore exactly the lag-zero
influence variance under this convention.

The implementation rejects any nonpositive LRV. A negative value more than
$100\epsilon\max(1,\widehat\gamma_0)$ below zero receives a distinct
"materially negative" error; no negative value is returned or silently
clipped into a valid uncertainty estimate. The estimator targets asymptotic
uncertainty under weak dependence and sufficient stationarity; it is not an
exact finite-sample solution. The influence series and bandwidth are retained
in `HACResult`.

## Circular-block bootstrap

Each bootstrap sample concatenates circular blocks starting at uniformly drawn
indices. `circular_block_indices` applies modulo-$n$ indexing and truncates
the concatenated indices to exactly $n$ observations before recalculating
the sample Sharpe.
Block length, replication count, and random seed are mandatory. The interval
uses NumPy's linear-interpolation empirical quantiles at $(1-c)/2$ and
$1-(1-c)/2$. Draws may be retained explicitly. If any resample has
degenerate variance, the function raises with the replication number instead
of emitting or skipping a non-finite draw.

The procedure can preserve short-range dependence when block length is
sensible, but it is not universally valid under arbitrary nonstationarity.
There are no automatic expensive defaults.

## Gaussian power and minimum length

For the one-sided test $H_0:S\leq S_0$ versus $H_1:S>S_0$, assumed true
Sharpe $S_1$, size $\alpha$, and sample size $n$, define

$$
SE_0=\sqrt{\frac{1+S_0^2/2}{n}},
\qquad
SE_1=\sqrt{\frac{1+S_1^2/2}{n}}.
$$

The asymptotic null rejection threshold and alternative power are

$$
c_\alpha=S_0+z_{1-\alpha}SE_0,
\qquad
\operatorname{Power}=\Phi\!\left(\frac{S_1-c_\alpha}{SE_1}\right).
$$

`minimum_sample_length` performs a bounded monotone integer search for the
smallest $n$ meeting target power. If periods per year are supplied, years
are reported as $n/P$; frequency is never inferred. No HAC or bootstrap
power claim is made. This is a Gaussian asymptotic approximation, not the exact
finite-sample distribution of the studentized sample Sharpe.
