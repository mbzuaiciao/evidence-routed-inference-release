# Phase 1 Mathematical Audit

This audit covers the deterministic, unannualized per-period Sharpe inference
implemented on `feature/phase1-statistical-core`. It distinguishes exact
functional identities from asymptotic approximations and finite-sample
implementation conventions.

## HAC influence-function inference

For the functional $S=\mu/\sigma=\mu/v^{1/2}$, the influence functions of
$\mu$ and $v$ are

$$
IF_\mu(r)=r-\mu,
\qquad
IF_v(r)=(r-\mu)^2-v.
$$

The gradient of $g(\mu,v)=\mu v^{-1/2}$ is

$$
\frac{\partial g}{\partial\mu}=\frac{1}{\sigma},
\qquad
\frac{\partial g}{\partial v}=-\frac{\mu}{2\sigma^3}
=-\frac{S}{2\sigma^2}.
$$

Therefore the exact influence-function identity is

$$
\psi_t=\frac{r_t-\mu}{\sigma}
-\frac{S}{2\sigma^2}\left((r_t-\mu)^2-\sigma^2\right).
$$

`influence_function_series` in `src/eri/inference/hac.py` maps these terms as
follows:

- `mean` estimates $\mu$;
- `variance = mean(centered**2)` estimates $\sigma^2$ with denominator $n$;
- `functional_sharpe = mean / sigma` estimates the functional $S$ using the
  same empirical-distribution moments;
- the two array terms implement the two terms of $\psi_t$;
- subtracting the influence mean enforces the zero-mean series used by the HAC
  covariance calculation, up to floating-point roundoff.

For explicit bandwidth $L$, `bartlett_long_run_variance` computes

$$
\widehat\gamma_k
=\frac{1}{n}\sum_{t=k+1}^{n}\widehat\psi_t\widehat\psi_{t-k},
\quad
w_k=1-\frac{k}{L+1},
\quad
\widehat\Omega=\widehat\gamma_0+2\sum_{k=1}^Lw_k\widehat\gamma_k.
$$

The denominator is $n$ at lag zero and every positive lag. The long-run
variance is not divided by $n$ inside this function. Only
`hac_sharpe_standard_error` applies

$$
\widehat{\operatorname{Var}}(\widehat S)=\widehat\Omega/n,
\qquad
\widehat{SE}(\widehat S)=\sqrt{\widehat\Omega/n}.
$$

Thus there is no double division by sample size. With bandwidth zero,
$\widehat\Omega=\widehat\gamma_0$, which is checked both directly and against
a separately coded influence/Newey-West calculation.

The reported point estimate uses the repository's `ddof=1` sample Sharpe,
whereas the influence plug-in uses empirical-distribution (`ddof=0`) moments.
That difference is $O(n^{-1})$ and is explicitly documented. Materially
negative HAC variance is rejected using a bound scaled to lag-zero variance;
no negative value is silently converted into a valid result.

## IID Gaussian inference

Under IID Gaussian asymptotics, the delta method gives

$$
\widehat S\ \dot\sim\ N\!\left(S,\frac{1+S^2/2}{n}\right).
$$

`iid_gaussian_sharpe_variance` implements this approximation. Confidence
intervals use the observed-Sharpe plug-in variance. A benchmark test of
$H_0:S=S_0$ uses the null variance:

$$
Z_0=\frac{\widehat S-S_0}{\sqrt{(1+S_0^2/2)/n}}.
$$

`greater`, `less`, and `two-sided` map to the upper tail, lower tail, and twice
the smaller symmetric tail respectively. All inputs and outputs are explicitly
unannualized per-period Sharpes. The audit corrected the earlier test code,
which evaluated the standard error at the observed Sharpe instead of the null
benchmark.

## Mertens variance and PSR

For skewness $\gamma_3$ and raw kurtosis $\gamma_4$, IID non-Gaussian
delta-method variance is approximated by

$$
\operatorname{Var}(\widehat S)
\approx\frac{1-\gamma_3S+(\gamma_4-1)S^2/4}{n}.
$$

`summarize_returns` uses uncorrected central moments
$m_k=n^{-1}\sum_t(r_t-\bar r)^k$, with
$\widehat\gamma_3=m_3/m_2^{3/2}$ and
$\widehat\gamma_4=m_4/m_2^2$. Kurtosis is raw, not excess; no finite-sample
bias correction is applied. `mertens_sharpe_variance` uses denominator $n$
and evaluates the expression at the observed `ddof=1` sample Sharpe.

For benchmark $S_0$, `probabilistic_sharpe_ratio` computes

$$
PSR(S_0)=\Phi\left((\widehat S-S_0)/\widehat{SE}_M\right).
$$

This is a frequentist normal-approximation tail probability, not a posterior
probability. Setting skewness to zero and raw kurtosis to three reduces the
variance exactly to $(1+S^2/2)/n$.

## Circular-block bootstrap

For every draw, the generator samples
$\lceil n/\ell\rceil$ block starts. `circular_block_indices` adds offsets
$0,\ldots,\ell-1$, applies modulo $n$, concatenates blocks, and truncates
to exactly $n$ indices. `circular_block_bootstrap` recalculates the
unannualized `ddof=1` sample Sharpe and uses explicit linear-interpolation
empirical quantiles at $(1-c)/2$ and $1-(1-c)/2$.

The seed, block length, replication count, confidence level, quantile method,
and annualization convention are recorded. A degenerate resample now raises an
error identifying its replication rather than producing, clipping, or skipping
a non-finite draw.

## Gaussian power and minimum length

For $H_0:S\leq S_0$, $H_1:S>S_0$, define

$$
SE_0=\sqrt{(1+S_0^2/2)/n},
\qquad
SE_1=\sqrt{(1+S_1^2/2)/n}.
$$

The asymptotic size-$\alpha$ rejection threshold is

$$
c_\alpha=S_0+z_{1-\alpha}SE_0.
$$

Under the assumed true Sharpe $S_1$, the independently derived power is

$$
P_{S_1}(\widehat S>c_\alpha)
=\Phi\left((S_1-c_\alpha)/SE_1\right).
$$

The audit corrected the earlier simplified formula, which effectively replaced
the null-to-alternative standard-error ratio by one. `PowerResult` now records
both standard errors and the rejection threshold. `minimum_sample_length`
uses exponential bracketing followed by binary search, returns the smallest
integer meeting target power, errors if its explicit maximum cannot meet the
target, and converts periods to years only as `n / periods_per_year`.

## Remaining limitations

Gaussian, Mertens, PSR, HAC, and power results are asymptotic. HAC additionally
requires sufficient weak dependence and stationarity. Circular blocks do not
solve arbitrary nonstationarity. Moment plug-ins can be unstable in small
samples. No inferential method supports annualized inputs or outputs; only the
separate descriptive Sharpe function annualizes explicitly.
