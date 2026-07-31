# Statistical Conventions

## Return inputs

Inputs are one-dimensional finite **excess returns** unless the caller
explicitly documents another convention. Empty inputs, fewer than two
observations for variance-based statistics, NaN, infinity, and zero or
numerically degenerate variance are rejected.

Sampling frequency is never inferred from row count. A caller must supply a
positive finite `periods_per_year` value whenever results are expressed on an
annual scale.

## Sample statistics

For returns $r_1,\ldots,r_n$, the arithmetic mean and sample standard
deviation are

$$
\bar r = \frac{1}{n}\sum_{t=1}^n r_t,
\qquad
s = \sqrt{\frac{1}{n-1}\sum_{t=1}^n(r_t-\bar r)^2}.
$$

The unannualized sample Sharpe is

$$
\widehat S = \frac{\bar r}{s}.
$$

Thus the reported standard-deviation convention is `ddof=1`. Internal HAC
influence-function calculations plug the empirical distribution into the
Sharpe functional: variance is `mean((r - mean(r))**2)` with denominator
$n$, and the corresponding functional Sharpe uses that population-form
standard deviation. The influence series is then centered before estimating
autocovariances. Its difference from the reported `ddof=1` Sharpe is
$O(n^{-1})$, an explicit asymptotic implementation convention.

## Annualization

Square-root-of-time annualization is performed only by an explicit call with
an explicit frequency:

$$
\widehat S_{annual} = \sqrt{P}\,\widehat S,
$$

where $P$ is supplied periods per year. Serial dependence can invalidate
this naive transformation. All Phase 1 inferential methods operate on the
unannualized per-period Sharpe and label their result objects accordingly.

## Higher moments

Skewness and kurtosis are uncorrected plug-in central moments. Each
$m_k=n^{-1}\sum_t(r_t-\bar r)^k$, and the standardized moments are

$$
\widehat\gamma_3 = \frac{m_3}{m_2^{3/2}},
\qquad
\widehat\gamma_4 = \frac{m_4}{m_2^2}.
$$

`raw_kurtosis` means Pearson/raw kurtosis, not excess kurtosis. A Gaussian
population has raw kurtosis three. No finite-sample bias correction is applied.
The Mertens variance uses denominator $n$, the observed `ddof=1` sample
Sharpe, and these uncorrected moment estimates.
