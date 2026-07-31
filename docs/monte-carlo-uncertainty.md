# Monte Carlo uncertainty in Phase 3B

## Binomial metrics

Phase 3B reports two-sided Wilson score intervals for selection, abstention,
coverage, oracle-agreement, and execution-failure proportions. For $x$
successes in $n$ eligible observations, $\hat p=x/n$, confidence level
$1-\alpha$, and $z=z_{1-\alpha/2}$, the Wilson center and half-width are

$$
c = \frac{\hat p + z^2/(2n)}{1+z^2/n},
$$

$$
h = \frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}.
$$

The reported interval is $[\max(0,c-h),\min(1,c+h)]$. Every row records its
numerator and denominator. Selective coverage excludes abstentions and rows
whose inferential target is undefined; it is never converted to zero when not
applicable.

Zero successes have an exact reported lower bound of zero and all successes
have an exact upper bound of one. A zero denominator is not assigned an
interval. Failed non-abstaining attempts on a defined target enter selective
coverage as uncovered; execution failures are reported separately.

## Mean regret and runtime

For observations $Y_1,\ldots,Y_m$, the row-level Monte Carlo interval uses

$$
\bar Y \mathbin{\pm} z_{1-\alpha/2}\frac{s_Y}{\sqrt m}.
$$

Scenario-level output uses individual scenario/draw rows. Aggregate output
first computes one mean per scenario and then applies the same normal interval
to those eight scenario means. This scenario-clustered summary gives each
registered scenario equal weight and does not treat four system rows from a
shared simulation as independent replicates.

Normal intervals are descriptive Monte Carlo approximations. There are only
eight scenario clusters, and coverage comparisons can have only six because
the two structural-break targets are undefined. The normal critical value is
not an exact small-cluster reference distribution. Neither these intervals nor
the row-level Wilson intervals establish generalization outside the fixed
scenario set; signs and apparent significance must be accompanied by the
scenario-level effects.

## Paired effects

Systems are compared on identical `(scenario_id, evaluation_seed)` cells.
Router-minus-comparator differences are formed first, then averaged. Invalid
selection, abstention, and regret use all shared cells. Coverage uses only cells
where the target is defined and both systems have valid coverage outcomes.
Method eligibility is not a coverage exclusion: an ineligible fixed method can
still produce an interval whose empirical coverage is scientifically relevant.
Each output records paired cells, excluded cells, denominator text, standard
error, confidence interval, and whether the aggregate is scenario-clustered.

No unpaired test or decorative p-value is reported when the paired effect is
available. The intervals quantify simulation variability for the registered
design; they do not correct for design selection or multiple exploratory
queries.
