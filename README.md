# When can machine learning beat a fixed rule?

**Simulation with known truth and a pre-registered test on JPMaQS duration, FX and equity returns**

Di Liu · November 2026

A theory-signed composite of macro indicators is hard to beat with machine learning. Is that because learning
cannot help with macro signals, or because the conditions it needs are absent? This project answers in two parts:

* **Part A: simulation.** On panels where the true relationship is known, find *when* a learner should beat the rule.
* **Part B: real data.** Test whether it *does*, on JPMaQS duration, FX and equity returns, with the analysis
  fixed before the run.

It uses the [macrosynergy](https://github.com/macrosynergy/macrosynergy) package and the public JPMaQS dataset on
Kaggle. Everything is in one executed notebook, [`ml_vs_rules.ipynb`](ml_vs_rules.ipynb).

## Summary

* **Simulation (432 panels).** Learners beat the rule only when the rule is mis-specified in a way a constrained
  model can represent: unequal weights, or signs that flip by country or regime. Even then they need about
  **24 markets × 25 years**, and the edge is about **+0.02 in out-of-sample IC**. Gradient boosting is the weakest
  learner throughout.
* **Real data (2007–2023).** Four learners against theory-signed rules in duration (24 markets), FX (23) and
  equity (16). **None of 16 comparisons passes the pre-registered test**: the largest t-statistic against a rule
  is 1.51, against a threshold of 2.96.
* **Two traps that headline Sharpe ratios hide.** In FX the learners lead only because the rule loses money; no
  FX signal is significant. In equity the learners' higher Sharpe is mostly long-equity beta; after removing it,
  the rule is best (0.53 against 0.26–0.47).

**Takeaway: structure beats flexibility.** Let theory fix the signs and the data set the weights; test a learner
only for a specific, economically motivated interaction; judge it on IC, sub-panel consistency and beta-neutral
performance, not headline Sharpe.

## The contenders

**The fixed rule: an equal-weight average of theory-signed z-scores.** It is built in three steps:

1. **Z-score:** express each indicator as "how unusual is it now?", in units of its typical deviation, using only
   data known at the time. +2 is unusually high; 0 is normal.
2. **Theory-signed:** flip the sign where economic theory says high is bad for the trade. For receiving duration,
   inflation above target is bad (minus); a high real yield is good (plus).
3. **Equal-weight average:** add the signed scores and divide by their number. Nothing is estimated from returns.

*Example:* inflation +1.5 (flipped to −1.5), growth +0.5 (flipped to −0.5), real yield +1.0 → average
(−1.5 − 0.5 + 1.0) / 3 ≈ −0.33, a mild signal to pay rather than receive.

| Contender | What it does | What it assumes |
|---|---|---|
| Rule | Equal-weight average of theory-signed z-scores | Signs from theory, equal weights, linear |
| Sign-constrained | Non-negative least squares / non-negative Ridge on the signed features | Signs from theory; learns the weights |
| Ridge (free signs) | Ridge on all features | Nothing about signs |
| Linear + interactions | Ridge plus feature × context terms | Is told *which* interactions may exist |
| Gradient boosting | Shallow boosted trees on features plus context | Must discover thresholds and interactions itself |

**Metric.** The information coefficient (IC): the correlation between a signal and next-month returns. Around
0.05 is worthwhile for macro signals and 0.10 strong.

## Part A: simulation with known truth

**Design.** N markets × T months of seven persistent macro features (AR(1), φ = 0.9, with a common factor).
Next-month returns are a true signal plus noise, scaled so the best attainable monthly R² is 0.5%, 2% or 8%
(real macro signals sit around 1–3%). Theory says every feature has a positive effect, which is true *on
average* in every scenario. Six truths, from the rule's home ground to cases built for learning:

| Scenario | Truth |
|---|---|
| Rule exactly right | Equal weights, linear, the same in every market |
| Thresholds | Each feature matters only beyond \|x\| > 1 |
| Unequal weights (7) | Correct signs, weights 3 : 2 : 1 : 0.5, three irrelevant features |
| Unequal weights (30) | Thirty features, only six relevant |
| Opposite signs by country | For 3 of 7 features the sign is + in half the markets and − in the other half (e.g. oil exporters vs importers); country type observable |
| Regime-dependent sign | For 3 of 7 features the effect is strong in a "high" regime and reversed in a "low" one; regime observable |

Panel sizes 4 × 180, 10 × 240, 24 × 300 and 48 × 480 (markets × months); six simulated panels per cell; every
learner refitted every 12 months on an expanding window and evaluated strictly out of sample.

![Phase diagram](figures/phase.png)

*Learner minus rule out-of-sample IC (top number) and share of simulated panels the learner wins (bottom).
Blue = learner better. The real data sit near the 24-market, R² 2% column.*

| Truth | Result |
|---|---|
| Rule exactly right | No learner wins on average in any cell; a learner can at best rediscover equal weights, with noise |
| Thresholds | The linear rule captures most of the effect (IC 0.13 vs oracle 0.15 at 24 markets, R² 2%); boosting draws level only at 48 markets, R² 8% |
| Unequal weights | Sign-constrained learner wins 100% of panels from 24 markets at R² 2% (+0.019 IC with 7 inputs, +0.018 with 30) |
| Opposite signs by country | Linear + interactions wins 100% from 24 markets at R² 2% (+0.023 IC), +0.06–0.08 at R² 8%; boosting still loses at R² 2% even at 48 × 480 |
| Regime-dependent sign | Same pattern, weaker: interactions win from 24 markets at R² 8%, or 48 at R² 2% |
| Any truth, 4 markets | The rule wins almost everywhere |

**Prediction for Part B.** At real-data scale a learner can beat a theory-signed rule only if the rule mis-weights
its inputs or misses sign heterogeneity, and then by about 0.02 in IC: small relative to sampling noise over
17 years.

## Part B: real data, pre-registered

**Data.** Kaggle JPMaQS (version 17), point-in-time, 2000–2023: 3.39 million rows, 24 markets (10 developed,
14 emerging).

| Asset class | Target return | Markets |
|---|---|---|
| Duration | `DU05YXR_VT10`: receive fixed in a 5-year interest-rate swap | 24 |
| FX | `FXXR_VT10`: long local currency vs its benchmark (USD or EUR) | 23 (no USD; FX-targeted periods excluded) |
| Equity | `EQXR_VT10`: long local equity index future | 16 |

All returns are excess returns scaled to 10% volatility; untradable periods are excluded.

**Features and theory signs.** The same nine sequential, point-in-time z-scores for every asset class. A sign of 0
means theory is unclear, so the rule leaves the feature out (8 used for duration, 3 for FX, 5 for equity).

| Feature | Meaning | Duration | FX | Equity |
|---|---|---|---|---|
| `XCPIC` | Core inflation above target | − | 0 | − |
| `XCPIH` | Headline inflation above target | − | 0 | − |
| `XGDP` | Growth above its 5-year median | − | + | + |
| `XPCG` | Private credit growth above a neutral pace | − | 0 | 0 |
| `DPCGDP` | Change in private credit / GDP | − | 0 | 0 |
| `RYLD5` | Real 5-year swap yield | + | 0 | − |
| `RYLD2` | Real 2-year swap yield | + | + | − |
| `RSLOPE` | Real 5y − real 2y yield | + | 0 | 0 |
| `FXCRR` | FX carry | 0 | + | 0 |

**Learners.** Re-selected quarterly by `macrosynergy.learning.SignalOptimizer` with expanding panel
cross-validation, no look-ahead and no intercept. Context variables for the interaction and boosting models:
an emerging-market dummy and a high-inflation dummy (headline inflation more than 2 points above target).

**Evaluation.** January 2007 – December 2023; monthly signals; naive PnL at 10% vol, monthly rebalancing, one-day
slippage, gross of costs.

**Decision rule (written before the run).** 12 learner-vs-rule comparisons plus 4 multi-asset comparisons = 16
tests. A learner beats the rule only if its paired HAC t-statistic on daily PnL differences is ≥ 2.96
(Bonferroni, 5%) **and** its gap is positive in both developed and emerging sub-panels.

### Results

| | Rule | Sign-constrained | Ridge | Linear + int. | Boosting |
|---|---|---|---|---|---|
| **Duration** IC | **0.126** | 0.102 | 0.083 | 0.067 | 0.033 |
| Sharpe / beta-neutral | **1.18 / 1.21** | 1.17 / 1.13 | 0.94 / 0.85 | 0.77 / 0.69 | 0.47 / 0.32 |
| t vs rule | | −0.03 | −0.68 | −1.19 | −1.86 |
| **FX** IC | −0.003 | 0.008 | 0.001 | 0.013 | 0.012 |
| Sharpe / beta-neutral | −0.17 / −0.17 | **0.14 / 0.16** | 0.05 / 0.08 | 0.00 / 0.01 | 0.09 / 0.13 |
| t vs rule | | 1.51 | 0.89 | 0.77 | 0.99 |
| **Equity** IC | **0.085** | 0.061 | 0.047 | 0.046 | 0.038 |
| Sharpe / beta-neutral | 0.42 / **0.53** | 0.47 / 0.47 | **0.67** / 0.37 | 0.62 / 0.33 | 0.58 / 0.26 |
| Correlation with long-only | −0.18 | −0.01 | 0.71 | 0.64 | 0.71 |
| t vs rule | | 0.33 | 0.73 | 0.54 | 0.45 |
| **Multi-asset** Sharpe | 0.82 | **0.99** | 0.95 | 0.79 | 0.66 |
| t vs rule | | 0.76 | 0.46 | −0.06 | −0.46 |

*IC = pooled monthly Pearson correlation with next-month returns. Beta-neutral Sharpe removes the PnL's regression
on the long-only return. All duration and equity ICs are significant at 5%; no FX IC is. The multi-asset
portfolio is an equal-risk average of the three asset classes.*

![Sharpe ratios](figures/partB_sharpe.png)

**Duration (theory clear): the rule wins.** It has the highest IC and Sharpe. The sign-constrained learner ties it;
every learner free to choose signs or discover interactions is worse, boosting much worse. This is Part A's
"rule exactly right" case.

**FX (theory unclear): nothing predicts.** The rule's IC is −0.003 and the learners' 0.001–0.013, none
significant. The learners lead by about 3% a year only because the rule loses money; their gaps are positive in
developed markets but mixed in emerging ones.

**Equity: the learners' extra Sharpe is beta.** Their PnLs correlate 0.64–0.71 with simply holding equities: they
learned to stay long through a strong decade. Their ICs are all below the rule's, and after removing beta the rule
is best.

**Multi-asset.** The sign-constrained learner's higher Sharpe (0.99 vs 0.82) is not significant (t = 0.76) and
comes mainly from the FX leg, where the rule fails.

**Against expectations written before the run.** Duration: rule at least as good, confirmed. FX: learners came
closest, as expected, but because the rule failed rather than because they found a signal. Equity: the rule was
more predictive than expected. No learner beats a rule: confirmed.

**Disclosed changes after the first run.** USD was missing from the learners that use FX carry (USD has no carry
series), so its carry was set to zero and everything re-run. A beta-neutral Sharpe was added as a diagnostic. The
decision rule was not changed.

## Conclusions

1. **Whether ML helps depends on how wrong the rule is.** With known truth, learning wins only against a
   mis-specified rule, and only if the learner can represent the error.
2. **Data volume is the binding constraint.** Even favourable cases need about 24 markets × 25 years for a +0.02 IC
   edge. Seventeen real years cannot confirm an edge that size, so Part B's null is what Part A predicts.
3. **Structure beats flexibility.** The learners that won were the most constrained: theory signs with learned
   weights, or a specified interaction. Gradient boosting was weakest in both parts.
4. **Judge learners on IC, sub-panels and beta-neutral returns, not headline Sharpe.** Both apparent real-data
   "wins" disappear on those measures.

**Practical rule.** Start with a theory-signed composite. Use a learner to re-weight inputs under sign constraints,
or to test a specific, economically motivated interaction; promote it only if it passes a pre-registered test.

**Limitations.** Gross of transaction costs; one feature set per asset class; simulated features are Gaussian
AR(1) series without structural breaks; six simulated panels per cell; the Bonferroni threshold makes the
real-data test conservative.

## Files

| File | Contents |
|---|---|
| `ml_vs_rules.ipynb` | Full analysis, executed with outputs |
| `sim_results.csv` | Out-of-sample ICs for all 432 simulated panels |
| `figures/` | Figures used above |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ml_vs_rules.ipynb
```

**Data.** The notebook reads JPMaQS CSV files from `./data` if present; otherwise it downloads the public Kaggle
dataset [macrosynergy/fixed-income-returns-and-macro-trends](https://www.kaggle.com/datasets/macrosynergy/fixed-income-returns-and-macro-trends)
with `kagglehub` (a free Kaggle account and API token are needed). The data are not redistributed here.

**Runtime.** About 10 minutes on 2 cores (Part A about 7, Part B about 3). Set `N_SEEDS = 2` in the first cell for a
quick run; results then differ slightly from those reported.

## Licence

Code and write-up: [MIT](LICENSE). The JPMaQS data are not included and remain subject to the
[Kaggle dataset](https://www.kaggle.com/datasets/macrosynergy/fixed-income-returns-and-macro-trends) terms.
