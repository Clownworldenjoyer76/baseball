# MLB Run Model Comparison

- Generated: `2026-10-28T07:16:13.245033+00:00`
- Untouched chronological test period: `2026-09-19` through `2026-10-27`
- Test games: `72`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= DRatings for BOTH run sides, plus log loss <= DRatings for moneyline, run line, and totals on comparable held-out games.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 1.806495 | 1.953280 | NO |
| away | 1.974017 | 1.974530 | NO |

- Production artifacts changed: **NO**.

## Market probability promotion gate

| Market | Rows | DRatings log loss | Candidate log loss | Pass |
| --- | --- | --- | --- | --- |
| moneyline | 72 | 0.642561 | 0.684902 | NO |
| run_line | 71 | 0.705130 | 0.741346 | NO |
| total | 68 | 0.696232 | 0.744932 | NO |

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 72 | 2.146111 | 1.806495 | 4.073333 | 4.013889 |
| new_model | home | 72 | 2.292475 | 1.953280 | 4.497455 | 4.013889 |
| dratings | away | 72 | 2.143889 | 1.974017 | 3.793889 | 4.347222 |
| new_model | away | 72 | 2.173509 | 1.974530 | 4.155668 | 4.347222 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.146111` -> `2.292475`; Poisson deviance `1.806495` -> `1.953280`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.143889` -> `2.173509`; Poisson deviance `1.974017` -> `1.974530`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.058641 | NO | 1.000000 | YES | 8 |
| run_line | 0.241114 | NO | 0.357143 | NO | 8 |
| total | 0.163864 | NO | 0.380510 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **YES**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 72 | 0.642561 |
| new_model | moneyline | home | 72 | 0.684902 |
| dratings | run_line | home | 71 | 0.705130 |
| new_model | run_line | home | 71 | 0.741346 |
| dratings | total | over_resolved | 68 | 0.696232 |
| new_model | total | over_resolved | 68 | 0.744932 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `430`; positive-EV candidates: `182`.
- New-model all-candidate mean predicted EV vs realized return: `-0.046356` vs `-0.049791`.
- New-model positive-EV mean predicted EV vs realized return: `0.186073` vs `-0.169560`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.068153`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.055210` vs `-0.049791`; EV/return Spearman `-0.041801`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.993940`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `71`.
- Higher-EV side was `-1.5` in `25` games (`35.21%` of non-ties).
- Higher-EV side was `+1.5` in `46` games (`64.79%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
