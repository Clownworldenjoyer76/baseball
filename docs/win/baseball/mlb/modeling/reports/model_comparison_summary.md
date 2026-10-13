# MLB Run Model Comparison

- Generated: `2026-10-13T13:14:06.423701+00:00`
- Untouched chronological test period: `2026-09-11` through `2026-10-12`
- Test games: `79`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= DRatings for BOTH run sides, plus log loss <= DRatings for moneyline, run line, and totals on comparable held-out games.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 1.880920 | 1.986346 | NO |
| away | 2.152066 | 2.119027 | YES |

- Production artifacts changed: **NO**.

## Market probability promotion gate

| Market | Rows | DRatings log loss | Candidate log loss | Pass |
| --- | --- | --- | --- | --- |
| moneyline | 79 | 0.631957 | 0.675230 | NO |
| run_line | 78 | 0.694383 | 0.743502 | NO |
| total | 76 | 0.703959 | 0.707237 | NO |

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 79 | 2.266962 | 1.880920 | 4.171013 | 4.253165 |
| new_model | home | 79 | 2.345446 | 1.986346 | 4.542421 | 4.253165 |
| dratings | away | 79 | 2.293038 | 2.152066 | 3.998861 | 4.708861 |
| new_model | away | 79 | 2.335778 | 2.119027 | 4.281506 | 4.708861 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.266962` -> `2.345446`; Poisson deviance `1.880920` -> `1.986346`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.293038` -> `2.335778`; Poisson deviance `2.152066` -> `2.119027`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.127011 | NO | 0.928571 | NO | 8 |
| run_line | 0.161294 | NO | 0.404762 | NO | 8 |
| total | 0.110346 | NO | 0.833333 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **NO**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 79 | 0.631957 |
| new_model | moneyline | home | 79 | 0.675230 |
| dratings | run_line | home | 78 | 0.694383 |
| new_model | run_line | home | 78 | 0.743502 |
| dratings | total | over_resolved | 76 | 0.703959 |
| new_model | total | over_resolved | 76 | 0.707237 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `472`; positive-EV candidates: `197`.
- New-model all-candidate mean predicted EV vs realized return: `-0.048076` vs `-0.054047`.
- New-model positive-EV mean predicted EV vs realized return: `0.177508` vs `-0.130761`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.060341`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.054847` vs `-0.054047`; EV/return Spearman `-0.034961`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.993863`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `78`.
- Higher-EV side was `-1.5` in `23` games (`29.49%` of non-ties).
- Higher-EV side was `+1.5` in `55` games (`70.51%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
