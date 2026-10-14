# MLB Run Model Comparison

- Generated: `2026-10-14T15:51:04.221181+00:00`
- Untouched chronological test period: `2026-09-12` through `2026-10-13`
- Test games: `77`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= DRatings for BOTH run sides, plus log loss <= DRatings for moneyline, run line, and totals on comparable held-out games.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 1.891869 | 1.945363 | NO |
| away | 2.168730 | 2.159762 | YES |

- Production artifacts changed: **NO**.

## Market probability promotion gate

| Market | Rows | DRatings log loss | Candidate log loss | Pass |
| --- | --- | --- | --- | --- |
| moneyline | 77 | 0.631212 | 0.669725 | NO |
| run_line | 76 | 0.697357 | 0.735460 | NO |
| total | 74 | 0.705398 | 0.719843 | NO |

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 77 | 2.248182 | 1.891869 | 4.144805 | 4.142857 |
| new_model | home | 77 | 2.313305 | 1.945363 | 4.503862 | 4.142857 |
| dratings | away | 77 | 2.297013 | 2.168730 | 3.955455 | 4.675325 |
| new_model | away | 77 | 2.321288 | 2.159762 | 4.191470 | 4.675325 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.248182` -> `2.313305`; Poisson deviance `1.891869` -> `1.945363`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.297013` -> `2.321288`; Poisson deviance `2.168730` -> `2.159762`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.083802 | NO | 0.857143 | NO | 8 |
| run_line | 0.162892 | NO | 0.778457 | NO | 8 |
| total | 0.121230 | NO | 0.761905 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **NO**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 77 | 0.631212 |
| new_model | moneyline | home | 77 | 0.669725 |
| dratings | run_line | home | 76 | 0.697357 |
| new_model | run_line | home | 76 | 0.735460 |
| dratings | total | over_resolved | 74 | 0.705398 |
| new_model | total | over_resolved | 74 | 0.719843 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `460`; positive-EV candidates: `189`.
- New-model all-candidate mean predicted EV vs realized return: `-0.046523` vs `-0.055370`.
- New-model positive-EV mean predicted EV vs realized return: `0.182870` vs `-0.180000`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.072152`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.055119` vs `-0.055370`; EV/return Spearman `-0.055194`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.994789`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `76`.
- Higher-EV side was `-1.5` in `23` games (`30.26%` of non-ties).
- Higher-EV side was `+1.5` in `53` games (`69.74%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
