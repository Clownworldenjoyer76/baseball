# MLB Run Model Comparison

- Generated: `2026-10-27T22:14:37.077429+00:00`
- Untouched chronological test period: `2026-09-18` through `2026-10-26`
- Test games: `75`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= DRatings for BOTH run sides, plus log loss <= DRatings for moneyline, run line, and totals on comparable held-out games.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 1.911160 | 2.030122 | NO |
| away | 1.975142 | 1.965345 | YES |

- Production artifacts changed: **NO**.

## Market probability promotion gate

| Market | Rows | DRatings log loss | Candidate log loss | Pass |
| --- | --- | --- | --- | --- |
| moneyline | 75 | 0.632050 | 0.674922 | NO |
| run_line | 74 | 0.719351 | 0.756249 | NO |
| total | 72 | 0.701099 | 0.731186 | NO |

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 75 | 2.224133 | 1.911160 | 4.088400 | 4.040000 |
| new_model | home | 75 | 2.348852 | 2.030122 | 4.478355 | 4.040000 |
| dratings | away | 75 | 2.162667 | 1.975142 | 3.784800 | 4.346667 |
| new_model | away | 75 | 2.169736 | 1.965345 | 4.009711 | 4.346667 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.224133` -> `2.348852`; Poisson deviance `1.911160` -> `2.030122`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.162667` -> `2.169736`; Poisson deviance `1.975142` -> `1.965345`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.113029 | NO | 0.857143 | NO | 8 |
| run_line | 0.179218 | NO | 0.467074 | NO | 8 |
| total | 0.145283 | NO | -0.238095 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **NO**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 75 | 0.632050 |
| new_model | moneyline | home | 75 | 0.674922 |
| dratings | run_line | home | 74 | 0.719351 |
| new_model | run_line | home | 74 | 0.756249 |
| dratings | total | over_resolved | 72 | 0.701099 |
| new_model | total | over_resolved | 72 | 0.731186 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `448`; positive-EV candidates: `202`.
- New-model all-candidate mean predicted EV vs realized return: `-0.045982` vs `-0.050580`.
- New-model positive-EV mean predicted EV vs realized return: `0.186666` vs `-0.100050`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.040342`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.055397` vs `-0.050580`; EV/return Spearman `-0.069462`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.992859`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `74`.
- Higher-EV side was `-1.5` in `32` games (`43.24%` of non-ties).
- Higher-EV side was `+1.5` in `42` games (`56.76%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
