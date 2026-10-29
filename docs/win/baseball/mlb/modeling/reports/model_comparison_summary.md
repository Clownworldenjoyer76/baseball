# MLB Run Model Comparison

- Generated: `2026-10-28T11:46:43.825093+00:00`
- Untouched chronological test period: `2026-09-19` through `2026-10-27`
- Test games: `78`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= DRatings for BOTH run sides, plus log loss <= DRatings for moneyline, run line, and totals on comparable held-out games.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 1.819117 | 1.955579 | NO |
| away | 1.865782 | 1.867823 | NO |

- Production artifacts changed: **NO**.

## Market probability promotion gate

| Market | Rows | DRatings log loss | Candidate log loss | Pass |
| --- | --- | --- | --- | --- |
| moneyline | 78 | 0.642913 | 0.692480 | NO |
| run_line | 77 | 0.697396 | 0.737055 | NO |
| total | 74 | 0.692172 | 0.732221 | NO |

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 78 | 2.127949 | 1.819117 | 4.088974 | 4.038462 |
| new_model | home | 78 | 2.247941 | 1.955579 | 4.522488 | 4.038462 |
| dratings | away | 78 | 2.055513 | 1.865782 | 3.774744 | 4.269231 |
| new_model | away | 78 | 2.124324 | 1.867823 | 4.144154 | 4.269231 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.127949` -> `2.247941`; Poisson deviance `1.819117` -> `1.955579`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.055513` -> `2.124324`; Poisson deviance `1.865782` -> `1.867823`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.098091 | NO | -0.357143 | NO | 8 |
| run_line | 0.146978 | NO | -0.285714 | NO | 8 |
| total | 0.149527 | NO | 0.785714 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **NO**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 78 | 0.642913 |
| new_model | moneyline | home | 78 | 0.692480 |
| dratings | run_line | home | 77 | 0.697396 |
| new_model | run_line | home | 77 | 0.737055 |
| dratings | total | over_resolved | 74 | 0.692172 |
| new_model | total | over_resolved | 74 | 0.732221 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `466`; positive-EV candidates: `194`.
- New-model all-candidate mean predicted EV vs realized return: `-0.047423` vs `-0.052039`.
- New-model positive-EV mean predicted EV vs realized return: `0.186398` vs `-0.117835`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.055690`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.055472` vs `-0.052039`; EV/return Spearman `-0.032204`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.994446`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `77`.
- Higher-EV side was `-1.5` in `23` games (`29.87%` of non-ties).
- Higher-EV side was `+1.5` in `54` games (`70.13%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
