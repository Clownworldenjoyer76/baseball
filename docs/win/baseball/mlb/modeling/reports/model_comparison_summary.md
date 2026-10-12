# MLB Run Model Comparison

- Generated: `2026-10-12T07:44:36.131164+00:00`
- Untouched chronological test period: `2026-09-10` through `2026-10-11`
- Test games: `78`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= the DRatings baseline for BOTH home and away models.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 1.962368 | 2.116413 | NO |
| away | 2.134735 | 2.130713 | YES |

- Production artifacts changed: **NO**.

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 78 | 2.286795 | 1.962368 | 4.182949 | 4.179487 |
| new_model | home | 78 | 2.388178 | 2.116413 | 4.538882 | 4.179487 |
| dratings | away | 78 | 2.282692 | 2.134735 | 4.022179 | 4.653846 |
| new_model | away | 78 | 2.331978 | 2.130713 | 4.250414 | 4.653846 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.286795` -> `2.388178`; Poisson deviance `1.962368` -> `2.116413`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.282692` -> `2.331978`; Poisson deviance `2.134735` -> `2.130713`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.096274 | NO | 0.857143 | NO | 8 |
| run_line | 0.183915 | NO | 0.476190 | NO | 8 |
| total | 0.156495 | NO | -0.083834 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **NO**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 78 | 0.625333 |
| dratings | run_line | home | 77 | 0.708359 |
| dratings | total | over_resolved | 75 | 0.704333 |
| new_model | moneyline | home | 78 | 0.681586 |
| new_model | run_line | home | 77 | 0.758658 |
| new_model | total | over_resolved | 75 | 0.736807 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `466`; positive-EV candidates: `195`.
- New-model all-candidate mean predicted EV vs realized return: `-0.047587` vs `-0.052639`.
- New-model positive-EV mean predicted EV vs realized return: `0.188823` vs `-0.149744`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.087565`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.054971` vs `-0.052639`; EV/return Spearman `-0.043624`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.993656`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `77`.
- Higher-EV side was `-1.5` in `27` games (`35.06%` of non-ties).
- Higher-EV side was `+1.5` in `50` games (`64.94%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
