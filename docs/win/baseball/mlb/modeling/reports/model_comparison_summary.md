# MLB Run Model Comparison

- Generated: `2026-11-03T19:15:33.478516+00:00`
- Untouched chronological test period: `2026-09-20` through `2026-10-28`
- Test games: `131`
- Model fitting/tuning performed by this evaluation script: `NO`
- Promotion status: `candidate_rejected`

## Production promotion gate

Candidate promotion requires mean Poisson deviance <= DRatings for BOTH run sides, plus log loss <= DRatings for moneyline, run line, and totals on comparable held-out games.

| Side | DRatings baseline Poisson | Candidate Poisson | Candidate <= baseline |
| --- | --- | --- | --- |
| home | 2.052391 | 2.256521 | NO |
| away | 2.126983 | 2.054655 | YES |

- Production artifacts changed: **NO**.

## Market probability promotion gate

| Market | Rows | DRatings log loss | Candidate log loss | Pass |
| --- | --- | --- | --- | --- |
| moneyline | 131 | 0.663144 | 0.705061 | NO |
| run_line | 131 | 0.705925 | 0.746822 | NO |
| total | 125 | 0.704151 | 0.750655 | NO |

## Run prediction metrics

| System | Side | Rows | MAE | Mean Poisson deviance | Mean predicted runs | Mean actual runs |
| --- | --- | --- | --- | --- | --- | --- |
| dratings | home | 131 | 2.268015 | 2.052391 | 4.102214 | 3.870229 |
| new_model | home | 131 | 2.435896 | 2.256521 | 4.491126 | 3.870229 |
| dratings | away | 131 | 2.178397 | 2.126983 | 3.786947 | 4.297710 |
| new_model | away | 131 | 2.211234 | 2.054655 | 4.099018 | 4.297710 |

### Run-prediction questions

- Does the new model improve home-run prediction error? **NO** (MAE `2.268015` -> `2.435896`; Poisson deviance `2.052391` -> `2.256521`).
- Does the new model improve away-run prediction error? **NO** (MAE `2.178397` -> `2.211234`; Poisson deviance `2.126983` -> `2.054655`).

## Probability calibration

Calibration YES/NO uses weighted expected calibration error (ECE) <= `0.05`. Totals use conditional win probability on resolved bets; pushes are excluded from the observed win-rate denominator.

| Market | New-model ECE | Calibrated | Predicted-vs-observed Spearman | Observed rate exactly non-decreasing | Populated bins |
| --- | --- | --- | --- | --- | --- |
| moneyline | 0.125105 | NO | 0.500000 | NO | 8 |
| run_line | 0.189233 | NO | 0.275454 | NO | 8 |
| total | 0.133192 | NO | -0.309524 | NO | 8 |

- Are predicted moneyline probabilities calibrated? **NO**.
- Are predicted run-line probabilities calibrated? **NO**.
- Are predicted total probabilities calibrated? **NO**.
- Does increasing predicted probability correspond to increasing observed win rate? Moneyline **NO**, run line **NO**, total **NO**. See Spearman values above for rank-direction strength.

## Probability log loss

| System | Market | Evaluation side | Rows | Log loss |
| --- | --- | --- | --- | --- |
| dratings | moneyline | home | 131 | 0.663144 |
| new_model | moneyline | home | 131 | 0.705061 |
| dratings | run_line | home | 131 | 0.705925 |
| new_model | run_line | home | 131 | 0.746822 |
| dratings | total | over_resolved | 125 | 0.704151 |
| new_model | total | over_resolved | 125 | 0.750655 |

## EV, realized return, and Kelly

- New-model priced candidates evaluated: `786`; positive-EV candidates: `339`.
- New-model all-candidate mean predicted EV vs realized return: `-0.048056` vs `-0.042392`.
- New-model positive-EV mean predicted EV vs realized return: `0.160572` vs `-0.120708`.
- Does higher predicted EV correspond to higher realized return? EV/return Spearman = `-0.083424`. A positive value indicates higher EV tended to correspond to higher realized return in this test sample.
- Is positive EV overstated versus realized return? **YES** (defined here as mean realized return below mean predicted EV among positive-EV candidates).
- DRatings-run baseline all-candidate mean predicted EV vs realized return: `-0.054326` vs `-0.042392`; EV/return Spearman `-0.065093`.
- Does Kelly increase monotonically with actual model edge? Edge/Kelly-raw Spearman = `0.993452`; mean raw Kelly across ordered edge bins is non-decreasing: **YES** across `10` populated edge bins.

## Run-line side preference

- Games with both run-line sides priced/evaluated: `131`.
- Higher-EV side was `-1.5` in `51` games (`38.93%` of non-ties).
- Higher-EV side was `+1.5` in `80` games (`61.07%` of non-ties).
- Exact EV ties: `0`.

## Interpretation constraint

This report evaluates the candidate on the untouched test period only. The script does not refit, retune, or select hyperparameters from these results. Do not tune the model on this final test period after reviewing the report.
