$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\ntmal\Downloads\mat\baseball_for_mat"
$Target = Join-Path $RepoRoot "docs\win\baseball\mlb\scripts\modeling\evaluate_run_model.py"

if (-not (Test-Path -LiteralPath $Target)) {
    throw "Target file not found: $Target"
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Backup = "$Target.item3_$Timestamp.bak"
Copy-Item -LiteralPath $Target -Destination $Backup -Force

$text = [System.IO.File]::ReadAllText($Target)

function Replace-Exact {
    param(
        [Parameter(Mandatory = $true)][string]$InputText,
        [Parameter(Mandatory = $true)][string]$OldText,
        [Parameter(Mandatory = $true)][string]$NewText,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $normalizedInput = $InputText.Replace("`r`n", "`n")
    $normalizedOld = $OldText.Replace("`r`n", "`n")
    $normalizedNew = $NewText.Replace("`r`n", "`n")

    $count = ([regex]::Matches($normalizedInput, [regex]::Escape($normalizedOld))).Count
    if ($count -ne 1) {
        throw "$Label replacement expected exactly 1 match; found $count"
    }

    return $normalizedInput.Replace($normalizedOld, $normalizedNew)
}

try {
    $oldPromotionDoc = @'
The candidate pair is promoted only when:
1. the mean of home/away candidate Poisson deviances is <= the corresponding
   DRatings pair mean on the untouched chronological test set; and
2. candidate log loss is <= DRatings log loss for moneyline, run line, and
   total probabilities on that same test period.
'@

    $newPromotionDoc = @'
The candidate pair is promoted only when:
1. the mean of home/away candidate Poisson deviances is <= the corresponding
   DRatings pair mean on the untouched chronological test set; and
2. for moneyline, run line, and total probabilities, candidate log loss is
   <= both DRatings log loss and the configured absolute market threshold; and
3. for those same markets, candidate expected calibration error (ECE) is
   <= both DRatings ECE and the configured absolute market threshold.
'@

    $text = Replace-Exact $text $oldPromotionDoc $newPromotionDoc "promotion docstring"

    $oldThreshold = @'
CALIBRATION_ECE_THRESHOLD = 0.05
'@

    $newThreshold = @'
MARKET_PROMOTION_THRESHOLDS = {
    "moneyline": {
        "log_loss": 0.67,
        "ece": 0.05,
    },
    "run_line": {
        "log_loss": 0.66,
        "ece": 0.05,
    },
    "total": {
        "log_loss": 0.67,
        "ece": 0.05,
    },
}
'@

    $text = Replace-Exact $text $oldThreshold $newThreshold "market threshold constants"

    $functionPattern = '(?s)def build_promotion_decision\(.*?\n\ndef _write_json\('
    $functionMatches = [regex]::Matches($text, $functionPattern)
    if ($functionMatches.Count -ne 1) {
        throw "build_promotion_decision replacement expected exactly 1 match; found $($functionMatches.Count)"
    }

    $newFunction = @'
def build_promotion_decision(
    run_metrics: pd.DataFrame,
    log_loss: pd.DataFrame,
    calibrations: dict[str, pd.DataFrame],
    *,
    test_start: str,
    test_end: str,
) -> dict:
    run_lookup = {
        (str(row.system), str(row.side)): row
        for row in run_metrics.itertuples(index=False)
    }

    required_run = [
        ("dratings", "home"),
        ("new_model", "home"),
        ("dratings", "away"),
        ("new_model", "away"),
    ]
    missing_run = [key for key in required_run if key not in run_lookup]
    if missing_run:
        fail(f"Promotion gate missing required run metrics: {missing_run}")

    comparison: dict[str, object] = {}
    baseline_poisson_values: list[float] = []
    candidate_poisson_values: list[float] = []

    for side in ["home", "away"]:
        baseline = float(
            run_lookup[("dratings", side)].mean_poisson_deviance
        )
        candidate = float(
            run_lookup[("new_model", side)].mean_poisson_deviance
        )

        if not np.isfinite(baseline) or not np.isfinite(candidate):
            fail(
                f"Promotion gate received non-finite {side} Poisson deviance: "
                f"baseline={baseline} candidate={candidate}"
            )

        side_improved = bool(candidate <= baseline)
        baseline_poisson_values.append(baseline)
        candidate_poisson_values.append(candidate)

        comparison[side] = {
            "baseline_system": "dratings",
            "metric": "mean_poisson_deviance",
            "baseline_value": baseline,
            "candidate_value": candidate,
            "candidate_lte_baseline": side_improved,
            "difference_candidate_minus_baseline": candidate - baseline,
            "gating_component": False,
        }

    baseline_pair_poisson = float(np.mean(baseline_poisson_values))
    candidate_pair_poisson = float(np.mean(candidate_poisson_values))
    run_pair_passed = bool(
        candidate_pair_poisson <= baseline_pair_poisson
    )

    comparison["run_pair"] = {
        "baseline_system": "dratings",
        "metric": "mean_home_away_poisson_deviance",
        "baseline_value": baseline_pair_poisson,
        "candidate_value": candidate_pair_poisson,
        "candidate_lte_baseline": run_pair_passed,
        "difference_candidate_minus_baseline": (
            candidate_pair_poisson - baseline_pair_poisson
        ),
        "gating_component": True,
    }

    log_lookup = {
        (str(row.system), str(row.market)): row
        for row in log_loss.itertuples(index=False)
    }

    market_comparison: dict[str, dict] = {}
    market_passed = True

    for market_name in ["moneyline", "run_line", "total"]:
        baseline_key = ("dratings", market_name)
        candidate_key = ("new_model", market_name)

        if baseline_key not in log_lookup or candidate_key not in log_lookup:
            fail(
                "Promotion gate missing required probability log loss: "
                f"market={market_name}"
            )

        if market_name not in calibrations:
            fail(
                "Promotion gate missing required probability calibration: "
                f"market={market_name}"
            )

        thresholds = MARKET_PROMOTION_THRESHOLDS.get(market_name)
        if not isinstance(thresholds, dict):
            fail(
                "Promotion gate missing configured market thresholds: "
                f"market={market_name}"
            )

        log_loss_threshold = float(thresholds["log_loss"])
        ece_threshold = float(thresholds["ece"])

        baseline_log_loss = float(log_lookup[baseline_key].log_loss)
        candidate_log_loss = float(log_lookup[candidate_key].log_loss)
        baseline_ece = expected_calibration_error(
            calibrations[market_name],
            "dratings",
        )
        candidate_ece = expected_calibration_error(
            calibrations[market_name],
            "new_model",
        )

        values_to_validate = {
            "baseline_log_loss": baseline_log_loss,
            "candidate_log_loss": candidate_log_loss,
            "log_loss_threshold": log_loss_threshold,
            "baseline_ece": baseline_ece,
            "candidate_ece": candidate_ece,
            "ece_threshold": ece_threshold,
        }
        invalid = {
            key: value
            for key, value in values_to_validate.items()
            if not np.isfinite(value)
        }
        if invalid:
            fail(
                "Promotion gate received non-finite probability metric: "
                f"market={market_name} values={invalid}"
            )

        log_loss_lte_baseline = bool(
            candidate_log_loss <= baseline_log_loss
        )
        log_loss_lte_threshold = bool(
            candidate_log_loss <= log_loss_threshold
        )
        ece_lte_baseline = bool(
            candidate_ece <= baseline_ece
        )
        ece_lte_threshold = bool(
            candidate_ece <= ece_threshold
        )
        passed = bool(
            log_loss_lte_baseline
            and log_loss_lte_threshold
            and ece_lte_baseline
            and ece_lte_threshold
        )
        market_passed = market_passed and passed

        market_comparison[market_name] = {
            "baseline_system": "dratings",
            "log_loss": {
                "baseline_value": baseline_log_loss,
                "candidate_value": candidate_log_loss,
                "absolute_threshold": log_loss_threshold,
                "candidate_lte_baseline": log_loss_lte_baseline,
                "candidate_lte_threshold": log_loss_lte_threshold,
                "difference_candidate_minus_baseline": (
                    candidate_log_loss - baseline_log_loss
                ),
            },
            "calibration_ece": {
                "baseline_value": baseline_ece,
                "candidate_value": candidate_ece,
                "absolute_threshold": ece_threshold,
                "candidate_lte_baseline": ece_lte_baseline,
                "candidate_lte_threshold": ece_lte_threshold,
                "difference_candidate_minus_baseline": (
                    candidate_ece - baseline_ece
                ),
            },
            "passed": passed,
            "gating_component": True,
        }

    comparison["markets"] = market_comparison

    all_passed = bool(run_pair_passed and market_passed)

    failed_components: list[str] = []
    if not run_pair_passed:
        failed_components.append("run_pair_poisson")

    for market_name, values in market_comparison.items():
        log_gate = values["log_loss"]
        ece_gate = values["calibration_ece"]

        if not log_gate["candidate_lte_baseline"]:
            failed_components.append(
                f"{market_name}_log_loss_baseline"
            )
        if not log_gate["candidate_lte_threshold"]:
            failed_components.append(
                f"{market_name}_log_loss_threshold"
            )
        if not ece_gate["candidate_lte_baseline"]:
            failed_components.append(
                f"{market_name}_ece_baseline"
            )
        if not ece_gate["candidate_lte_threshold"]:
            failed_components.append(
                f"{market_name}_ece_threshold"
            )

    return {
        "status": (
            "candidate_promoted"
            if all_passed
            else "candidate_rejected"
        ),
        "gate_passed": all_passed,
        "gate_rule": (
            "coupled candidate mean(home,away) Poisson deviance <= DRatings "
            "coupled mean AND, for moneyline, run_line, and total, candidate "
            "log loss <= DRatings and configured absolute threshold AND "
            "candidate ECE <= DRatings and configured absolute threshold"
        ),
        "test_start_date": str(test_start),
        "test_end_date": str(test_end),
        "comparison": comparison,
        "failed_components": failed_components,
    }


def _write_json(
'@

    $text = [regex]::Replace($text, $functionPattern, $newFunction, 1)

    $oldCalibrationCheck = @'
            "calibrated": (
                np.isfinite(ece)
                and ece <= CALIBRATION_ECE_THRESHOLD
            ),
'@

    $newCalibrationCheck = @'
            "calibrated": (
                np.isfinite(ece)
                and ece
                <= MARKET_PROMOTION_THRESHOLDS[market_name]["ece"]
            ),
'@

    $text = Replace-Exact $text $oldCalibrationCheck $newCalibrationCheck "summary calibration threshold"

    $oldGateRows = @'
    gate_market_rows = []
    for market_name in ["moneyline", "run_line", "total"]:
        values = market_gate[market_name]
        gate_market_rows.append(
            [
                market_name,
                _fmt_float(values["baseline_value"]),
                _fmt_float(values["candidate_value"]),
                _yes_no(values["candidate_lte_baseline"]),
            ]
        )
'@

    $newGateRows = @'
    gate_market_rows = []
    for market_name in ["moneyline", "run_line", "total"]:
        values = market_gate[market_name]
        log_gate = values["log_loss"]
        ece_gate = values["calibration_ece"]
        gate_market_rows.append(
            [
                market_name,
                _fmt_float(log_gate["baseline_value"]),
                _fmt_float(log_gate["candidate_value"]),
                _fmt_float(log_gate["absolute_threshold"]),
                _yes_no(log_gate["candidate_lte_baseline"]),
                _yes_no(log_gate["candidate_lte_threshold"]),
                _fmt_float(ece_gate["baseline_value"]),
                _fmt_float(ece_gate["candidate_value"]),
                _fmt_float(ece_gate["absolute_threshold"]),
                _yes_no(ece_gate["candidate_lte_baseline"]),
                _yes_no(ece_gate["candidate_lte_threshold"]),
                _yes_no(values["passed"]),
            ]
        )
'@

    $text = Replace-Exact $text $oldGateRows $newGateRows "promotion summary market rows"

    $oldGateDescription = @'
            "Candidate promotion requires the coupled mean home/away Poisson "
            "deviance to meet or beat DRatings AND moneyline, run-line, and "
            "total probability log loss to each meet or beat DRatings."
'@

    $newGateDescription = @'
            "Candidate promotion requires the coupled mean home/away Poisson "
            "deviance to meet or beat DRatings AND each probability market "
            "to pass both relative and absolute log-loss/ECE thresholds."
'@

    $text = Replace-Exact $text $oldGateDescription $newGateDescription "promotion summary description"

    $oldGateTable = @'
        markdown_table(
            [
                "Probability market gate",
                "DRatings log loss",
                "Candidate log loss",
                "Candidate <= baseline",
            ],
            gate_market_rows,
        ),
'@

    $newGateTable = @'
        markdown_table(
            [
                "Probability market gate",
                "DRatings log loss",
                "Candidate log loss",
                "Log-loss ceiling",
                "LL <= baseline",
                "LL <= ceiling",
                "DRatings ECE",
                "Candidate ECE",
                "ECE ceiling",
                "ECE <= baseline",
                "ECE <= ceiling",
                "Market PASS",
            ],
            gate_market_rows,
        ),
'@

    $text = Replace-Exact $text $oldGateTable $newGateTable "promotion summary table"

    $oldCalibrationDescription = @'
            "Calibration YES/NO uses weighted expected calibration error "
            f"(ECE) <= `{CALIBRATION_ECE_THRESHOLD:.2f}`. "
            "Totals use conditional win probability on resolved bets; pushes "
            "are excluded from the observed win-rate denominator."
'@

    $newCalibrationDescription = @'
            "Calibration YES/NO uses the configured per-market weighted "
            "expected calibration error (ECE) ceiling. All current market "
            "ECE ceilings are `0.05`. Totals use conditional win probability "
            "on resolved bets; pushes are excluded from the observed win-rate "
            "denominator."
'@

    $text = Replace-Exact $text $oldCalibrationDescription $newCalibrationDescription "calibration summary description"

    $oldCall = @'
        promotion_decision = build_promotion_decision(
            run_metrics,
            log_loss,
            test_start=test_start,
            test_end=test_end,
        )
'@

    $newCall = @'
        promotion_decision = build_promotion_decision(
            run_metrics,
            log_loss,
            calibrations,
            test_start=test_start,
            test_end=test_end,
        )
'@

    $text = Replace-Exact $text $oldCall $newCall "promotion decision call"

    $oldPromotionNote = @'
        "Coupled run-model pair passed the production gate: pair-level mean "
        "Poisson deviance and moneyline/run-line/total probability log loss "
        "all met or beat DRatings. The two models were promoted together."
'@

    $newPromotionNote = @'
        "Coupled run-model pair passed the production gate: pair-level mean "
        "Poisson deviance met or beat DRatings, and moneyline/run-line/total "
        "log loss and ECE each met both the DRatings baseline and configured "
        "absolute threshold. The two models were promoted together."
'@

    $text = Replace-Exact $text $oldPromotionNote $newPromotionNote "promotion success note"

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Target, $text, $utf8NoBom)

    & python -m py_compile $Target
    if ($LASTEXITCODE -ne 0) {
        throw "Python syntax validation failed with exit code $LASTEXITCODE"
    }

    Write-Host "Updated: $Target"
    Write-Host "Backup:  $Backup"
    Write-Host "Syntax:  PASS"
    Write-Host "GitHub:  unchanged"
}
catch {
    Copy-Item -LiteralPath $Backup -Destination $Target -Force
    Write-Error "Update failed. Original file restored from $Backup. $($_.Exception.Message)"
    exit 1
}
