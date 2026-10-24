$ErrorActionPreference = "Stop"

$Repo = "C:\Users\Mat\Documents\GitHub\baseball_for_mat"
$Target = Join-Path $Repo "docs\win\baseball\mlb\scripts\modeling\evaluate_run_model.py"

if (-not (Test-Path $Target)) {
    throw "Target not found: $Target"
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Backup = "$Target.reconcile_$Timestamp.bak"
Copy-Item $Target $Backup -Force

try {
    $Text = [IO.File]::ReadAllText($Target)

    if ($Text.Contains("from scipy.stats import spearmanr")) {
        $Text = $Text.Replace(
            "from scipy.stats import spearmanr",
            "from scipy.stats import poisson, skellam, spearmanr"
        )
    }
    elseif (-not $Text.Contains("from scipy.stats import poisson, skellam, spearmanr")) {
        throw "Expected scipy.stats import was not found."
    }

    if (-not $Text.Contains("COMMON_TOTAL_LINES = np.array(")) {
        $ConstantPattern = 'PROB_TOLERANCE = 1e-10\r?\n'
        $ConstantReplacement = @'
PROB_TOLERANCE = 1e-10
COMMON_TOTAL_LINES = np.array(
    [6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
    dtype=float,
)
'@
        $Matches = [regex]::Matches($Text, $ConstantPattern)
        if ($Matches.Count -ne 1) {
            throw "Expected exactly one PROB_TOLERANCE definition; found $($Matches.Count)."
        }
        $Text = [regex]::Replace(
            $Text,
            $ConstantPattern,
            $ConstantReplacement + "`n",
            1
        )
    }

    $CalibrationReplacement = @'
def _search_aligned_probability_sets(
    market: pd.DataFrame,
    system: str,
) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Return the fixed probability markets used by exhaustive model search."""
    systems = {
        "dratings": (
            "dratings_home_projected_runs",
            "dratings_away_projected_runs",
        ),
        "new_model": (
            "model_home_runs",
            "model_away_runs",
        ),
    }

    if system not in systems:
        fail(f"Unsupported probability system: {system}")

    home_col, away_col = systems[system]

    required = [
        home_col,
        away_col,
        "target_home_runs",
        "target_away_runs",
    ]
    require_columns(
        market,
        required,
        f"{system} search-aligned probability frame",
    )

    home_mean = pd.to_numeric(
        market[home_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    away_mean = pd.to_numeric(
        market[away_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    actual_home = pd.to_numeric(
        market["target_home_runs"],
        errors="coerce",
    ).to_numpy(dtype=float)
    actual_away = pd.to_numeric(
        market["target_away_runs"],
        errors="coerce",
    ).to_numpy(dtype=float)

    arrays = {
        "home_mean": home_mean,
        "away_mean": away_mean,
        "actual_home": actual_home,
        "actual_away": actual_away,
    }
    invalid = {
        name: int((~np.isfinite(values)).sum())
        for name, values in arrays.items()
        if (~np.isfinite(values)).any()
    }
    if invalid:
        fail(
            f"{system} search-aligned probability inputs contain "
            f"non-finite values: {invalid}"
        )

    if np.any(home_mean <= 0) or np.any(away_mean <= 0):
        fail(
            f"{system} search-aligned run means must be positive"
        )

    diff = actual_home - actual_away
    actual_total = actual_home + actual_away

    run_line_sets = [
        (
            "home_-1.5",
            1.0 - skellam.cdf(1, home_mean, away_mean),
            (diff >= 2).astype(float),
        ),
        (
            "home_+1.5",
            1.0 - skellam.cdf(-2, home_mean, away_mean),
            (diff >= -1).astype(float),
        ),
    ]

    total_mean = home_mean + away_mean
    total_sets: list[tuple[str, np.ndarray, np.ndarray]] = []

    for line in COMMON_TOTAL_LINES:
        threshold = int(math.floor(float(line)))
        probability = (
            1.0
            - poisson.cdf(
                threshold,
                total_mean,
            )
        )
        observed = (
            actual_total > float(line)
        ).astype(float)
        total_sets.append(
            (
                f"over_{line:.1f}",
                np.asarray(probability, dtype=float),
                observed,
            )
        )

    for market_name, sets in [
        ("run_line", run_line_sets),
        ("total", total_sets),
    ]:
        for side, probability, observed in sets:
            if (
                np.any(~np.isfinite(probability))
                or np.any(probability < -PROB_TOLERANCE)
                or np.any(probability > 1.0 + PROB_TOLERANCE)
            ):
                fail(
                    f"{system} {market_name} {side} produced invalid "
                    "search-aligned probabilities"
                )
            if np.any(~np.isfinite(observed)):
                fail(
                    f"{system} {market_name} {side} produced invalid "
                    "search-aligned outcomes"
                )

    return {
        "run_line": run_line_sets,
        "total": total_sets,
    }


def build_calibration_reports(
    market: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    moneyline_records: list[dict] = []
    run_line_records: list[dict] = []
    total_records: list[dict] = []

    for system in ["dratings", "new_model"]:
        for side in ["home", "away"]:
            moneyline_records.extend(
                {
                    "system": system,
                    "side": side,
                    "predicted_probability": probability,
                    "observed_win": observed,
                }
                for probability, observed in zip(
                    market[f"{system}_{side}_ml_prob"],
                    market[f"observed_{side}_ml_win"],
                )
            )

        aligned = _search_aligned_probability_sets(
            market,
            system,
        )

        for side, probability, observed in aligned["run_line"]:
            run_line_records.extend(
                {
                    "system": system,
                    "side": side,
                    "predicted_probability": float(p),
                    "observed_win": float(y),
                }
                for p, y in zip(probability, observed)
            )

        for side, probability, observed in aligned["total"]:
            total_records.extend(
                {
                    "system": system,
                    "side": side,
                    "predicted_probability": float(p),
                    "observed_win": float(y),
                }
                for p, y in zip(probability, observed)
            )

    return {
        "moneyline": calibration_table(
            pd.DataFrame(moneyline_records)
        ),
        "run_line": calibration_table(
            pd.DataFrame(run_line_records)
        ),
        "total": calibration_table(
            pd.DataFrame(total_records)
        ),
    }


'@

    $CalibrationPattern = '(?ms)^def build_calibration_reports\(\r?\n.*?(?=^def binary_log_loss\()'
    $CalibrationMatches = [regex]::Matches($Text, $CalibrationPattern)
    if ($CalibrationMatches.Count -ne 1) {
        throw "Expected exactly one build_calibration_reports block; found $($CalibrationMatches.Count)."
    }
    $Text = [regex]::Replace(
        $Text,
        $CalibrationPattern,
        $CalibrationReplacement,
        1
    )

    $LogLossReplacement = @'
def build_probability_log_loss(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Score the same fixed markets used by exhaustive_run_model_search.py."""
    rows: list[dict] = []

    for system in ["dratings", "new_model"]:
        moneyline_observed = pd.to_numeric(
            market["observed_home_ml_win"],
            errors="coerce",
        )
        moneyline_probability = pd.to_numeric(
            market[f"{system}_home_ml_prob"],
            errors="coerce",
        )
        moneyline_valid = (
            moneyline_observed.notna()
            & moneyline_probability.notna()
        )

        rows.append(
            {
                "system": system,
                "market": "moneyline",
                "evaluation_side": "home",
                "rows": int(moneyline_valid.sum()),
                "log_loss": binary_log_loss(
                    moneyline_observed[moneyline_valid],
                    moneyline_probability[moneyline_valid],
                ),
            }
        )

        aligned = _search_aligned_probability_sets(
            market,
            system,
        )

        run_line_observed = np.concatenate(
            [
                observed
                for _, _, observed in aligned["run_line"]
            ]
        )
        run_line_probability = np.concatenate(
            [
                probability
                for _, probability, _ in aligned["run_line"]
            ]
        )
        rows.append(
            {
                "system": system,
                "market": "run_line",
                "evaluation_side": "home_fixed_-1.5_and_+1.5",
                "rows": int(run_line_observed.size),
                "log_loss": binary_log_loss(
                    run_line_observed,
                    run_line_probability,
                ),
            }
        )

        total_observed = np.concatenate(
            [
                observed
                for _, _, observed in aligned["total"]
            ]
        )
        total_probability = np.concatenate(
            [
                probability
                for _, probability, _ in aligned["total"]
            ]
        )
        rows.append(
            {
                "system": system,
                "market": "total",
                "evaluation_side": "over_common_half_lines",
                "rows": int(total_observed.size),
                "log_loss": binary_log_loss(
                    total_observed,
                    total_probability,
                ),
            }
        )

    return pd.DataFrame(rows)


'@

    $LogLossPattern = '(?ms)^def build_probability_log_loss\(\r?\n.*?(?=^def _scalar\()'
    $LogLossMatches = [regex]::Matches($Text, $LogLossPattern)
    if ($LogLossMatches.Count -ne 1) {
        throw "Expected exactly one build_probability_log_loss block; found $($LogLossMatches.Count)."
    }
    $Text = [regex]::Replace(
        $Text,
        $LogLossPattern,
        $LogLossReplacement,
        1
    )

    [IO.File]::WriteAllText(
        $Target,
        $Text,
        [Text.UTF8Encoding]::new($false)
    )

    $RepoVenv = Join-Path $Repo ".venv314\Scripts\python.exe"
    if (Test-Path $RepoVenv) {
        & $RepoVenv -m py_compile $Target
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.14 -m py_compile $Target
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m py_compile $Target
    }
    else {
        throw "No Python interpreter found for syntax validation."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python syntax validation failed."
    }

    Write-Host "Updated: $Target"
    Write-Host "Backup:  $Backup"
    Write-Host "Syntax:  PASS"
    Write-Host "GitHub:  unchanged"
    Write-Host "Run-line gate now scores fixed home -1.5 and +1.5 together."
    Write-Host "Totals gate now scores 6.5, 7.5, 8.5, 9.5, 10.5, 11.5 over probabilities."
}
catch {
    Copy-Item $Backup $Target -Force
    Write-Host "Patch failed; original file restored from $Backup"
    throw
}
