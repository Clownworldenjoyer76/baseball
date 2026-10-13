#!/usr/bin/env python3
"""Evaluate an MLB run-model candidate and gate production promotion.

Inputs
------
docs/win/baseball/mlb/modeling/data/mlb_run_training_set.csv
docs/win/baseball/mlb/models/run_projection/candidates/home_runs_model.joblib
docs/win/baseball/mlb/models/run_projection/candidates/away_runs_model.joblib
docs/win/baseball/mlb/models/run_projection/candidates/home_runs_model_metadata.json
docs/win/baseball/mlb/models/run_projection/candidates/away_runs_model_metadata.json
docs/win/baseball/mlb/00_intake/sportsbook/{date}_MLB.csv

Outputs
-------
docs/win/baseball/mlb/modeling/reports/run_prediction_metrics.csv
docs/win/baseball/mlb/modeling/reports/moneyline_calibration.csv
docs/win/baseball/mlb/modeling/reports/run_line_calibration.csv
docs/win/baseball/mlb/modeling/reports/total_calibration.csv
docs/win/baseball/mlb/modeling/reports/probability_log_loss.csv
docs/win/baseball/mlb/modeling/reports/model_comparison_summary.md
docs/win/baseball/mlb/modeling/reports/run_model_promotion.json

Promotion rule
--------------
The candidate is promoted only when candidate mean Poisson deviance is less than
or equal to the DRatings baseline mean Poisson deviance for BOTH home and away
run models on the untouched chronological test set. If either side fails, the
existing production model and metadata files remain unchanged.

This script never fits or tunes a model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from module_loader import load_module_from_path


BASE_DIR = Path("docs/win/baseball/mlb")
DEFAULT_TRAINING_DATA = BASE_DIR / "modeling/data/mlb_run_training_set.csv"
DEFAULT_PRODUCTION_MODEL_DIR = BASE_DIR / "models/run_projection"
DEFAULT_CANDIDATE_DIR = DEFAULT_PRODUCTION_MODEL_DIR / "candidates"
DEFAULT_SPORTSBOOK_DIR = BASE_DIR / "00_intake/sportsbook"
DEFAULT_REPORT_DIR = BASE_DIR / "modeling/reports"

ERROR_DIR = BASE_DIR / "errors/modeling"
LOG_FILE = ERROR_DIR / "evaluate_run_model.txt"

HOME_MODEL_NAME = "home_runs_model.joblib"
AWAY_MODEL_NAME = "away_runs_model.joblib"
HOME_METADATA_NAME = "home_runs_model_metadata.json"
AWAY_METADATA_NAME = "away_runs_model_metadata.json"

REPORT_FILES = {
    "run_metrics": "run_prediction_metrics.csv",
    "moneyline_calibration": "moneyline_calibration.csv",
    "run_line_calibration": "run_line_calibration.csv",
    "total_calibration": "total_calibration.csv",
    "log_loss": "probability_log_loss.csv",
    "summary": "model_comparison_summary.md",
    "promotion": "run_model_promotion.json",
}

PROBABILITY_BINS = np.linspace(0.0, 1.0, 11)
PROBABILITY_BIN_LABELS = [
    f"{PROBABILITY_BINS[i]:.1f}-{PROBABILITY_BINS[i + 1]:.1f}"
    for i in range(len(PROBABILITY_BINS) - 1)
]

EPSILON = 1e-12
PROB_TOLERANCE = 1e-10
CALIBRATION_ECE_THRESHOLD = 0.05

SPORTSBOOK_REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "away_run_line",
    "home_run_line",
    "total",
    "away_dk_run_line_decimal",
    "home_dk_run_line_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_moneyline_decimal",
]

TEST_REQUIRED_COLUMNS = [
    "game_date",
    "game_id",
    "gamePk",
    "home_team",
    "away_team",
    "dratings_home_projected_runs",
    "dratings_away_projected_runs",
    "dratings_total_projected_runs",
    "target_home_runs",
    "target_away_runs",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_dirs(report_dir: Path) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)


def _log(message: str, level: str = "INFO") -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {message.rstrip()}\n")


def fail(message: str) -> Never:
    _log(message, "ERROR")
    raise RuntimeError(message)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()

    for parent in here.parents:
        mlb_requirements = (
            parent
            / "docs/win/baseball/mlb/requirements.txt"
        )

        if mlb_requirements.is_file():
            return parent

    cwd = Path.cwd().resolve()
    mlb_requirements = (
        cwd
        / "docs/win/baseball/mlb/requirements.txt"
    )

    if mlb_requirements.is_file():
        return cwd

    raise RuntimeError(
        "Could not resolve repository root containing "
        "docs/win/baseball/mlb/requirements.txt "
        f"from script={here} cwd={cwd}"
    )


def _load_module(name: str, path: Path):
    return load_module_from_path(
        name,
        path,
        fail=fail,
        missing_message=(
            "Required production module not found: {path}"
        ),
        invalid_spec_message=(
            "Could not load production module: {path}"
        ),
        add_parent_to_path=True,
    )


def _load_production_math():
    repo_root = _find_repo_root()

    probability_path = (
        repo_root
        / "docs/win/baseball/mlb/scripts/01_merge/build_juice_files.py"
    )
    ev_kelly_path = (
        repo_root
        / "docs/win/baseball/mlb/scripts/03_edges/compute_ev_kelly.py"
    )

    probs = _load_module(
        "mlb_evaluate_build_juice_files",
        probability_path,
    )
    evk = _load_module(
        "mlb_evaluate_compute_ev_kelly",
        ev_kelly_path,
    )

    required_probability_helpers = [
        "moneyline_probabilities",
        "run_line_probabilities",
        "totals_probabilities",
    ]
    required_ev_helpers = [
        "compute_binary_ev",
        "compute_binary_kelly_raw",
        "compute_total_ev",
        "compute_total_kelly_raw",
    ]

    missing_probs = [
        name for name in required_probability_helpers
        if not hasattr(probs, name)
    ]
    missing_evk = [
        name for name in required_ev_helpers
        if not hasattr(evk, name)
    ]

    if missing_probs:
        fail(
            "Production probability module missing helpers: "
            f"{missing_probs}"
        )

    if missing_evk:
        fail(
            "Production EV/Kelly module missing helpers: "
            f"{missing_evk}"
        )

    return probs, evk


def duplicate_columns(columns) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []

    for col in columns:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)

    return dupes


def assert_unique_columns(df: pd.DataFrame, label: str) -> None:
    dupes = duplicate_columns(list(df.columns))
    if dupes:
        fail(f"{label} contains duplicate columns: {dupes}")


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")


def normalize_game_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.str.replace(r"\.0$", "", regex=True)
    values = values.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )
    return values


def parse_game_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype("string").str.replace("_", "-", regex=False),
        errors="coerce",
    ).dt.normalize()


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    allow_missing: bool = False,
    minimum: float | None = None,
    strictly_greater_than: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")

    bad = ~np.isfinite(values)
    if allow_missing:
        bad = values.notna() & ~np.isfinite(values)

    if minimum is not None:
        bad = bad | (values < minimum)

    if strictly_greater_than is not None:
        bad = bad | (values <= strictly_greater_than)

    if bad.any():
        sample_cols = [
            col for col in ["game_id", "game_date", column]
            if col in frame.columns
        ]
        sample = frame.loc[
            bad,
            sample_cols,
        ].head(10).to_dict("records")
        fail(
            f"Invalid numeric values in {column}; "
            f"bad_rows={int(bad.sum())}; sample={sample}"
        )

    return values


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        fail(f"{label} not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        message = f"Could not parse {label} {path}: {exc}"
        _log(message, "ERROR")
        raise RuntimeError(message) from exc

    if not isinstance(payload, dict):
        fail(f"{label} must be a JSON object: {path}")

    return payload


def validate_metadata(
    home_metadata: dict,
    away_metadata: dict,
) -> tuple[list[str], str, str, int]:
    required = [
        "test_start_date",
        "test_end_date",
        "test_row_count",
        "feature_columns",
    ]

    for label, metadata in [
        ("home metadata", home_metadata),
        ("away metadata", away_metadata),
    ]:
        missing = [key for key in required if key not in metadata]
        if missing:
            fail(f"{label} missing required keys: {missing}")

        features = metadata["feature_columns"]
        if not isinstance(features, list) or not features:
            fail(f"{label} feature_columns is empty or invalid")

        if len(features) != len(set(features)):
            fail(f"{label} feature_columns contains duplicates")

    home_features = list(home_metadata["feature_columns"])
    away_features = list(away_metadata["feature_columns"])

    if home_features != away_features:
        fail(
            "Home and away metadata feature order differs: "
            f"home={home_features} away={away_features}"
        )

    keys_that_must_match = [
        "test_start_date",
        "test_end_date",
        "test_row_count",
    ]
    for key in keys_that_must_match:
        if home_metadata[key] != away_metadata[key]:
            fail(
                f"Home/away metadata mismatch for {key}: "
                f"home={home_metadata[key]} away={away_metadata[key]}"
            )

    test_start = str(home_metadata["test_start_date"])
    test_end = str(home_metadata["test_end_date"])

    try:
        start_dt = pd.Timestamp(test_start).normalize()
        end_dt = pd.Timestamp(test_end).normalize()
    except Exception as exc:
        message = (
            "Invalid test date range in metadata: "
            f"start={test_start} end={test_end} error={exc}"
        )
        _log(message, "ERROR")
        raise RuntimeError(message) from exc

    if start_dt > end_dt:
        fail(
            "Metadata test_start_date is after test_end_date: "
            f"{test_start} > {test_end}"
        )

    try:
        test_row_count = int(home_metadata["test_row_count"])
    except (TypeError, ValueError) as exc:
        message = (
            "Metadata test_row_count is not an integer: "
            f"{home_metadata['test_row_count']}"
        )
        _log(message, "ERROR")
        raise RuntimeError(message) from exc

    if test_row_count <= 0:
        fail(f"Metadata test_row_count must be positive: {test_row_count}")

    return home_features, test_start, test_end, test_row_count


def load_test_period(
    training_path: Path,
    feature_columns: list[str],
    test_start: str,
    test_end: str,
    expected_rows: int,
) -> pd.DataFrame:
    if not training_path.exists():
        fail(f"Training set not found: {training_path}")

    df = pd.read_csv(training_path, encoding="utf-8-sig")
    if df.empty:
        fail(f"Training set is empty: {training_path}")

    assert_unique_columns(df, "training set")
    require_columns(
        df,
        TEST_REQUIRED_COLUMNS + feature_columns,
        "training set",
    )

    df = df.copy()
    df["game_id"] = normalize_game_id(df["game_id"])

    if df["game_id"].isna().any():
        sample = df.loc[
            df["game_id"].isna(),
            ["game_date", "game_id"],
        ].head(10).to_dict("records")
        fail(f"Training set contains blank game_id values: {sample}")

    duplicate_ids = df["game_id"].duplicated(keep=False)
    if duplicate_ids.any():
        sample = df.loc[
            duplicate_ids,
            ["game_date", "game_id"],
        ].head(10).to_dict("records")
        fail(
            "Training set contains duplicate game_id values; "
            f"sample={sample}"
        )

    df["_game_date_dt"] = parse_game_dates(df["game_date"])
    bad_dates = df["_game_date_dt"].isna()
    if bad_dates.any():
        sample = df.loc[
            bad_dates,
            ["game_id", "game_date"],
        ].head(10).to_dict("records")
        fail(f"Training set contains invalid game_date values: {sample}")

    start_dt = pd.Timestamp(test_start).normalize()
    end_dt = pd.Timestamp(test_end).normalize()

    test = df[
        (df["_game_date_dt"] >= start_dt)
        & (df["_game_date_dt"] <= end_dt)
    ].copy()

    if test.empty:
        fail(
            "No rows found in metadata test period: "
            f"{test_start} through {test_end}"
        )

    if len(test) != expected_rows:
        fail(
            "Test-period row count does not match saved model metadata: "
            f"metadata={expected_rows} reconstructed={len(test)}"
        )

    for col in [
        "target_home_runs",
        "target_away_runs",
        "dratings_home_projected_runs",
        "dratings_away_projected_runs",
        "dratings_total_projected_runs",
    ]:
        test[col] = numeric_series(
            test,
            col,
            minimum=0.0,
        )

    for col in feature_columns:
        values = pd.to_numeric(test[col], errors="coerce")
        invalid = values.notna() & ~np.isfinite(values)

        if invalid.any():
            sample = test.loc[
                invalid,
                ["game_id", "game_date", col],
            ].head(10).to_dict("records")
            fail(
                f"Non-finite feature values in {col}; sample={sample}"
            )

        test[col] = values

    dratings_total_diff = (
        test["dratings_total_projected_runs"]
        - (
            test["dratings_home_projected_runs"]
            + test["dratings_away_projected_runs"]
        )
    ).abs()

    if (dratings_total_diff > 1e-6).any():
        sample = test.loc[
            dratings_total_diff > 1e-6,
            [
                "game_id",
                "dratings_home_projected_runs",
                "dratings_away_projected_runs",
                "dratings_total_projected_runs",
            ],
        ].head(10).to_dict("records")
        fail(
            "DRatings total projected runs does not equal home + away; "
            f"sample={sample}"
        )

    return test


def validate_model_feature_order(
    model,
    feature_columns: list[str],
    label: str,
) -> None:
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is None:
        fail(
            f"{label} model does not expose feature_names_in_; "
            "cannot validate production feature order"
        )

    actual = [str(value) for value in model_features.tolist()]
    if actual != feature_columns:
        fail(
            f"{label} model feature order differs from metadata: "
            f"model={actual} metadata={feature_columns}"
        )


def score_models(
    test: pd.DataFrame,
    feature_columns: list[str],
    model_dir: Path,
) -> pd.DataFrame:
    home_path = model_dir / HOME_MODEL_NAME
    away_path = model_dir / AWAY_MODEL_NAME

    for path in [home_path, away_path]:
        if not path.exists():
            fail(f"Model artifact not found: {path}")

    home_model = joblib.load(home_path)
    away_model = joblib.load(away_path)

    validate_model_feature_order(
        home_model,
        feature_columns,
        "home",
    )
    validate_model_feature_order(
        away_model,
        feature_columns,
        "away",
    )

    x_features = test[feature_columns]

    home_predictions = np.asarray(
        home_model.predict(x_features),
        dtype=float,
    )
    away_predictions = np.asarray(
        away_model.predict(x_features),
        dtype=float,
    )

    for label, predictions in [
        ("home", home_predictions),
        ("away", away_predictions),
    ]:
        if len(predictions) != len(test):
            fail(
                f"{label} model returned wrong prediction count: "
                f"expected={len(test)} actual={len(predictions)}"
            )

        if np.any(~np.isfinite(predictions)):
            fail(f"{label} model produced non-finite predictions")

        if np.any(predictions < 0):
            fail(f"{label} model produced negative run predictions")

    scored = test.copy()
    scored["model_home_runs"] = home_predictions
    scored["model_away_runs"] = away_predictions
    scored["model_total_runs"] = (
        scored["model_home_runs"]
        + scored["model_away_runs"]
    )

    return scored


def safe_poisson_deviance(
    actual,
    predicted,
) -> float:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    if np.any(~np.isfinite(y_true)) or np.any(y_true < 0):
        fail("Invalid actual values supplied to Poisson deviance")

    if np.any(~np.isfinite(y_pred)) or np.any(y_pred < 0):
        fail("Invalid predicted values supplied to Poisson deviance")

    return float(
        mean_poisson_deviance(
            y_true,
            np.maximum(y_pred, EPSILON),
        )
    )


def build_run_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    definitions = [
        (
            "dratings",
            "home",
            "dratings_home_projected_runs",
            "target_home_runs",
        ),
        (
            "new_model",
            "home",
            "model_home_runs",
            "target_home_runs",
        ),
        (
            "dratings",
            "away",
            "dratings_away_projected_runs",
            "target_away_runs",
        ),
        (
            "new_model",
            "away",
            "model_away_runs",
            "target_away_runs",
        ),
    ]

    for system, side, pred_col, actual_col in definitions:
        actual = scored[actual_col].to_numpy(dtype=float)
        predicted = scored[pred_col].to_numpy(dtype=float)

        rows.append(
            {
                "system": system,
                "side": side,
                "rows": int(len(scored)),
                "mae": float(
                    mean_absolute_error(
                        actual,
                        predicted,
                    )
                ),
                "mean_poisson_deviance": safe_poisson_deviance(
                    actual,
                    predicted,
                ),
                "mean_predicted_runs": float(np.mean(predicted)),
                "mean_actual_runs": float(np.mean(actual)),
            }
        )

    return pd.DataFrame(rows)


def build_promotion_decision(
    run_metrics: pd.DataFrame,
    log_loss: pd.DataFrame,
    *,
    test_start: str,
    test_end: str,
) -> dict:
    lookup = {
        (str(row.system), str(row.side)): row
        for row in run_metrics.itertuples(index=False)
    }

    required = [
        ("dratings", "home"),
        ("new_model", "home"),
        ("dratings", "away"),
        ("new_model", "away"),
    ]
    missing = [key for key in required if key not in lookup]
    if missing:
        fail(f"Promotion gate missing required run metrics: {missing}")

    comparison = {}
    all_passed = True

    for side in ["home", "away"]:
        baseline = float(
            lookup[("dratings", side)].mean_poisson_deviance
        )
        candidate = float(
            lookup[("new_model", side)].mean_poisson_deviance
        )

        if not np.isfinite(baseline) or not np.isfinite(candidate):
            fail(
                f"Promotion gate received non-finite {side} Poisson deviance: "
                f"baseline={baseline} candidate={candidate}"
            )

        passed = bool(candidate <= baseline)
        all_passed = all_passed and passed

        comparison[side] = {
            "baseline_system": "dratings",
            "metric": "mean_poisson_deviance",
            "baseline_value": baseline,
            "candidate_value": candidate,
            "candidate_lte_baseline": passed,
            "difference_candidate_minus_baseline": candidate - baseline,
        }

    # Market probabilities are the quantities consumed by edge/EV/Kelly.
    # Promotion must not improve runs at the expense of any evaluated market.
    market_lookup = {
        (str(row.system), str(row.market), str(row.evaluation_side)): row
        for row in log_loss.itertuples(index=False)
    }
    market_comparison = {}
    for market_name, side in (
        ("moneyline", "home"),
        ("run_line", "home"),
        ("total", "over_resolved"),
    ):
        baseline_key = ("dratings", market_name, side)
        candidate_key = ("new_model", market_name, side)
        if baseline_key not in market_lookup or candidate_key not in market_lookup:
            fail(f"Promotion gate missing market log-loss rows: {market_name}")
        baseline_row = market_lookup[baseline_key]
        candidate_row = market_lookup[candidate_key]
        baseline_rows = int(baseline_row.rows)
        candidate_rows = int(candidate_row.rows)
        if baseline_rows <= 0 or candidate_rows != baseline_rows:
            fail(
                f"Promotion gate requires identical nonempty market samples "
                f"for {market_name}: baseline={baseline_rows} "
                f"candidate={candidate_rows}"
            )
        baseline = float(baseline_row.log_loss)
        candidate = float(candidate_row.log_loss)
        if not np.isfinite(baseline) or not np.isfinite(candidate):
            fail(f"Non-finite market log loss in promotion gate: {market_name}")
        passed = bool(candidate <= baseline)
        all_passed = all_passed and passed
        market_comparison[market_name] = {
            "baseline_system": "dratings",
            "metric": "log_loss",
            "evaluation_side": side,
            "rows": baseline_rows,
            "baseline_value": baseline,
            "candidate_value": candidate,
            "candidate_lte_baseline": passed,
            "difference_candidate_minus_baseline": candidate - baseline,
        }

    return {
        "status": (
            "candidate_promoted"
            if all_passed
            else "candidate_rejected"
        ),
        "gate_passed": bool(all_passed),
        "gate_rule": (
            "candidate mean_poisson_deviance <= DRatings for BOTH sides "
            "AND candidate log_loss <= DRatings for ALL evaluated markets "
            "(moneyline, run_line, totals)"
        ),
        "test_start_date": str(test_start),
        "test_end_date": str(test_end),
        "comparison": comparison,
        "market_comparison": market_comparison,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _restore_production_backups(
    backups: dict[Path, bytes | None],
) -> None:
    for path, original in backups.items():
        if original is None:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes(original)


def apply_promotion_decision(
    *,
    decision: dict,
    candidate_dir: Path,
    production_model_dir: Path,
    home_candidate_metadata: dict,
    away_candidate_metadata: dict,
    report_path: Path,
) -> dict:
    evaluated_at = _now()
    result = {
        **decision,
        "evaluated_at": evaluated_at,
        "candidate_dir": str(candidate_dir),
        "production_model_dir": str(production_model_dir),
        "production_artifacts_changed": False,
    }

    if not decision["gate_passed"]:
        result["notes"] = (
            "Candidate failed at least one run-deviance or market-log-loss "
            "gate. Existing production model and metadata artifacts "
            "were left unchanged."
        )
        _write_json(report_path, result)
        _log(
            "PROMOTION candidate_rejected "
            f"comparison={decision['comparison']}"
        )
        return result

    production_model_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = {
        "home_model": candidate_dir / HOME_MODEL_NAME,
        "away_model": candidate_dir / AWAY_MODEL_NAME,
        "home_metadata": candidate_dir / HOME_METADATA_NAME,
        "away_metadata": candidate_dir / AWAY_METADATA_NAME,
    }

    missing = [
        str(path)
        for path in candidate_paths.values()
        if not path.exists()
    ]
    if missing:
        fail(
            "Promotion gate passed but candidate artifacts are missing: "
            f"{missing}"
        )

    production_paths = {
        "home_model": production_model_dir / HOME_MODEL_NAME,
        "away_model": production_model_dir / AWAY_MODEL_NAME,
        "home_metadata": production_model_dir / HOME_METADATA_NAME,
        "away_metadata": production_model_dir / AWAY_METADATA_NAME,
    }

    promotion_metadata = {
        "status": "candidate_promoted",
        "promoted_at": evaluated_at,
        "gate_rule": decision["gate_rule"],
        "test_start_date": decision["test_start_date"],
        "test_end_date": decision["test_end_date"],
        "comparison": decision["comparison"],
    }

    home_production_metadata = dict(home_candidate_metadata)
    away_production_metadata = dict(away_candidate_metadata)

    home_production_metadata["artifact_stage"] = "production"
    away_production_metadata["artifact_stage"] = "production"
    home_production_metadata["promotion_status"] = "candidate_promoted"
    away_production_metadata["promotion_status"] = "candidate_promoted"
    home_production_metadata["promotion_baseline_comparison"] = (
        promotion_metadata
    )
    away_production_metadata["promotion_baseline_comparison"] = (
        promotion_metadata
    )

    staged_paths = {
        key: path.with_name(path.name + ".candidate_tmp")
        for key, path in production_paths.items()
    }

    backups = {
        path: path.read_bytes() if path.exists() else None
        for path in production_paths.values()
    }

    try:
        shutil.copy2(
            candidate_paths["home_model"],
            staged_paths["home_model"],
        )
        shutil.copy2(
            candidate_paths["away_model"],
            staged_paths["away_model"],
        )
        _write_json(
            staged_paths["home_metadata"],
            home_production_metadata,
        )
        _write_json(
            staged_paths["away_metadata"],
            away_production_metadata,
        )

        for key in [
            "home_model",
            "away_model",
            "home_metadata",
            "away_metadata",
        ]:
            staged_paths[key].replace(production_paths[key])

    except Exception:
        for staged in staged_paths.values():
            if staged.exists():
                staged.unlink()
        _restore_production_backups(backups)
        raise

    result["production_artifacts_changed"] = True
    result["production_artifacts"] = {
        key: str(path)
        for key, path in production_paths.items()
    }
    result["notes"] = (
        "Both candidate run models met or beat the DRatings baseline "
        "Poisson deviance and were promoted together."
    )

    _write_json(report_path, result)
    _log(
        "PROMOTION candidate_promoted "
        f"comparison={decision['comparison']}"
    )

    return result


def sportsbook_path_for_date(
    sportsbook_dir: Path,
    value,
) -> Path:
    date_text = pd.Timestamp(value).strftime("%Y_%m_%d")
    return sportsbook_dir / f"{date_text}_MLB.csv"


def load_sportsbook_test_period(
    scored: pd.DataFrame,
    sportsbook_dir: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    unique_dates = sorted(
        scored["_game_date_dt"].drop_duplicates().tolist()
    )

    for game_date in unique_dates:
        path = sportsbook_path_for_date(
            sportsbook_dir,
            game_date,
        )

        if not path.exists():
            fail(
                "Missing historical sportsbook file required for "
                f"test-period market evaluation: {path}"
            )

        frame = pd.read_csv(path, encoding="utf-8-sig")
        if frame.empty:
            fail(f"Historical sportsbook file is empty: {path}")

        assert_unique_columns(
            frame,
            f"sportsbook {path}",
        )
        require_columns(
            frame,
            SPORTSBOOK_REQUIRED_COLUMNS,
            f"sportsbook {path}",
        )

        frame = frame[SPORTSBOOK_REQUIRED_COLUMNS].copy()
        frame["game_id"] = normalize_game_id(frame["game_id"])

        if frame["game_id"].isna().any():
            fail(f"Sportsbook file contains blank game_id: {path}")

        duplicates = frame["game_id"].duplicated(keep=False)
        if duplicates.any():
            sample = frame.loc[
                duplicates,
                ["game_id", "game_date"],
            ].head(10).to_dict("records")
            fail(
                f"Sportsbook file contains duplicate game_id: "
                f"{path}; sample={sample}"
            )

        frames.append(frame)

    sportsbook = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    duplicate_ids = sportsbook["game_id"].duplicated(keep=False)
    if duplicate_ids.any():
        sample = sportsbook.loc[
            duplicate_ids,
            ["game_id", "game_date"],
        ].head(10).to_dict("records")
        fail(
            "Sportsbook test-period files contain duplicate game_id "
            f"across dates; sample={sample}"
        )

    line_columns = [
        "away_run_line",
        "home_run_line",
        "total",
    ]
    for col in line_columns:
        sportsbook[col] = numeric_series(
            sportsbook,
            col,
            allow_missing=col in {
                "away_run_line",
                "home_run_line",
            },
        )

    price_columns = [
        "away_dk_run_line_decimal",
        "home_dk_run_line_decimal",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
        "away_dk_moneyline_decimal",
        "home_dk_moneyline_decimal",
    ]

    for col in price_columns:
        values = pd.to_numeric(
            sportsbook[col],
            errors="coerce",
        )
        invalid_nonmissing = (
            values.notna()
            & (
                ~np.isfinite(values)
                | (values <= 1.0)
            )
        )
        if invalid_nonmissing.any():
            sample = sportsbook.loc[
                invalid_nonmissing,
                ["game_id", "game_date", col],
            ].head(10).to_dict("records")
            fail(
                f"Sportsbook {col} contains invalid non-missing prices; "
                f"sample={sample}"
            )
        sportsbook[col] = values

    return sportsbook


def join_sportsbook(
    scored: pd.DataFrame,
    sportsbook: pd.DataFrame,
) -> pd.DataFrame:
    sportsbook_cols = [
        "game_id",
        "home_team",
        "away_team",
        "away_run_line",
        "home_run_line",
        "total",
        "away_dk_run_line_decimal",
        "home_dk_run_line_decimal",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
        "away_dk_moneyline_decimal",
        "home_dk_moneyline_decimal",
    ]

    market = scored.merge(
        sportsbook[sportsbook_cols],
        on="game_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_sportsbook"),
        indicator=True,
    )

    missing = market["_merge"] != "both"
    if missing.any():
        sample = market.loc[
            missing,
            ["game_id", "game_date", "home_team", "away_team"],
        ].head(10).to_dict("records")
        fail(
            "Test rows missing historical sportsbook match; "
            f"bad_rows={int(missing.sum())}; sample={sample}"
        )

    market = market.drop(columns=["_merge"])

    for side in ["home", "away"]:
        training_team = market[f"{side}_team"].astype("string").str.strip()
        sportsbook_team = (
            market[f"{side}_team_sportsbook"]
            .astype("string")
            .str.strip()
        )
        mismatch = training_team != sportsbook_team

        if mismatch.any():
            sample = market.loc[
                mismatch,
                [
                    "game_id",
                    f"{side}_team",
                    f"{side}_team_sportsbook",
                ],
            ].head(10).to_dict("records")
            fail(
                f"{side} team mismatch between training and sportsbook "
                f"rows; sample={sample}"
            )

    market = market.drop(
        columns=[
            "home_team_sportsbook",
            "away_team_sportsbook",
        ]
    )

    run_line_available = (
        market["home_run_line"].notna()
        & market["away_run_line"].notna()
    )
    partial_run_line = (
        market["home_run_line"].notna()
        ^ market["away_run_line"].notna()
    )

    if partial_run_line.any():
        sample = market.loc[
            partial_run_line,
            [
                "game_id",
                "home_run_line",
                "away_run_line",
            ],
        ].head(10).to_dict("records")
        fail(
            "Untouched test period contains incomplete run-line markets; "
            f"sample={sample}"
        )

    unsupported_run_line = run_line_available & (
        (
            (
                market["home_run_line"]
                + market["away_run_line"]
            ).abs()
            > 1e-6
        )
        | ~(
            market["home_run_line"].round(6).isin([-1.5, 1.5])
            & market["away_run_line"].round(6).isin([-1.5, 1.5])
        )
    )

    if unsupported_run_line.any():
        sample = market.loc[
            unsupported_run_line,
            [
                "game_id",
                "home_run_line",
                "away_run_line",
            ],
        ].head(10).to_dict("records")
        fail(
            "Untouched test period contains unsupported/noncomplementary "
            f"run-line markets; sample={sample}"
        )

    total_fraction = (
        market["total"]
        - market["total"].round()
    ).abs()
    supported_total = (
        (total_fraction < 1e-9)
        | ((total_fraction - 0.5).abs() < 1e-9)
    )

    if (~supported_total).any():
        sample = market.loc[
            ~supported_total,
            ["game_id", "total"],
        ].head(10).to_dict("records")
        fail(
            "Untouched test period contains unsupported total lines; "
            f"sample={sample}"
        )

    return market


def _probability_triplet(
    probs_module,
    home_runs: float,
    away_runs: float,
    total_line: float,
):
    p_over, p_under, p_push = probs_module.totals_probabilities(
        home_runs,
        away_runs,
        total_line,
    )

    return (
        float(p_over),
        float(p_under),
        float(p_push),
    )


def derive_market_probabilities(
    market: pd.DataFrame,
    probs_module,
) -> pd.DataFrame:
    out = market.copy()

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

    for system, (home_col, away_col) in systems.items():
        ml_home: list[float] = []
        ml_away: list[float] = []
        rl_home: list[float] = []
        rl_away: list[float] = []
        total_over: list[float] = []
        total_under: list[float] = []
        total_push: list[float] = []

        for row in out.itertuples(index=False):
            home_runs = float(getattr(row, home_col))
            away_runs = float(getattr(row, away_col))

            p_home_ml, p_away_ml, _ = (
                probs_module.moneyline_probabilities(
                    home_runs,
                    away_runs,
                )
            )
            if (
                pd.notna(row.home_run_line)
                and pd.notna(row.away_run_line)
            ):
                p_home_rl, p_away_rl = (
                    probs_module.run_line_probabilities(
                        home_runs,
                        away_runs,
                        float(row.home_run_line),
                        float(row.away_run_line),
                    )
                )
            else:
                p_home_rl = np.nan
                p_away_rl = np.nan
            p_over, p_under, p_push = _probability_triplet(
                probs_module,
                home_runs,
                away_runs,
                float(row.total),
            )

            ml_home.append(float(p_home_ml))
            ml_away.append(float(p_away_ml))
            rl_home.append(float(p_home_rl))
            rl_away.append(float(p_away_rl))
            total_over.append(p_over)
            total_under.append(p_under)
            total_push.append(p_push)

        out[f"{system}_home_ml_prob"] = ml_home
        out[f"{system}_away_ml_prob"] = ml_away
        out[f"{system}_home_rl_prob"] = rl_home
        out[f"{system}_away_rl_prob"] = rl_away
        out[f"{system}_over_total_win_prob"] = total_over
        out[f"{system}_under_total_win_prob"] = total_under
        out[f"{system}_total_push_prob"] = total_push

        resolved = (
            out[f"{system}_over_total_win_prob"]
            + out[f"{system}_under_total_win_prob"]
        )

        bad_resolved = (
            ~np.isfinite(resolved)
            | (resolved <= 0)
        )
        if bad_resolved.any():
            sample = out.loc[
                bad_resolved,
                ["game_id", "total"],
            ].head(10).to_dict("records")
            fail(
                f"{system} total probabilities have zero/invalid "
                f"resolved mass; sample={sample}"
            )

        out[f"{system}_over_total_conditional_prob"] = (
            out[f"{system}_over_total_win_prob"]
            / resolved
        )
        out[f"{system}_under_total_conditional_prob"] = (
            out[f"{system}_under_total_win_prob"]
            / resolved
        )

    probability_cols = [
        col for col in out.columns
        if (
            col.endswith("_prob")
            or col.endswith("_win_prob")
            or col.endswith("_conditional_prob")
        )
    ]

    run_line_available = (
        out["home_run_line"].notna()
        & out["away_run_line"].notna()
    )

    for col in probability_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        if col.endswith("_rl_prob"):
            bad = (
                (run_line_available & values.isna())
                | (
                    values.notna()
                    & (
                        ~np.isfinite(values)
                        | (values < -PROB_TOLERANCE)
                        | (values > 1.0 + PROB_TOLERANCE)
                    )
                )
            )
        else:
            bad = (
                values.isna()
                | ~np.isfinite(values)
                | (values < -PROB_TOLERANCE)
                | (values > 1.0 + PROB_TOLERANCE)
            )
        if bad.any():
            sample = out.loc[
                bad,
                ["game_id", col],
            ].head(10).to_dict("records")
            fail(
                f"Derived probability {col} is invalid; sample={sample}"
            )

    return out


def add_observed_outcomes(market: pd.DataFrame) -> pd.DataFrame:
    out = market.copy()

    home_runs = out["target_home_runs"].to_numpy(dtype=float)
    away_runs = out["target_away_runs"].to_numpy(dtype=float)

    tie = np.isclose(
        home_runs,
        away_runs,
        atol=PROB_TOLERANCE,
        rtol=0.0,
    )
    if tie.any():
        sample = out.loc[
            tie,
            [
                "game_id",
                "target_home_runs",
                "target_away_runs",
            ],
        ].head(10).to_dict("records")
        fail(
            "Completed MLB test rows contain tied final scores; "
            f"sample={sample}"
        )

    out["observed_home_ml_win"] = (
        home_runs > away_runs
    ).astype(int)
    out["observed_away_ml_win"] = (
        1 - out["observed_home_ml_win"]
    )

    home_adjusted = (
        home_runs
        + out["home_run_line"].to_numpy(dtype=float)
    )
    away_adjusted = (
        away_runs
        + out["away_run_line"].to_numpy(dtype=float)
    )

    run_line_available = (
        out["home_run_line"].notna()
        & out["away_run_line"].notna()
    ).to_numpy(dtype=bool)

    out["observed_home_rl_win"] = np.where(
        run_line_available,
        (home_adjusted > away_runs).astype(float),
        np.nan,
    )
    out["observed_away_rl_win"] = np.where(
        run_line_available,
        (away_adjusted > home_runs).astype(float),
        np.nan,
    )

    rl_push = run_line_available & (
        np.isclose(
            home_adjusted,
            away_runs,
            atol=PROB_TOLERANCE,
            rtol=0.0,
        )
        | np.isclose(
            away_adjusted,
            home_runs,
            atol=PROB_TOLERANCE,
            rtol=0.0,
        )
    )

    if rl_push.any():
        sample = out.loc[
            rl_push,
            [
                "game_id",
                "target_home_runs",
                "target_away_runs",
                "home_run_line",
                "away_run_line",
            ],
        ].head(10).to_dict("records")
        fail(
            "Standard half-run run-line market produced a push; "
            f"sample={sample}"
        )

    actual_total = home_runs + away_runs
    line = out["total"].to_numpy(dtype=float)

    out["observed_total_push"] = np.isclose(
        actual_total,
        line,
        atol=PROB_TOLERANCE,
        rtol=0.0,
    ).astype(int)

    out["observed_over_win"] = np.where(
        out["observed_total_push"].eq(1),
        np.nan,
        (actual_total > line).astype(float),
    )
    out["observed_under_win"] = np.where(
        out["observed_total_push"].eq(1),
        np.nan,
        (actual_total < line).astype(float),
    )

    return out


def calibration_table(
    records: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "system",
        "side",
        "predicted_probability",
        "observed_win",
    ]
    require_columns(
        records,
        required,
        "calibration records",
    )

    frame = records.copy()
    frame["predicted_probability"] = pd.to_numeric(
        frame["predicted_probability"],
        errors="coerce",
    )

    bad_probability = (
        frame["predicted_probability"].isna()
        | ~np.isfinite(frame["predicted_probability"])
        | (frame["predicted_probability"] < 0)
        | (frame["predicted_probability"] > 1)
    )
    if bad_probability.any():
        fail("Calibration records contain invalid probabilities")

    frame["probability_bin"] = pd.cut(
        frame["predicted_probability"],
        bins=PROBABILITY_BINS,
        labels=PROBABILITY_BIN_LABELS,
        include_lowest=True,
        right=True,
    )

    rows: list[dict] = []

    for (
        system,
        side,
        probability_bin,
    ), group in frame.groupby(
        ["system", "side", "probability_bin"],
        observed=False,
        dropna=False,
    ):
        if group.empty:
            continue

        resolved = group["observed_win"].notna()
        resolved_group = group.loc[resolved]

        if resolved_group.empty:
            observed_rate = np.nan
            mean_probability = float(
                group["predicted_probability"].mean()
            )
            abs_error = np.nan
        else:
            observed_rate = float(
                pd.to_numeric(
                    resolved_group["observed_win"],
                    errors="coerce",
                ).mean()
            )
            mean_probability = float(
                resolved_group["predicted_probability"].mean()
            )
            abs_error = abs(
                mean_probability - observed_rate
            )

        rows.append(
            {
                "system": str(system),
                "side": str(side),
                "probability_bin": str(probability_bin),
                "rows": int(len(group)),
                "resolved_rows": int(resolved.sum()),
                "pushes_excluded": int((~resolved).sum()),
                "mean_predicted_probability": mean_probability,
                "observed_win_rate": observed_rate,
                "absolute_calibration_error": abs_error,
            }
        )

    return pd.DataFrame(rows)


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

            run_line_records.extend(
                {
                    "system": system,
                    "side": side,
                    "predicted_probability": probability,
                    "observed_win": observed,
                }
                for probability, observed in zip(
                    market[f"{system}_{side}_rl_prob"],
                    market[f"observed_{side}_rl_win"],
                )
                if (
                    pd.notna(probability)
                    and pd.notna(observed)
                )
            )

        for side in ["over", "under"]:
            total_records.extend(
                {
                    "system": system,
                    "side": side,
                    "predicted_probability": probability,
                    "observed_win": observed,
                }
                for probability, observed in zip(
                    market[
                        f"{system}_{side}_total_conditional_prob"
                    ],
                    market[f"observed_{side}_win"],
                )
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


def binary_log_loss(
    observed,
    probability,
) -> float:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(probability, dtype=float)

    valid = (
        np.isfinite(y)
        & np.isfinite(p)
    )
    y = y[valid]
    p = p[valid]

    if y.size == 0:
        return float("nan")

    p = np.clip(
        p,
        EPSILON,
        1.0 - EPSILON,
    )

    return float(
        np.mean(
            -(
                y * np.log(p)
                + (1.0 - y) * np.log(1.0 - p)
            )
        )
    )


def build_probability_log_loss(
    market: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    definitions = [
        (
            "moneyline",
            "home",
            "observed_home_ml_win",
            "home_ml_prob",
        ),
        (
            "run_line",
            "home",
            "observed_home_rl_win",
            "home_rl_prob",
        ),
        (
            "total",
            "over_resolved",
            "observed_over_win",
            "over_total_conditional_prob",
        ),
    ]

    for (
        market_name,
        evaluation_side,
        observed_col,
        probability_suffix,
    ) in definitions:
        observed = pd.to_numeric(
            market[observed_col], errors="coerce"
        )
        baseline = pd.to_numeric(
            market[f"dratings_{probability_suffix}"], errors="coerce"
        )
        candidate = pd.to_numeric(
            market[f"new_model_{probability_suffix}"], errors="coerce"
        )
        # Compare both systems on the EXACT same resolved games, not
        # merely samples with equal row counts.
        valid = (
            np.isfinite(observed)
            & np.isfinite(baseline)
            & np.isfinite(candidate)
        )
        for system, probabilities in (
            ("dratings", baseline),
            ("new_model", candidate),
        ):
            rows.append(
                {
                    "system": system,
                    "market": market_name,
                    "evaluation_side": evaluation_side,
                    "rows": int(valid.sum()),
                    "log_loss": binary_log_loss(
                        observed[valid],
                        probabilities[valid],
                    ),
                }
            )

    return pd.DataFrame(rows)


def _scalar(value) -> float:
    if isinstance(value, pd.Series):
        if len(value) != 1:
            fail(
                "Expected a one-value Series from production math "
                f"helper; got {len(value)}"
            )
        return float(value.iloc[0])

    array = np.asarray(value, dtype=float)
    if array.size != 1:
        fail(
            "Expected scalar-like value from production math helper; "
            f"shape={array.shape}"
        )

    return float(array.reshape(-1)[0])


def binary_ev(
    evk_module,
    probability: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        evk_module.compute_binary_ev(
            pd.Series([probability], dtype=float),
            pd.Series([decimal_odds], dtype=float),
        )
    )


def binary_kelly_raw(
    evk_module,
    probability: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        evk_module.compute_binary_kelly_raw(
            pd.Series([probability], dtype=float),
            pd.Series([decimal_odds], dtype=float),
        )
    )


def total_ev(
    evk_module,
    p_win: float,
    p_loss: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        evk_module.compute_total_ev(
            pd.Series([p_win], dtype=float),
            pd.Series([p_loss], dtype=float),
            pd.Series([decimal_odds], dtype=float),
        )
    )


def total_kelly_raw(
    evk_module,
    p_win: float,
    p_loss: float,
    decimal_odds: float,
    game_id: str,
    label: str,
) -> float:
    return _scalar(
        evk_module.compute_total_kelly_raw(
            pd.Series([p_win], dtype=float),
            pd.Series([p_loss], dtype=float),
            pd.Series([decimal_odds], dtype=float),
            pd.Series([game_id], dtype="string"),
            label,
        )
    )


def realized_return(
    observed_win: float | int | None,
    decimal_odds: float,
) -> float:
    if observed_win is None or pd.isna(observed_win):
        return 0.0

    if float(observed_win) == 1.0:
        return float(decimal_odds - 1.0)

    if float(observed_win) == 0.0:
        return -1.0

    return fail(f"Invalid observed_win value: {observed_win}")


def build_value_records(
    market: pd.DataFrame,
    evk_module,
) -> pd.DataFrame:
    rows: list[dict] = []

    for row in market.itertuples(index=False):
        game_id = str(row.game_id)

        for system in ["dratings", "new_model"]:
            binary_candidates = [
                (
                    "moneyline",
                    "home",
                    float(getattr(row, f"{system}_home_ml_prob")),
                    row.home_dk_moneyline_decimal,
                    int(row.observed_home_ml_win),
                    np.nan,
                ),
                (
                    "moneyline",
                    "away",
                    float(getattr(row, f"{system}_away_ml_prob")),
                    row.away_dk_moneyline_decimal,
                    int(row.observed_away_ml_win),
                    np.nan,
                ),
                (
                    "run_line",
                    "home",
                    getattr(row, f"{system}_home_rl_prob"),
                    row.home_dk_run_line_decimal,
                    row.observed_home_rl_win,
                    row.home_run_line,
                ),
                (
                    "run_line",
                    "away",
                    getattr(row, f"{system}_away_rl_prob"),
                    row.away_dk_run_line_decimal,
                    row.observed_away_rl_win,
                    row.away_run_line,
                ),
            ]

            for (
                market_name,
                side,
                probability,
                price,
                observed,
                line,
            ) in binary_candidates:
                if pd.isna(price):
                    continue

                if (
                    market_name == "run_line"
                    and (
                        pd.isna(probability)
                        or pd.isna(observed)
                        or pd.isna(line)
                    )
                ):
                    continue

                probability = float(probability)
                observed = int(observed)
                line = (
                    float(line)
                    if not pd.isna(line)
                    else np.nan
                )

                decimal_odds = float(price)
                break_even = 1.0 / decimal_odds
                edge = probability - break_even
                ev = binary_ev(
                    evk_module,
                    probability,
                    decimal_odds,
                )
                raw_kelly = binary_kelly_raw(
                    evk_module,
                    probability,
                    decimal_odds,
                )

                rows.append(
                    {
                        "game_id": game_id,
                        "system": system,
                        "market": market_name,
                        "side": side,
                        "line": line,
                        "model_probability": probability,
                        "decimal_odds": decimal_odds,
                        "break_even_probability": break_even,
                        "probability_edge": edge,
                        "ev": ev,
                        "kelly_raw": raw_kelly,
                        "kelly": max(raw_kelly, 0.0),
                        "realized_return": realized_return(
                            observed,
                            decimal_odds,
                        ),
                    }
                )

            total_candidates = [
                (
                    "over",
                    float(
                        getattr(
                            row,
                            f"{system}_over_total_win_prob",
                        )
                    ),
                    float(
                        getattr(
                            row,
                            f"{system}_under_total_win_prob",
                        )
                    ),
                    float(
                        getattr(
                            row,
                            f"{system}_over_total_conditional_prob",
                        )
                    ),
                    row.dk_total_over_decimal,
                    row.observed_over_win,
                ),
                (
                    "under",
                    float(
                        getattr(
                            row,
                            f"{system}_under_total_win_prob",
                        )
                    ),
                    float(
                        getattr(
                            row,
                            f"{system}_over_total_win_prob",
                        )
                    ),
                    float(
                        getattr(
                            row,
                            f"{system}_under_total_conditional_prob",
                        )
                    ),
                    row.dk_total_under_decimal,
                    row.observed_under_win,
                ),
            ]

            for (
                side,
                p_win,
                p_loss,
                conditional_probability,
                price,
                observed,
            ) in total_candidates:
                if pd.isna(price):
                    continue

                decimal_odds = float(price)
                break_even = 1.0 / decimal_odds
                edge = (
                    conditional_probability
                    - break_even
                )
                ev = total_ev(
                    evk_module,
                    p_win,
                    p_loss,
                    decimal_odds,
                )
                raw_kelly = total_kelly_raw(
                    evk_module,
                    p_win,
                    p_loss,
                    decimal_odds,
                    game_id,
                    f"evaluate_run_model {system} {side}",
                )

                rows.append(
                    {
                        "game_id": game_id,
                        "system": system,
                        "market": "total",
                        "side": side,
                        "line": float(row.total),
                        "model_probability": conditional_probability,
                        "decimal_odds": decimal_odds,
                        "break_even_probability": break_even,
                        "probability_edge": edge,
                        "ev": ev,
                        "kelly_raw": raw_kelly,
                        "kelly": max(raw_kelly, 0.0),
                        "realized_return": realized_return(
                            observed,
                            decimal_odds,
                        ),
                    }
                )

    values = pd.DataFrame(rows)

    if values.empty:
        fail(
            "No valid sportsbook prices were available for EV/Kelly "
            "and realized-return evaluation"
        )

    return values


def expected_calibration_error(
    calibration: pd.DataFrame,
    system: str,
) -> float:
    frame = calibration[
        calibration["system"] == system
    ].copy()

    frame = frame[
        frame["resolved_rows"] > 0
    ].copy()

    if frame.empty:
        return float("nan")

    weights = frame["resolved_rows"].to_numpy(dtype=float)
    errors = frame[
        "absolute_calibration_error"
    ].to_numpy(dtype=float)

    return float(
        np.average(
            errors,
            weights=weights,
        )
    )


def calibration_monotonicity(
    calibration: pd.DataFrame,
    system: str,
) -> dict:
    frame = calibration[
        (calibration["system"] == system)
        & (calibration["resolved_rows"] > 0)
    ].copy()

    if frame.empty:
        return {
            "spearman": float("nan"),
            "exact_non_decreasing": False,
            "bins": 0,
        }

    grouped_rows: list[dict] = []

    for probability_bin, group in frame.groupby(
        "probability_bin",
        sort=False,
        observed=True,
    ):
        rows = int(group["resolved_rows"].sum())

        if rows <= 0:
            continue

        grouped_rows.append(
            {
                "probability_bin": str(probability_bin),
                "predicted": float(
                    np.average(
                        group["mean_predicted_probability"],
                        weights=group["resolved_rows"],
                    )
                ),
                "observed": float(
                    np.average(
                        group["observed_win_rate"],
                        weights=group["resolved_rows"],
                    )
                ),
                "rows": rows,
            }
        )

    grouped = pd.DataFrame(grouped_rows)

    if grouped.empty:
        return {
            "spearman": float("nan"),
            "exact_non_decreasing": False,
            "bins": 0,
        }

    grouped = grouped.sort_values("predicted")

    if len(grouped) < 2:
        return {
            "spearman": float("nan"),
            "exact_non_decreasing": False,
            "bins": int(len(grouped)),
        }

    rho = spearmanr(
        grouped["predicted"],
        grouped["observed"],
        nan_policy="omit",
    ).statistic

    observed = grouped["observed"].to_numpy(dtype=float)
    exact = bool(
        np.all(
            np.diff(observed) >= -PROB_TOLERANCE
        )
    )

    return {
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "exact_non_decreasing": exact,
        "bins": int(len(grouped)),
    }


def value_summary(
    values: pd.DataFrame,
    system: str,
) -> dict:
    frame = values[
        values["system"] == system
    ].copy()

    if frame.empty:
        return {
            "rows": 0,
            "positive_ev_rows": 0,
            "mean_predicted_ev": float("nan"),
            "mean_realized_return": float("nan"),
            "positive_ev_mean_predicted_ev": float("nan"),
            "positive_ev_mean_realized_return": float("nan"),
            "ev_return_spearman": float("nan"),
        }

    positive = frame[frame["ev"] > 0].copy()

    if len(frame) >= 2:
        rho = spearmanr(
            frame["ev"],
            frame["realized_return"],
            nan_policy="omit",
        ).statistic
    else:
        rho = float("nan")

    return {
        "rows": int(len(frame)),
        "positive_ev_rows": int(len(positive)),
        "mean_predicted_ev": float(frame["ev"].mean()),
        "mean_realized_return": float(
            frame["realized_return"].mean()
        ),
        "positive_ev_mean_predicted_ev": (
            float(positive["ev"].mean())
            if not positive.empty
            else float("nan")
        ),
        "positive_ev_mean_realized_return": (
            float(positive["realized_return"].mean())
            if not positive.empty
            else float("nan")
        ),
        "ev_return_spearman": (
            float(rho)
            if np.isfinite(rho)
            else float("nan")
        ),
    }


def kelly_edge_monotonicity(
    values: pd.DataFrame,
    system: str,
) -> dict:
    frame = values[
        values["system"] == system
    ].copy()

    frame = frame[
        np.isfinite(frame["probability_edge"])
        & np.isfinite(frame["kelly_raw"])
    ].copy()

    if len(frame) < 2:
        return {
            "spearman": float("nan"),
            "edge_bins": 0,
            "bin_means_non_decreasing": False,
        }

    rho = spearmanr(
        frame["probability_edge"],
        frame["kelly_raw"],
        nan_policy="omit",
    ).statistic

    try:
        frame["edge_bin"] = pd.qcut(
            frame["probability_edge"],
            q=min(10, int(frame["probability_edge"].nunique())),
            duplicates="drop",
        )
    except ValueError:
        return {
            "spearman": (
                float(rho)
                if np.isfinite(rho)
                else float("nan")
            ),
            "edge_bins": 0,
            "bin_means_non_decreasing": False,
        }

    grouped = (
        frame.groupby(
            "edge_bin",
            observed=True,
        )
        .agg(
            mean_edge=("probability_edge", "mean"),
            mean_kelly_raw=("kelly_raw", "mean"),
            rows=("game_id", "size"),
        )
        .reset_index(drop=True)
        .sort_values("mean_edge")
    )

    if len(grouped) < 2:
        non_decreasing = False
    else:
        non_decreasing = bool(
            np.all(
                np.diff(
                    grouped["mean_kelly_raw"].to_numpy(dtype=float)
                )
                >= -PROB_TOLERANCE
            )
        )

    return {
        "spearman": (
            float(rho)
            if np.isfinite(rho)
            else float("nan")
        ),
        "edge_bins": int(len(grouped)),
        "bin_means_non_decreasing": non_decreasing,
    }


def run_line_preference_summary(
    values: pd.DataFrame,
) -> dict:
    frame = values[
        (values["system"] == "new_model")
        & (values["market"] == "run_line")
    ].copy()

    counts = {
        "-1.5": 0,
        "+1.5": 0,
        "ties": 0,
        "games": 0,
    }

    for _, group in frame.groupby("game_id"):
        if set(group["side"]) != {"home", "away"}:
            continue

        if len(group) != 2:
            continue

        counts["games"] += 1

        max_ev = group["ev"].max()
        winners = group[
            np.isclose(
                group["ev"],
                max_ev,
                atol=PROB_TOLERANCE,
                rtol=0.0,
            )
        ]

        if len(winners) != 1:
            counts["ties"] += 1
            continue

        line = float(winners.iloc[0]["line"])

        if math.isclose(
            line,
            -1.5,
            abs_tol=PROB_TOLERANCE,
        ):
            counts["-1.5"] += 1
        elif math.isclose(
            line,
            1.5,
            abs_tol=PROB_TOLERANCE,
        ):
            counts["+1.5"] += 1
        else:
            fail(
                "Unexpected run-line preference line after supported "
                f"market validation: {line}"
            )

    return counts


def _fmt_float(
    value,
    digits: int = 6,
) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"

    if not np.isfinite(numeric):
        return "NA"

    return f"{numeric:.{digits}f}"


def _yes_no(value: bool) -> str:
    return "YES" if bool(value) else "NO"


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]

    return "\n".join(
        [header_line, separator] + body
    )


def write_summary(
    *,
    path: Path,
    test_start: str,
    test_end: str,
    scored: pd.DataFrame,
    run_metrics: pd.DataFrame,
    calibrations: dict[str, pd.DataFrame],
    log_loss: pd.DataFrame,
    values: pd.DataFrame,
    promotion: dict,
) -> None:
    metric_lookup = {
        (row.system, row.side): row
        for row in run_metrics.itertuples(index=False)
    }

    dr_home = metric_lookup[("dratings", "home")]
    new_home = metric_lookup[("new_model", "home")]
    dr_away = metric_lookup[("dratings", "away")]
    new_away = metric_lookup[("new_model", "away")]

    home_mae_improved = new_home.mae < dr_home.mae
    away_mae_improved = new_away.mae < dr_away.mae

    calibration_stats: dict[str, dict] = {}
    for market_name, calibration in calibrations.items():
        ece = expected_calibration_error(
            calibration,
            "new_model",
        )
        monotonic = calibration_monotonicity(
            calibration,
            "new_model",
        )
        calibration_stats[market_name] = {
            "ece": ece,
            "calibrated": (
                np.isfinite(ece)
                and ece <= CALIBRATION_ECE_THRESHOLD
            ),
            **monotonic,
        }

    new_value = value_summary(
        values,
        "new_model",
    )
    dratings_value = value_summary(
        values,
        "dratings",
    )

    new_kelly = kelly_edge_monotonicity(
        values,
        "new_model",
    )

    rl_preference = run_line_preference_summary(
        values
    )

    positive_ev_overstated = (
        np.isfinite(
            new_value[
                "positive_ev_mean_predicted_ev"
            ]
        )
        and np.isfinite(
            new_value[
                "positive_ev_mean_realized_return"
            ]
        )
        and (
            new_value[
                "positive_ev_mean_realized_return"
            ]
            < new_value[
                "positive_ev_mean_predicted_ev"
            ]
        )
    )

    run_rows = []
    for row in run_metrics.itertuples(index=False):
        run_rows.append(
            [
                row.system,
                row.side,
                str(row.rows),
                _fmt_float(row.mae),
                _fmt_float(row.mean_poisson_deviance),
                _fmt_float(row.mean_predicted_runs),
                _fmt_float(row.mean_actual_runs),
            ]
        )

    calibration_rows = []
    for market_name in [
        "moneyline",
        "run_line",
        "total",
    ]:
        stats = calibration_stats[market_name]
        calibration_rows.append(
            [
                market_name,
                _fmt_float(stats["ece"]),
                _yes_no(stats["calibrated"]),
                _fmt_float(stats["spearman"]),
                _yes_no(
                    stats["exact_non_decreasing"]
                ),
                str(stats["bins"]),
            ]
        )

    log_rows = []
    for row in log_loss.itertuples(index=False):
        log_rows.append(
            [
                row.system,
                row.market,
                row.evaluation_side,
                str(row.rows),
                _fmt_float(row.log_loss),
            ]
        )

    resolved_rl_games = (
        rl_preference["-1.5"]
        + rl_preference["+1.5"]
    )

    if resolved_rl_games > 0:
        minus_pct = (
            rl_preference["-1.5"]
            / resolved_rl_games
        )
        plus_pct = (
            rl_preference["+1.5"]
            / resolved_rl_games
        )
    else:
        minus_pct = float("nan")
        plus_pct = float("nan")

    home_gate = promotion["comparison"]["home"]
    away_gate = promotion["comparison"]["away"]

    summary_lines = [
        "# MLB Run Model Comparison",
        "",
        f"- Generated: `{_now()}`",
        f"- Untouched chronological test period: `{test_start}` through `{test_end}`",
        f"- Test games: `{len(scored)}`",
        "- Model fitting/tuning performed by this evaluation script: `NO`",
        f"- Promotion status: `{promotion['status']}`",
        "",
        "## Production promotion gate",
        "",
        (
            "Candidate promotion requires mean Poisson deviance <= DRatings "
            "for BOTH run sides, plus log loss <= DRatings for moneyline, "
            "run line, and totals on comparable held-out games."
        ),
        "",
        markdown_table(
            [
                "Side",
                "DRatings baseline Poisson",
                "Candidate Poisson",
                "Candidate <= baseline",
            ],
            [
                [
                    "home",
                    _fmt_float(home_gate["baseline_value"]),
                    _fmt_float(home_gate["candidate_value"]),
                    _yes_no(home_gate["candidate_lte_baseline"]),
                ],
                [
                    "away",
                    _fmt_float(away_gate["baseline_value"]),
                    _fmt_float(away_gate["candidate_value"]),
                    _yes_no(away_gate["candidate_lte_baseline"]),
                ],
            ],
        ),
        "",
        (
            "- Production artifacts changed: "
            f"**{_yes_no(promotion['production_artifacts_changed'])}**."
        ),
        "",
        "## Market probability promotion gate",
        "",
        markdown_table(
            ["Market", "Rows", "DRatings log loss", "Candidate log loss", "Pass"],
            [
                [
                    market_name,
                    str(promotion["market_comparison"][market_name]["rows"]),
                    _fmt_float(promotion["market_comparison"][market_name]["baseline_value"]),
                    _fmt_float(promotion["market_comparison"][market_name]["candidate_value"]),
                    _yes_no(promotion["market_comparison"][market_name]["candidate_lte_baseline"]),
                ]
                for market_name in ("moneyline", "run_line", "total")
            ],
        ),
        "",
        "## Run prediction metrics",
        "",
        markdown_table(
            [
                "System",
                "Side",
                "Rows",
                "MAE",
                "Mean Poisson deviance",
                "Mean predicted runs",
                "Mean actual runs",
            ],
            run_rows,
        ),
        "",
        "### Run-prediction questions",
        "",
        (
            "- Does the new model improve home-run prediction error? "
            f"**{_yes_no(home_mae_improved)}** "
            f"(MAE `{_fmt_float(dr_home.mae)}` -> "
            f"`{_fmt_float(new_home.mae)}`; "
            f"Poisson deviance `{_fmt_float(dr_home.mean_poisson_deviance)}` -> "
            f"`{_fmt_float(new_home.mean_poisson_deviance)}`)."
        ),
        (
            "- Does the new model improve away-run prediction error? "
            f"**{_yes_no(away_mae_improved)}** "
            f"(MAE `{_fmt_float(dr_away.mae)}` -> "
            f"`{_fmt_float(new_away.mae)}`; "
            f"Poisson deviance `{_fmt_float(dr_away.mean_poisson_deviance)}` -> "
            f"`{_fmt_float(new_away.mean_poisson_deviance)}`)."
        ),
        "",
        "## Probability calibration",
        "",
        (
            "Calibration YES/NO uses weighted expected calibration error "
            f"(ECE) <= `{CALIBRATION_ECE_THRESHOLD:.2f}`. "
            "Totals use conditional win probability on resolved bets; pushes "
            "are excluded from the observed win-rate denominator."
        ),
        "",
        markdown_table(
            [
                "Market",
                "New-model ECE",
                "Calibrated",
                "Predicted-vs-observed Spearman",
                "Observed rate exactly non-decreasing",
                "Populated bins",
            ],
            calibration_rows,
        ),
        "",
        (
            "- Are predicted moneyline probabilities calibrated? "
            f"**{_yes_no(calibration_stats['moneyline']['calibrated'])}**."
        ),
        (
            "- Are predicted run-line probabilities calibrated? "
            f"**{_yes_no(calibration_stats['run_line']['calibrated'])}**."
        ),
        (
            "- Are predicted total probabilities calibrated? "
            f"**{_yes_no(calibration_stats['total']['calibrated'])}**."
        ),
        (
            "- Does increasing predicted probability correspond to increasing "
            "observed win rate? "
            f"Moneyline **{_yes_no(calibration_stats['moneyline']['exact_non_decreasing'])}**, "
            f"run line **{_yes_no(calibration_stats['run_line']['exact_non_decreasing'])}**, "
            f"total **{_yes_no(calibration_stats['total']['exact_non_decreasing'])}**. "
            "See Spearman values above for rank-direction strength."
        ),
        "",
        "## Probability log loss",
        "",
        markdown_table(
            [
                "System",
                "Market",
                "Evaluation side",
                "Rows",
                "Log loss",
            ],
            log_rows,
        ),
        "",
        "## EV, realized return, and Kelly",
        "",
        (
            f"- New-model priced candidates evaluated: `{new_value['rows']}`; "
            f"positive-EV candidates: `{new_value['positive_ev_rows']}`."
        ),
        (
            "- New-model all-candidate mean predicted EV vs realized return: "
            f"`{_fmt_float(new_value['mean_predicted_ev'])}` vs "
            f"`{_fmt_float(new_value['mean_realized_return'])}`."
        ),
        (
            "- New-model positive-EV mean predicted EV vs realized return: "
            f"`{_fmt_float(new_value['positive_ev_mean_predicted_ev'])}` vs "
            f"`{_fmt_float(new_value['positive_ev_mean_realized_return'])}`."
        ),
        (
            "- Does higher predicted EV correspond to higher realized return? "
            f"EV/return Spearman = `{_fmt_float(new_value['ev_return_spearman'])}`. "
            "A positive value indicates higher EV tended to correspond to "
            "higher realized return in this test sample."
        ),
        (
            "- Is positive EV overstated versus realized return? "
            f"**{_yes_no(positive_ev_overstated)}** "
            "(defined here as mean realized return below mean predicted EV "
            "among positive-EV candidates)."
        ),
        (
            "- DRatings-run baseline all-candidate mean predicted EV vs realized return: "
            f"`{_fmt_float(dratings_value['mean_predicted_ev'])}` vs "
            f"`{_fmt_float(dratings_value['mean_realized_return'])}`; "
            f"EV/return Spearman `{_fmt_float(dratings_value['ev_return_spearman'])}`."
        ),
        (
            "- Does Kelly increase monotonically with actual model edge? "
            f"Edge/Kelly-raw Spearman = `{_fmt_float(new_kelly['spearman'])}`; "
            f"mean raw Kelly across ordered edge bins is non-decreasing: "
            f"**{_yes_no(new_kelly['bin_means_non_decreasing'])}** "
            f"across `{new_kelly['edge_bins']}` populated edge bins."
        ),
        "",
        "## Run-line side preference",
        "",
        (
            f"- Games with both run-line sides priced/evaluated: "
            f"`{rl_preference['games']}`."
        ),
        (
            f"- Higher-EV side was `-1.5` in `{rl_preference['-1.5']}` games "
            f"(`{_fmt_float(minus_pct * 100, 2)}%` of non-ties)."
        ),
        (
            f"- Higher-EV side was `+1.5` in `{rl_preference['+1.5']}` games "
            f"(`{_fmt_float(plus_pct * 100, 2)}%` of non-ties)."
        ),
        (
            f"- Exact EV ties: `{rl_preference['ties']}`."
        ),
        "",
        "## Interpretation constraint",
        "",
        (
            "This report evaluates the candidate on the untouched test "
            "period only. The script does not refit, retune, or select "
            "hyperparameters from these results. Do not tune the model on "
            "this final test period after reviewing the report."
        ),
        "",
    ]

    path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-data",
        type=Path,
        default=DEFAULT_TRAINING_DATA,
        help=(
            "Training-set CSV used by train_run_model.py "
            f"(default: {DEFAULT_TRAINING_DATA})"
        ),
    )
    parser.add_argument(
        "--candidate-dir",
        "--model-dir",
        dest="candidate_dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help=(
            "Directory containing candidate run-model artifacts/metadata. "
            "--model-dir is retained as a compatibility alias. "
            f"(default: {DEFAULT_CANDIDATE_DIR})"
        ),
    )
    parser.add_argument(
        "--production-model-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_MODEL_DIR,
        help=(
            "Production run-model directory used only after the promotion gate "
            f"passes (default: {DEFAULT_PRODUCTION_MODEL_DIR})"
        ),
    )
    parser.add_argument(
        "--sportsbook-dir",
        type=Path,
        default=DEFAULT_SPORTSBOOK_DIR,
        help=(
            "Historical sportsbook snapshot directory "
            f"(default: {DEFAULT_SPORTSBOOK_DIR})"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=(
            "Output report directory "
            f"(default: {DEFAULT_REPORT_DIR})"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _ensure_dirs(args.report_dir)

    with LOG_FILE.open("w", encoding="utf-8") as f:
        f.write(f"=== evaluate_run_model RUN {_now()} ===\n")

    try:
        probs_module, evk_module = _load_production_math()

        home_metadata = load_json(
            args.candidate_dir / HOME_METADATA_NAME,
            "home candidate metadata",
        )
        away_metadata = load_json(
            args.candidate_dir / AWAY_METADATA_NAME,
            "away candidate metadata",
        )

        (
            feature_columns,
            test_start,
            test_end,
            expected_test_rows,
        ) = validate_metadata(
            home_metadata,
            away_metadata,
        )

        test = load_test_period(
            args.training_data,
            feature_columns,
            test_start,
            test_end,
            expected_test_rows,
        )

        scored = score_models(
            test,
            feature_columns,
            args.candidate_dir,
        )

        run_metrics = build_run_metrics(
            scored
        )

        sportsbook = load_sportsbook_test_period(
            scored,
            args.sportsbook_dir,
        )

        market = join_sportsbook(
            scored,
            sportsbook,
        )

        market = derive_market_probabilities(
            market,
            probs_module,
        )

        market = add_observed_outcomes(
            market
        )

        calibrations = build_calibration_reports(
            market
        )

        log_loss = build_probability_log_loss(
            market
        )

        promotion_decision = build_promotion_decision(
            run_metrics,
            log_loss,
            test_start=test_start,
            test_end=test_end,
        )

        values = build_value_records(
            market,
            evk_module,
        )

        run_metrics.to_csv(
            args.report_dir / REPORT_FILES["run_metrics"],
            index=False,
        )

        calibrations["moneyline"].to_csv(
            args.report_dir
            / REPORT_FILES["moneyline_calibration"],
            index=False,
        )

        calibrations["run_line"].to_csv(
            args.report_dir
            / REPORT_FILES["run_line_calibration"],
            index=False,
        )

        calibrations["total"].to_csv(
            args.report_dir
            / REPORT_FILES["total_calibration"],
            index=False,
        )

        log_loss.to_csv(
            args.report_dir / REPORT_FILES["log_loss"],
            index=False,
        )

        promotion = apply_promotion_decision(
            decision=promotion_decision,
            candidate_dir=args.candidate_dir,
            production_model_dir=args.production_model_dir,
            home_candidate_metadata=home_metadata,
            away_candidate_metadata=away_metadata,
            report_path=(
                args.report_dir
                / REPORT_FILES["promotion"]
            ),
        )

        write_summary(
            path=(
                args.report_dir
                / REPORT_FILES["summary"]
            ),
            test_start=test_start,
            test_end=test_end,
            scored=scored,
            run_metrics=run_metrics,
            calibrations=calibrations,
            log_loss=log_loss,
            values=values,
            promotion=promotion,
        )

        written = [
            args.report_dir / name
            for name in REPORT_FILES.values()
        ]

        missing_outputs = [
            str(path)
            for path in written
            if not path.exists()
        ]

        if missing_outputs:
            fail(
                "Evaluation finished without all required report files: "
                f"{missing_outputs}"
            )

        _log(
            "SUCCESS "
            f"promotion_status={promotion['status']} "
            f"test_rows={len(scored)} "
            f"test_start={test_start} "
            f"test_end={test_end} "
            f"value_rows={len(values)} "
            f"reports={len(written)}"
        )

        print(
            "MLB run-model comparison complete: "
            f"{args.report_dir} | "
            f"promotion_status={promotion['status']}"
        )

    except Exception:
        _log(traceback.format_exc(), "ERROR")
        raise


if __name__ == "__main__":
    main()
