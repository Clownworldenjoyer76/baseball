#!/usr/bin/env python3
"""Build MLB run projections with leakage-safe walk-forward history.

Inputs per date:
    docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/{date}_MLB.csv
    docs/win/baseball/mlb/00_intake/games/{date}_games.csv
    docs/win/baseball/mlb/00_intake/sportsdataverse/{date}_sportsdataverse.csv
    docs/win/baseball/mlb/00_intake/mlb_raw/{date}_game_context.csv

Production models:
    docs/win/baseball/mlb/models/run_projection/home_runs_model.joblib
    docs/win/baseball/mlb/models/run_projection/away_runs_model.joblib
    docs/win/baseball/mlb/models/run_projection/home_runs_model_metadata.json
    docs/win/baseball/mlb/models/run_projection/away_runs_model_metadata.json

Historical training rows are rebuilt in memory from the same leakage-safe
training-data builder used by the model-training workflow. Historical target
dates are projected with walk-forward models trained only on games strictly
before the target date. Dates later than the latest finalized training-history
date use the committed production models.

Output:
    docs/win/baseball/mlb/00_intake/predictions/model_projection/{date}_MLB.csv

When no dates are supplied, every *_MLB.csv file in pred_with_game_id is rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


BASE_DIR = Path("docs/win/baseball/mlb")

PRED_DIR = BASE_DIR / "00_intake/predictions/pred_with_game_id"
GAMES_DIR = BASE_DIR / "00_intake/games"
SDV_DIR = BASE_DIR / "00_intake/sportsdataverse"
CONTEXT_DIR = BASE_DIR / "00_intake/mlb_raw"

MODEL_DIR = BASE_DIR / "models/run_projection"
HOME_MODEL_FILE = MODEL_DIR / "home_runs_model.joblib"
AWAY_MODEL_FILE = MODEL_DIR / "away_runs_model.joblib"
HOME_METADATA_FILE = MODEL_DIR / "home_runs_model_metadata.json"
AWAY_METADATA_FILE = MODEL_DIR / "away_runs_model_metadata.json"

TRAINING_BUILDER_FILE = BASE_DIR / "scripts/modeling/build_run_training_set.py"
TRAINER_FILE = BASE_DIR / "scripts/modeling/train_run_model.py"

OUTPUT_DIR = BASE_DIR / "00_intake/predictions/model_projection"
ERROR_DIR = BASE_DIR / "errors/00_intake"
LOG_FILE = ERROR_DIR / "build_run_projection.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


PRED_REQUIRED = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "home_projected_runs",
    "away_projected_runs",
    "total_projected_runs",
]

GAMES_REQUIRED = [
    "game_id",
    "gamePk",
    "game_date",
    "home_team",
    "away_team",
]

SDV_REQUIRED = [
    "gamePk",
    "game_id",
    "sdv_as_of_date",
    "sdv_status",
    "sdv_home_sp_found",
    "sdv_away_sp_found",
    "sdv_home_bp_pa_14d",
    "sdv_away_bp_pa_14d",
    "sdv_home_bp_woba_allowed_14d",
    "sdv_away_bp_woba_allowed_14d",
    "sdv_home_bp_k_rate_14d",
    "sdv_away_bp_k_rate_14d",
    "sdv_home_bp_bb_rate_14d",
    "sdv_away_bp_bb_rate_14d",
    "sdv_home_bp_hard_rate_14d",
    "sdv_away_bp_hard_rate_14d",
    "sdv_home_bp_pitches_3d",
    "sdv_away_bp_pitches_3d",
    "sdv_home_bp_pa_7d",
    "sdv_away_bp_pa_7d",
    "sdv_home_bp_woba_allowed_7d",
    "sdv_away_bp_woba_allowed_7d",
    "sdv_home_bp_k_rate_7d",
    "sdv_away_bp_k_rate_7d",
    "sdv_home_bp_bb_rate_7d",
    "sdv_away_bp_bb_rate_7d",
    "sdv_home_bp_hard_rate_7d",
    "sdv_away_bp_hard_rate_7d",
]


_SDV_SUFFIX_MAP = {
    "sp_stuff_plus": "sp_pitch_quality_plus",
    "sp_command_plus": "sp_command_plus",
    "sp_xera": "sp_xera",
    "sp_xera_30d": "sp_xera_30d",
    "sp_xwoba": "sp_xwoba",
    "sp_xwoba_30d": "sp_xwoba_30d",
    "sp_avg_velo": "sp_avg_velo",
    "sp_avg_velo_30d": "sp_avg_velo_30d",
    "sp_velo_delta_30d": "sp_velo_delta_30d",
    "sp_pitches": "sp_pitches",
    "sp_games": "sp_games",
    "sp_pitches_30d": "sp_pitches_30d",
    "sp_games_30d": "sp_games_30d",
    "bp_pa_14d": "bp_pa_14d",
    "bp_woba_allowed_14d": "bp_woba_allowed_14d",
    "bp_k_rate_14d": "bp_k_rate_14d",
    "bp_bb_rate_14d": "bp_bb_rate_14d",
    "bp_hard_rate_14d": "bp_hard_rate_14d",
    "bp_pitches_3d": "bp_pitches_3d",
    "bp_pa_7d": "bp_pa_7d",
    "bp_woba_allowed_7d": "bp_woba_allowed_7d",
    "bp_k_rate_7d": "bp_k_rate_7d",
    "bp_bb_rate_7d": "bp_bb_rate_7d",
    "bp_hard_rate_7d": "bp_hard_rate_7d",
}

SDV_FEATURE_MAP = {
    f"sdv_{side}_{source_suffix}": f"{side}_{target_suffix}"
    for side in ("home", "away")
    for source_suffix, target_suffix in _SDV_SUFFIX_MAP.items()
}


DRATINGS_RENAME = {
    "home_prob": "dratings_home_prob",
    "away_prob": "dratings_away_prob",
    "home_projected_runs": "dratings_home_projected_runs",
    "away_projected_runs": "dratings_away_projected_runs",
    "total_projected_runs": "dratings_total_projected_runs",
}


SAFE_CONTEXT_FEATURES = {
    "temp_f",
    "wind_mph",
    "wind_blowing_out",
    "humidity",
    "air_pressure_at_sea_level",
    "dew_point_f",
    "weather_applicable",
    "venue_id",
    "day_night",
}

TARGET_COLUMNS = [
    "target_home_runs",
    "target_away_runs",
]

MIN_PRIOR_UNIQUE_DATES = 3
NY = ZoneInfo("America/New_York")


class _ProjectionLog:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def timestamp() -> str:
        return datetime.now(UTC).isoformat()

    def write(self, message: str, level: str = "INFO") -> None:
        timestamp = self.timestamp()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{timestamp} | {level:<5} | {message.rstrip()}\n"
            )

    def abort(self, message: str) -> None:
        self.write(message, "ERROR")
        raise RuntimeError(message)


_RUN_LOG = _ProjectionLog(LOG_FILE)
_now = _RUN_LOG.timestamp
_log = _RUN_LOG.write
fail = _RUN_LOG.abort


def _row_issue(date_str: str, message: str) -> None:
    _log(
        f"ROW ISSUE SKIPPED date={date_str} | {message}",
        "WARN",
    )


def duplicate_columns(columns) -> list[str]:
    seen = set()
    dupes = []

    for col in columns:
        if col in seen and col not in dupes:
            dupes.append(col)

        seen.add(col)

    return dupes


def _validate_required_frame(
    frame: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    dupes = duplicate_columns(list(frame.columns))
    if dupes:
        fail(f"{label} duplicate columns: {dupes}")

    missing = sorted(set(required).difference(frame.columns))
    if missing:
        fail(f"{label} missing required columns: {missing}")


def read_csv_checked(
    path: Path,
    required: list[str],
    label: str,
) -> pd.DataFrame:
    if not path.is_file():
        fail(f"{label} missing required file: {path}")

    frame = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
    )
    _validate_required_frame(frame, required, label)
    return frame


def normalize_game_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_gamepk(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()

    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    out = raw.copy()
    valid = numeric.notna()

    out.loc[valid] = (
        numeric.loc[valid]
        .round()
        .astype("Int64")
        .astype("string")
    )

    return out


def assert_unique_nonblank_key(
    df: pd.DataFrame,
    key: str,
    label: str,
) -> None:
    if key not in df.columns:
        return

    values = df[key].astype("string").str.strip()

    nonblank = values.notna() & (values != "")

    duplicated = nonblank & values.duplicated(keep=False)

    if duplicated.any():
        sample = values.loc[duplicated].head(10).tolist()

        fail(
            f"{label} duplicate {key}; "
            f"duplicate_rows={int(duplicated.sum())}; "
            f"sample={sample}"
        )


def prepare_keys(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    df = df.copy()

    if "game_id" in df.columns:
        df["game_id"] = normalize_game_id(df["game_id"])

        assert_unique_nonblank_key(
            df,
            "game_id",
            label,
        )

    if "gamePk" in df.columns:
        df["gamePk"] = normalize_gamepk(df["gamePk"])

        assert_unique_nonblank_key(
            df,
            "gamePk",
            label,
        )

    return df


def load_metadata(
    path: Path,
    label: str,
) -> dict:
    if not path.exists():
        fail(f"{label} metadata missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata.get("feature_columns")

    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or any(
            not isinstance(col, str) or not col
            for col in feature_columns
        )
    ):
        fail(f"{label} metadata has invalid feature_columns")

    if len(feature_columns) != len(set(feature_columns)):
        fail(
            f"{label} metadata feature_columns "
            "contain duplicates"
        )

    return metadata


def load_model(
    path: Path,
    label: str,
):
    if not path.exists():
        fail(f"{label} model missing: {path}")

    return joblib.load(path)


def model_version(
    metadata: dict,
    label: str,
) -> str:
    created_at = str(
        metadata.get("created_at") or ""
    ).strip()

    if created_at:
        return f"{label}:{created_at}"

    return f"{label}:unknown"


def assert_metadata_feature_contract(
    home_metadata: dict,
    away_metadata: dict,
) -> list[str]:
    home_features = list(home_metadata["feature_columns"])
    away_features = list(away_metadata["feature_columns"])

    if home_features != away_features:
        fail(
            "Home and away model metadata "
            "feature order differs; "
            f"home={home_features} "
            f"away={away_features}"
        )

    return home_features


def assert_model_feature_order(
    model,
    metadata_features: list[str],
    label: str,
) -> None:
    model_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_names is None:
        fail(
            f"{label} model does not expose "
            "feature_names_in_; cannot verify "
            "exact feature order against metadata"
        )

    model_features = [
        str(x)
        for x in model_names.tolist()
    ]

    if model_features != metadata_features:
        fail(
            f"{label} model feature order "
            "differs from metadata; "
            f"model={model_features} "
            f"metadata={metadata_features}"
        )


def assert_secondary_game_id_match(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    label: str,
) -> None:
    left = df[left_col].astype("string").str.strip()
    right = df[right_col].astype("string").str.strip()

    comparable = (
        left.notna()
        & right.notna()
        & (left != "")
        & (right != "")
    )

    mismatch = comparable & (left != right)

    if mismatch.any():
        sample = (
            df.loc[
                mismatch,
                [
                    left_col,
                    right_col,
                    "gamePk",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        fail(
            f"{label} game_id mismatch; "
            f"bad_rows={int(mismatch.sum())}; "
            f"sample={sample}"
        )


def _load_python_module(
    path: Path,
    module_name: str,
):
    module_dir = str(path.parent)
    added_to_path = module_dir not in sys.path

    if added_to_path:
        sys.path.insert(0, module_dir)

    try:
        from module_loader import load_module_from_path

        return load_module_from_path(
            module_name,
            path,
            fail=fail,
            missing_message=(
                "Required modeling script missing: {path}"
            ),
            invalid_spec_message=(
                "Unable to load modeling script: {path}"
            ),
        )
    finally:
        if added_to_path:
            sys.path.remove(module_dir)


def _filter_projection_prediction_ids(date_str, pred):
    pred = pred.copy()
    pred["game_id"] = normalize_game_id(pred["game_id"])
    blank = pred["game_id"].isna() | (pred["game_id"] == "")
    for row in pred.loc[
        blank, ["game_id", "game_date", "home_team", "away_team"]
    ].head(20).to_dict("records"):
        _row_issue(date_str, f"blank prediction game_id row={row}")
    return pred.loc[~blank].copy()


def _merge_projection_games(date_str, pred, games):
    base = pred.rename(columns=DRATINGS_RENAME).copy()
    for source_col in DRATINGS_RENAME:
        base[source_col] = pred[source_col]
    keep = [
        col for col in [
            "game_id", "gamePk", "game_date", "home_team", "away_team",
            "venue_id", "day_night",
        ] if col in games.columns
    ]
    joined = base.merge(
        games[keep], on="game_id", how="left", suffixes=("", "_games"),
        validate="one_to_one",
    )
    missing = joined["gamePk"].isna()
    for row in joined.loc[
        missing, ["game_id", "game_date", "home_team", "away_team"]
    ].head(20).to_dict("records"):
        _row_issue(date_str, f"prediction missing games.gamePk row={row}")
    return joined.loc[~missing].copy()


def _merge_projection_sdv(joined, sdv):
    keep = [
        col for col in ([
            "gamePk", "game_id", "sdv_as_of_date", "sdv_status",
            "sdv_home_sp_found", "sdv_away_sp_found",
        ] + list(SDV_FEATURE_MAP.keys()))
        if col in sdv.columns
    ]
    joined = joined.merge(
        sdv[keep].rename(columns={"game_id": "game_id_sdv"}),
        on="gamePk", how="left", validate="one_to_one",
    )
    assert_secondary_game_id_match(
        joined, "game_id", "game_id_sdv", "games->sportsdataverse"
    )
    return joined.rename(columns=SDV_FEATURE_MAP)


def _merge_projection_context(joined, context, feature_columns):
    needed = [
        col for col in feature_columns
        if col in SAFE_CONTEXT_FEATURES and col in context.columns
    ]
    if not needed:
        return joined
    return joined.merge(
        context[["gamePk", *needed]], on="gamePk", how="left",
        validate="one_to_one", suffixes=("", "_context"),
    )


def _coerce_projection_features(date_str, joined, feature_columns):
    x_features = joined.loc[:, feature_columns].copy()
    if list(x_features.columns) != feature_columns:
        fail(
            "Constructed model feature order differs from metadata; "
            f"constructed={list(x_features.columns)} metadata={feature_columns}"
        )
    for col in x_features.columns:
        raw = x_features[col].copy()
        numeric = pd.to_numeric(raw, errors="coerce")
        raw_text = raw.astype("string").str.strip()
        bad = (
            (raw_text.notna() & (raw_text != "") & numeric.isna())
            | (numeric.notna() & ~np.isfinite(numeric))
        )
        for idx in x_features.index[bad][:20]:
            _row_issue(
                date_str,
                f"feature={col} invalid value={raw.loc[idx]!r} "
                f"game_id={joined.loc[idx, 'game_id']}",
            )
        numeric.loc[bad] = np.nan
        x_features[col] = numeric
    return x_features












def build_feature_frame(
    date_str: str,
    pred: pd.DataFrame,
    games: pd.DataFrame,
    sdv: pd.DataFrame,
    context: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = _filter_projection_prediction_ids(date_str, pred)
    if pred.empty:
        return pred.copy(), pd.DataFrame(columns=feature_columns)
    pred = prepare_keys(pred, "predictions")
    games = prepare_keys(games, "games")
    sdv = prepare_keys(sdv, "sportsdataverse")
    context = prepare_keys(context, "game_context")
    joined = _merge_projection_games(date_str, pred, games)
    if joined.empty:
        return joined, pd.DataFrame(columns=feature_columns)
    joined = _merge_projection_sdv(joined, sdv)
    joined = _merge_projection_context(joined, context, feature_columns)
    missing = [col for col in feature_columns if col not in joined.columns]
    if missing:
        fail(
            "Required model feature columns unavailable for production projection: "
            f"{missing}"
        )
    return joined, _coerce_projection_features(date_str, joined, feature_columns)

def build_feature_status(
    joined: pd.DataFrame,
    x_features: pd.DataFrame,
) -> pd.Series:
    statuses = []

    home_found = pd.to_numeric(
        joined.get("sdv_home_sp_found"),
        errors="coerce",
    )

    away_found = pd.to_numeric(
        joined.get("sdv_away_sp_found"),
        errors="coerce",
    )

    for idx in joined.index:
        missing_features = [
            col
            for col in x_features.columns
            if pd.isna(x_features.loc[idx, col])
        ]

        home_ok = (
            idx in home_found.index
            and pd.notna(home_found.loc[idx])
            and float(home_found.loc[idx]) == 1.0
        )

        away_ok = (
            idx in away_found.index
            and pd.notna(away_found.loc[idx])
            and float(away_found.loc[idx]) == 1.0
        )

        if home_ok and away_ok:
            starter_status = "both_starters_sdv_found"
        elif home_ok and not away_ok:
            starter_status = "away_starter_sdv_missing"
        elif away_ok and not home_ok:
            starter_status = "home_starter_sdv_missing"
        else:
            starter_status = "both_starters_sdv_missing"

        if missing_features:
            status = (
                f"{starter_status};"
                "missing_features="
                + ",".join(missing_features)
            )
        else:
            status = (
                f"{starter_status};"
                "missing_features=none"
            )

        statuses.append(status)

    return pd.Series(
        statuses,
        index=joined.index,
        dtype="string",
    )


def validate_predictions(
    values,
    label: str,
    date_str: str,
    joined: pd.DataFrame,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    bad = (
        ~np.isfinite(values)
        | (values < 0)
    )

    if bad.any():
        bad_positions = np.where(bad)[0][:20]

        for pos in bad_positions:
            if pos < len(joined):
                game_id = joined.iloc[pos]["game_id"]
            else:
                game_id = "unknown"

            _row_issue(
                date_str,
                f"{label} invalid "
                f"prediction={values[pos]!r} "
                f"game_id={game_id}",
            )

        values = values.copy()
        values[bad] = np.nan

    return values


def _training_summary_template() -> dict:
    return {
        "rows_loaded": 0,
        "rows_joined": 0,
        "missing_sdv": 0,
        "missing_final_score_file": 0,
        "unresolved_gamePk": 0,
        "failed_final_score_join": 0,
        "duplicate_game_id": 0,
        "duplicate_gamePk": 0,
        "leakage_rejections": 0,
        "rows_written": 0,
    }


def _drop_invalid_training_dates(training):
    training["_game_date_dt"] = pd.to_datetime(
        training["game_date"].astype("string").str.replace("_", "-", regex=False),
        errors="coerce",
    ).dt.normalize()
    bad = training["_game_date_dt"].isna()
    for row in training.loc[bad, ["game_id", "game_date"]].head(20).to_dict("records"):
        _log(f"TRAINING ROW SKIPPED invalid game_date row={row}", "WARN")
    return training.loc[~bad].copy()


def _coerce_training_history_column(training, col):
    raw = training[col].copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    if col in TARGET_COLUMNS:
        bad = numeric.isna() | ~np.isfinite(numeric) | (numeric < 0)
        for row in training.loc[bad, ["game_id", "game_date", col]].head(20).to_dict("records"):
            _log(f"TRAINING ROW SKIPPED invalid target={col} row={row}", "WARN")
        training = training.loc[~bad].copy()
        training[col] = pd.to_numeric(training[col], errors="coerce")
        return training
    bad = numeric.notna() & ~np.isfinite(numeric)
    for row in training.loc[bad, ["game_id", "game_date", col]].head(20).to_dict("records"):
        _log(f"TRAINING FEATURE SET TO MISSING feature={col} row={row}", "WARN")
    numeric.loc[bad] = np.nan
    training[col] = numeric
    return training






def build_training_history_in_memory(training_builder, feature_columns: list[str]) -> pd.DataFrame:
    summary = _training_summary_template()
    dates = training_builder.discover_dates(summary)
    if not dates:
        fail("No finalized historical dates available for walk-forward training")
    frames = []
    _log(
        "Building leakage-safe walk-forward training history in memory; "
        f"dates={len(dates)} first={dates[0]} last={dates[-1]}"
    )
    for date_str in dates:
        frame = training_builder.build_date_training_rows(date_str, summary)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        fail("Walk-forward training history is empty")
    training = pd.concat(frames, ignore_index=True, sort=False)
    training_builder.validate_final_output(training)
    missing = [
        col for col in feature_columns + TARGET_COLUMNS
        if col not in training.columns
    ]
    if missing:
        fail(
            "Walk-forward training history missing production model columns: "
            f"{missing}"
        )
    training = _drop_invalid_training_dates(training)
    for col in feature_columns + TARGET_COLUMNS:
        training = _coerce_training_history_column(training, col)
    if training.empty:
        fail("Walk-forward training history has no valid rows")
    training = training.sort_values(
        ["_game_date_dt", "game_id"], kind="stable"
    ).reset_index(drop=True)
    _log(
        "Walk-forward training history ready; "
        f"rows={len(training)} first={training['_game_date_dt'].min().date()} "
        f"last={training['_game_date_dt'].max().date()}"
    )
    return training

def _fit_full_prior_model(
    prior: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict,
    random_state: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        random_state=random_state,
    )

    model.fit(
        prior.loc[:, feature_columns],
        prior[target_column],
    )

    return model


def fit_walk_forward_models(
    target_date: pd.Timestamp,
    training_history: pd.DataFrame,
    feature_columns: list[str],
    trainer,
):
    prior = training_history.loc[
        training_history["_game_date_dt"] < target_date
    ].copy()

    prior_dates = sorted(
        prior["_game_date_dt"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if len(prior_dates) < MIN_PRIOR_UNIQUE_DATES:
        return None

    splits = trainer.chronological_date_split(prior)

    (
        home_params,
        home_validation_score,
    ) = trainer.select_hyperparameters(
        splits["train"],
        splits["validation"],
        feature_columns,
        "target_home_runs",
        f"walk_forward_home_{target_date.date()}",
    )

    (
        away_params,
        away_validation_score,
    ) = trainer.select_hyperparameters(
        splits["train"],
        splits["validation"],
        feature_columns,
        "target_away_runs",
        f"walk_forward_away_{target_date.date()}",
    )

    # Keep the last earlier-date partition OUT of hyperparameter tuning.
    # The learned model must beat the raw DRatings run projection on this
    # untouched, pre-target-date holdout before being used for that side.
    gate_training = pd.concat(
        [splits["train"], splits["validation"]],
        ignore_index=True,
        sort=False,
    )
    gate_test = splits["test"]
    selection = {}
    models = {}

    for side, params in (("home", home_params), ("away", away_params)):
        target_col = f"target_{side}_runs"
        baseline_col = f"dratings_{side}_projected_runs"
        if baseline_col not in gate_test.columns:
            fail(f"Walk-forward comparison missing {baseline_col}")

        baseline = pd.to_numeric(
            gate_test[baseline_col], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.all(np.isfinite(baseline) & (baseline > 0)):
            fail(
                "Walk-forward DRatings baseline must be finite and positive "
                f"on holdout: side={side} target={target_date.date()}"
            )

        gate_model = _fit_full_prior_model(
            gate_training,
            feature_columns,
            target_col,
            params,
            trainer.RANDOM_STATE,
        )
        actual = gate_test[target_col].to_numpy(dtype=float)
        candidate_score = trainer.safe_mean_poisson_deviance(
            actual,
            gate_model.predict(gate_test.loc[:, feature_columns]),
        )
        baseline_score = trainer.safe_mean_poisson_deviance(actual, baseline)
        if not np.isfinite(candidate_score) or not np.isfinite(baseline_score):
            fail(f"Non-finite walk-forward gate metric for {side}")

        source = (
            "learned" if candidate_score <= baseline_score else "dratings"
        )
        selection[side] = {
            "source": source,
            "candidate_holdout_deviance": float(candidate_score),
            "baseline_holdout_deviance": float(baseline_score),
            "holdout_rows": len(gate_test),
            "holdout_first_date": str(gate_test["_game_date_dt"].min().date()),
            "holdout_last_date": str(gate_test["_game_date_dt"].max().date()),
        }
        if source == "learned":
            models[side] = _fit_full_prior_model(
                prior,
                feature_columns,
                target_col,
                params,
                trainer.RANDOM_STATE,
            )
            assert_model_feature_order(
                models[side],
                feature_columns,
                f"walk_forward_{side}_{target_date.date()}",
            )
        else:
            # The caller reads the existing DRatings column for this side;
            # no fitted learned model is needed.
            models[side] = None

    train_end = (
        prior["_game_date_dt"]
        .max()
        .date()
        .isoformat()
    )

    version = (
        "walk_forward"
        f";target={target_date.date().isoformat()}"
        f";train_end={train_end}"
        f";rows={len(prior)}"
        ";home_params="
        + json.dumps(
            home_params,
            sort_keys=True,
            separators=(",", ":"),
        )
        + ";away_params="
        + json.dumps(
            away_params,
            sort_keys=True,
            separators=(",", ":"),
        )
        + f";home_source={selection['home']['source']}"
        + f";away_source={selection['away']['source']}"
    )

    audit = {
        "projection_mode": "walk_forward",
        "training_rows": len(prior),
        "training_unique_dates": len(prior_dates),
        "training_end_date": train_end,
        "home_validation_score": float(
            home_validation_score
        ),
        "away_validation_score": float(
            away_validation_score
        ),
        "home_selection": selection["home"],
        "away_selection": selection["away"],
        "version": version,
    }

    return (
        models["home"],
        models["away"],
        audit,
    )


def _current_pipeline_date() -> str:
    env_date = str(os.environ.get("DATE", "") or "").strip().replace("-", "_")

    if env_date:
        parsed = pd.to_datetime(
            env_date.replace("_", "-"),
            errors="coerce",
        )

        if pd.notna(parsed):
            return pd.Timestamp(parsed).strftime("%Y_%m_%d")

    return datetime.now(NY).strftime("%Y_%m_%d")


def _assert_frame_date_contract(
    date_str: str,
    df: pd.DataFrame,
    label: str,
) -> None:
    if "game_date" not in df.columns or df.empty:
        return

    expected = _target_timestamp(date_str)
    parsed = pd.to_datetime(
        df["game_date"].astype("string").str.replace("_", "-", regex=False),
        errors="coerce",
    ).dt.normalize()

    bad = parsed.isna() | (parsed != expected)

    if bad.any():
        sample_columns = [
            col
            for col in ["game_id", "gamePk", "game_date", "home_team", "away_team"]
            if col in df.columns
        ]
        sample = df.loc[bad, sample_columns].head(20).to_dict("records")
        fail(
            f"{label} contains rows outside projection date {date_str}; "
            f"bad_rows={int(bad.sum())}; sample={sample}"
        )


def _validated_game_ids(
    frame: pd.DataFrame,
    label: str,
    role: str,
) -> pd.Series:
    if "game_id" not in frame.columns:
        fail(f"{label} {role} missing game_id column")

    game_ids = normalize_game_id(frame["game_id"])
    blank = game_ids.isna() | (game_ids == "")

    if blank.any():
        sample = frame.loc[blank].head(20).to_dict("records")
        fail(
            f"{label} current-day {role} contains blank game_id; "
            f"bad_rows={int(blank.sum())}; sample={sample}"
        )

    dupes = game_ids.duplicated(keep=False)
    if dupes.any():
        sample = game_ids.loc[dupes].head(20).tolist()
        fail(
            f"{label} current-day {role} contains duplicate game_id; "
            f"duplicate_rows={int(dupes.sum())}; sample={sample}"
        )

    return game_ids


def _assert_current_day_source_coverage(
    date_str: str,
    source: pd.DataFrame,
    result: pd.DataFrame,
    label: str,
) -> None:
    if date_str != _current_pipeline_date():
        return

    source_ids = _validated_game_ids(source, label, "source")
    result_ids = _validated_game_ids(result, label, "result")

    source_set = set(source_ids.astype(str))
    result_set = set(result_ids.astype(str))
    missing = sorted(source_set - result_set)
    extra = sorted(result_set - source_set)

    if missing or extra:
        fail(
            f"{label} current-day source/result game_id coverage mismatch; "
            f"missing_from_result={missing}; extra_in_result={extra}"
        )


def _target_timestamp(
    date_str: str,
) -> pd.Timestamp:
    parsed = pd.to_datetime(
        str(date_str).replace("_", "-"),
        errors="coerce",
    )

    if pd.isna(parsed):
        fail(f"Invalid projection date: {date_str}")

    return pd.Timestamp(parsed).normalize()


def _write_unavailable_historical_result(
    date_str: str,
    output_path: Path,
    joined: pd.DataFrame,
    x_features: pd.DataFrame,
    reason: str,
) -> Path:
    result = joined.copy()

    result["model_home_runs"] = np.nan
    result["model_away_runs"] = np.nan
    result["model_total_runs"] = np.nan

    result["run_model_version"] = (
        "walk_forward_unavailable"
        f";date={date_str}"
        f";reason={reason}"
    )

    status = build_feature_status(
        result,
        x_features,
    )

    result["run_model_feature_status"] = (
        status.astype("string")
        + ";projection_mode=walk_forward_unavailable"
        + f";reason={reason}"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    _log(
        f"WROTE {output_path} "
        f"rows={len(result)} "
        "projection_mode="
        "walk_forward_unavailable "
        f"reason={reason}",
        "WARN",
    )

    return output_path


def process_date(
    date_str: str,
    production_home_model,
    production_away_model,
    home_metadata: dict,
    away_metadata: dict,
    feature_columns: list[str],
    training_history: pd.DataFrame,
    trainer,
) -> Path:
    pred_path = PRED_DIR / f"{date_str}_MLB.csv"
    games_path = GAMES_DIR / f"{date_str}_games.csv"
    sdv_path = SDV_DIR / f"{date_str}_sportsdataverse.csv"
    context_path = CONTEXT_DIR / f"{date_str}_game_context.csv"
    output_path = OUTPUT_DIR / f"{date_str}_MLB.csv"

    pred = read_csv_checked(
        pred_path,
        PRED_REQUIRED,
        f"predictions {date_str}",
    )

    games = read_csv_checked(
        games_path,
        GAMES_REQUIRED,
        f"games {date_str}",
    )

    sdv = read_csv_checked(
        sdv_path,
        SDV_REQUIRED,
        f"sportsdataverse {date_str}",
    )

    context = read_csv_checked(
        context_path,
        ["gamePk"],
        f"game_context {date_str}",
    )

    _assert_frame_date_contract(
        date_str,
        pred,
        f"predictions {date_str}",
    )
    _assert_frame_date_contract(
        date_str,
        games,
        f"games {date_str}",
    )

    joined, x_features = build_feature_frame(
        date_str,
        pred,
        games,
        sdv,
        context,
        feature_columns,
    )

    if output_path.resolve() == pred_path.resolve():
        fail(
            "Refusing to overwrite "
            "pred_with_game_id source file"
        )

    if joined.empty:
        result = joined.copy()

        result["model_home_runs"] = pd.Series(dtype=float)
        result["model_away_runs"] = pd.Series(dtype=float)
        result["model_total_runs"] = pd.Series(dtype=float)
        result["run_model_version"] = pd.Series(dtype="string")
        result["run_model_feature_status"] = pd.Series(
            dtype="string"
        )

        _assert_current_day_source_coverage(
            date_str,
            pred,
            result,
            "build_run_projection",
        )

        result.to_csv(
            output_path,
            index=False,
        )

        _log(
            f"WROTE {output_path} "
            "rows=0"
        )

        return output_path

    target_date = _target_timestamp(date_str)

    history_max_date = training_history[
        "_game_date_dt"
    ].max()

    if target_date <= history_max_date:
        prior = training_history.loc[
            training_history["_game_date_dt"] < target_date
        ]

        prior_unique_dates = int(
            prior["_game_date_dt"].nunique()
        )

        if prior_unique_dates < MIN_PRIOR_UNIQUE_DATES:
            return _write_unavailable_historical_result(
                date_str,
                output_path,
                joined,
                x_features,
                (
                    "insufficient_prior_dates_"
                    f"{prior_unique_dates}"
                    "_need_"
                    f"{MIN_PRIOR_UNIQUE_DATES}"
                ),
            )

        try:
            fitted = fit_walk_forward_models(
                target_date,
                training_history,
                feature_columns,
                trainer,
            )

            if fitted is None:
                return _write_unavailable_historical_result(
                    date_str,
                    output_path,
                    joined,
                    x_features,
                    "walk_forward_fit_unavailable",
                )

            (
                home_model,
                away_model,
                audit,
            ) = fitted

            home_predictions = (
                home_model.predict(x_features)
                if audit["home_selection"]["source"] == "learned"
                else pd.to_numeric(
                    joined["dratings_home_projected_runs"], errors="coerce"
                ).to_numpy(dtype=float)
            )
            away_predictions = (
                away_model.predict(x_features)
                if audit["away_selection"]["source"] == "learned"
                else pd.to_numeric(
                    joined["dratings_away_projected_runs"], errors="coerce"
                ).to_numpy(dtype=float)
            )
            home_runs = validate_predictions(
                home_predictions,
                "walk_forward_home_runs",
                date_str,
                joined,
            )
            away_runs = validate_predictions(
                away_predictions,
                "walk_forward_away_runs",
                date_str,
                joined,
            )

            run_model_version = audit["version"]

            projection_status_suffix = (
                ";projection_mode=walk_forward"
                ";training_end_date="
                f"{audit['training_end_date']}"
                ";training_rows="
                f"{audit['training_rows']}"
                ";home_source="
                f"{audit['home_selection']['source']}"
                ";away_source="
                f"{audit['away_selection']['source']}"
            )

            _log(
                "WALK_FORWARD "
                f"date={date_str} "
                "training_rows="
                f"{audit['training_rows']} "
                "training_unique_dates="
                f"{audit['training_unique_dates']} "
                "training_end="
                f"{audit['training_end_date']} "
                "home_validation_score="
                f"{float(audit['home_validation_score']):.12f} "
                "away_validation_score="
                f"{float(audit['away_validation_score']):.12f} "
                f"home_holdout={audit['home_selection']} "
                f"away_holdout={audit['away_selection']}"
            )

        except Exception as exc:
            _log(
                "WALK_FORWARD DATE FAILED "
                f"date={date_str}: {exc}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )

            return _write_unavailable_historical_result(
                date_str,
                output_path,
                joined,
                x_features,
                "walk_forward_fit_failed",
            )

    else:
        home_runs = validate_predictions(
            production_home_model.predict(x_features),
            "production_home_runs_model",
            date_str,
            joined,
        )

        away_runs = validate_predictions(
            production_away_model.predict(x_features),
            "production_away_runs_model",
            date_str,
            joined,
        )

        run_model_version = (
            model_version(
                home_metadata,
                "home",
            )
            + "|"
            + model_version(
                away_metadata,
                "away",
            )
        )

        projection_status_suffix = (
            ";projection_mode=production"
            ";historical_training_max="
            f"{history_max_date.date().isoformat()}"
        )

        _log(
            "PRODUCTION_MODEL "
            f"date={date_str} "
            "historical_training_max="
            f"{history_max_date.date().isoformat()}"
        )

    result = joined.copy()

    result["model_home_runs"] = home_runs
    result["model_away_runs"] = away_runs

    result["model_total_runs"] = (
        result["model_home_runs"]
        + result["model_away_runs"]
    )

    result["run_model_version"] = run_model_version

    result["run_model_feature_status"] = (
        build_feature_status(
            result,
            x_features,
        )
        .astype("string")
        + projection_status_suffix
    )

    for col in [
        "dratings_home_prob",
        "dratings_away_prob",
        "dratings_home_projected_runs",
        "dratings_away_projected_runs",
        "dratings_total_projected_runs",
    ]:
        if col not in result.columns:
            fail(
                "Missing required preserved "
                f"DRatings column: {col}"
            )

    _assert_current_day_source_coverage(
        date_str,
        pred,
        result,
        "build_run_projection",
    )

    result.to_csv(
        output_path,
        index=False,
    )

    _log(
        f"WROTE {output_path} "
        f"rows={len(result)} "
        f"features={len(feature_columns)}"
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dates",
        nargs="*",
        help=(
            "Optional date(s) as "
            "YYYY_MM_DD or YYYY-MM-DD. "
            "If omitted, rebuild every "
            "prediction date."
        ),
    )

    return parser.parse_args()


def normalize_date_arg(
    value: str,
) -> str:
    return str(value).strip().replace("-", "_")


def discover_all_dates() -> list[str]:
    files = sorted(
        PRED_DIR.glob("*_MLB.csv")
    )

    if not files:
        fail(
            "No prediction files found "
            f"in {PRED_DIR}"
        )

    return [
        path.stem[:-4]
        for path in files
    ]


def load_runtime_models():
    _log(
        "Loading production "
        "model metadata"
    )

    home_metadata = load_metadata(
        HOME_METADATA_FILE,
        "home_runs",
    )

    away_metadata = load_metadata(
        AWAY_METADATA_FILE,
        "away_runs",
    )

    feature_columns = assert_metadata_feature_contract(
        home_metadata,
        away_metadata,
    )

    _log(
        "Loading committed "
        "production models"
    )

    home_model = load_model(
        HOME_MODEL_FILE,
        "home_runs",
    )

    away_model = load_model(
        AWAY_MODEL_FILE,
        "away_runs",
    )

    assert_model_feature_order(
        home_model,
        feature_columns,
        "home_runs",
    )

    assert_model_feature_order(
        away_model,
        feature_columns,
        "away_runs",
    )

    _log(
        "Production model loading "
        "complete; "
        f"features={len(feature_columns)}"
    )

    return (
        home_model,
        away_model,
        home_metadata,
        away_metadata,
        feature_columns,
    )


def main() -> None:
    args = parse_args()

    with LOG_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "=== build_run_projection "
            f"RUN {_now()} ===\n"
        )

    try:
        dates = [
            normalize_date_arg(value)
            for value in args.dates
            if str(value).strip()
        ]

        if not dates:
            dates = discover_all_dates()

        dates = sorted(
            dict.fromkeys(dates)
        )

        _log(
            "Dates to rebuild: "
            f"{len(dates)} "
            f"first={dates[0]} "
            f"last={dates[-1]}"
        )

        (
            production_home_model,
            production_away_model,
            home_metadata,
            away_metadata,
            feature_columns,
        ) = load_runtime_models()

        training_builder = _load_python_module(
            TRAINING_BUILDER_FILE,
            "_mlb_build_run_training_set_for_projection",
        )

        trainer = _load_python_module(
            TRAINER_FILE,
            "_mlb_train_run_model_for_projection",
        )

        training_history = build_training_history_in_memory(
            training_builder,
            feature_columns,
        )

        dates_processed = 0

        for date_str in dates:
            output_path = process_date(
                date_str=date_str,
                production_home_model=production_home_model,
                production_away_model=production_away_model,
                home_metadata=home_metadata,
                away_metadata=away_metadata,
                feature_columns=feature_columns,
                training_history=training_history,
                trainer=trainer,
            )

            dates_processed += 1

            print(f"WROTE {output_path}")

        _log(
            "SUCCESS "
            f"dates_processed={dates_processed}"
        )

    except Exception as exc:
        _log(
            f"FATAL: {exc}\n"
            f"{traceback.format_exc()}",
            "ERROR",
        )

        print(
            "build_run_projection failed: "
            f"{exc}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
