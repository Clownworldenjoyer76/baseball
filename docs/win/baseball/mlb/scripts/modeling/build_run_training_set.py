#!/usr/bin/env python3
"""Build the leakage-safe MLB run-model training dataset.

Required per-date sources:
    00_intake/predictions/pred_with_game_id/{date}_MLB.csv
    00_intake/games/{date}_games.csv
    00_intake/sportsdataverse/{date}_sportsdataverse.csv
    05_final_scores/results/final_scores/{date}_final_scores_MLB.csv

Optional:
    data/weather/{date}_weather.csv

The output contains only explicitly approved pregame model inputs plus targets
and audit identifiers. Sportsbook prices, implied probabilities, grading
results, and postgame-derived fields other than final-run targets are never
copied into the training dataset.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("docs/win/baseball/mlb")

PRED_DIR = BASE_DIR / "00_intake/predictions/pred_with_game_id"
GAMES_DIR = BASE_DIR / "00_intake/games"
SDV_DIR = BASE_DIR / "00_intake/sportsdataverse"
FINAL_DIR = BASE_DIR / "05_final_scores/results/final_scores"
WEATHER_DIR = BASE_DIR / "data/weather"

OUTPUT_DIR = BASE_DIR / "modeling/data"
OUTPUT_FILE = OUTPUT_DIR / "mlb_run_training_set.csv"

ERROR_DIR = BASE_DIR / "errors/modeling"
LOG_FILE = ERROR_DIR / "build_run_training_set.txt"

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
    "gamePk",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
]

SDV_REQUIRED = [
    "gamePk",
    "game_id",
    "game_date",
    "sdv_as_of_date",
    "sdv_status",
    "sdv_home_sp_found",
    "sdv_away_sp_found",
    "sdv_home_sp_stuff_plus",
    "sdv_away_sp_stuff_plus",
    "sdv_home_sp_command_plus",
    "sdv_away_sp_command_plus",
    "sdv_home_sp_xera",
    "sdv_away_sp_xera",
    "sdv_home_sp_xera_30d",
    "sdv_away_sp_xera_30d",
    "sdv_home_sp_xwoba",
    "sdv_away_sp_xwoba",
    "sdv_home_sp_xwoba_30d",
    "sdv_away_sp_xwoba_30d",
    "sdv_home_sp_avg_velo",
    "sdv_away_sp_avg_velo",
    "sdv_home_sp_avg_velo_30d",
    "sdv_away_sp_avg_velo_30d",
    "sdv_home_sp_velo_delta_30d",
    "sdv_away_sp_velo_delta_30d",
    "sdv_home_sp_pitches",
    "sdv_away_sp_pitches",
    "sdv_home_sp_games",
    "sdv_away_sp_games",
    "sdv_home_sp_pitches_30d",
    "sdv_away_sp_pitches_30d",
    "sdv_home_sp_games_30d",
    "sdv_away_sp_games_30d",
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

FINAL_REQUIRED = [
    "gamePk",
    "game_id",
    "game_date",
    "game_status",
    "final_home_score",
    "final_away_score",
]

WEATHER_FEATURES = [
    "temp_f",
    "wind_mph",
    "wind_blowing_out",
    "humidity",
    "air_pressure_at_sea_level",
    "dew_point_f",
    "weather_applicable",
]

WEATHER_PROVENANCE_COLUMNS = [
    "weather_generated_at",
    "forecast_generated_at",
    "generated_at",
    "retrieved_at",
    "created_at",
]

SDV_FEATURE_MAP = {
    "sdv_home_sp_stuff_plus": "home_sp_pitch_quality_plus",
    "sdv_away_sp_stuff_plus": "away_sp_pitch_quality_plus",
    "sdv_home_sp_command_plus": "home_sp_command_plus",
    "sdv_away_sp_command_plus": "away_sp_command_plus",
    "sdv_home_sp_xera": "home_sp_xera",
    "sdv_away_sp_xera": "away_sp_xera",
    "sdv_home_sp_xera_30d": "home_sp_xera_30d",
    "sdv_away_sp_xera_30d": "away_sp_xera_30d",
    "sdv_home_sp_xwoba": "home_sp_xwoba",
    "sdv_away_sp_xwoba": "away_sp_xwoba",
    "sdv_home_sp_xwoba_30d": "home_sp_xwoba_30d",
    "sdv_away_sp_xwoba_30d": "away_sp_xwoba_30d",
    "sdv_home_sp_avg_velo": "home_sp_avg_velo",
    "sdv_away_sp_avg_velo": "away_sp_avg_velo",
    "sdv_home_sp_avg_velo_30d": "home_sp_avg_velo_30d",
    "sdv_away_sp_avg_velo_30d": "away_sp_avg_velo_30d",
    "sdv_home_sp_velo_delta_30d": "home_sp_velo_delta_30d",
    "sdv_away_sp_velo_delta_30d": "away_sp_velo_delta_30d",
    "sdv_home_sp_pitches": "home_sp_pitches",
    "sdv_away_sp_pitches": "away_sp_pitches",
    "sdv_home_sp_games": "home_sp_games",
    "sdv_away_sp_games": "away_sp_games",
    "sdv_home_sp_pitches_30d": "home_sp_pitches_30d",
    "sdv_away_sp_pitches_30d": "away_sp_pitches_30d",
    "sdv_home_sp_games_30d": "home_sp_games_30d",
    "sdv_away_sp_games_30d": "away_sp_games_30d",
    "sdv_home_bp_pa_14d": "home_bp_pa_14d",
    "sdv_away_bp_pa_14d": "away_bp_pa_14d",
    "sdv_home_bp_woba_allowed_14d": "home_bp_woba_allowed_14d",
    "sdv_away_bp_woba_allowed_14d": "away_bp_woba_allowed_14d",
    "sdv_home_bp_k_rate_14d": "home_bp_k_rate_14d",
    "sdv_away_bp_k_rate_14d": "away_bp_k_rate_14d",
    "sdv_home_bp_bb_rate_14d": "home_bp_bb_rate_14d",
    "sdv_away_bp_bb_rate_14d": "away_bp_bb_rate_14d",
    "sdv_home_bp_hard_rate_14d": "home_bp_hard_rate_14d",
    "sdv_away_bp_hard_rate_14d": "away_bp_hard_rate_14d",
    "sdv_home_bp_pitches_3d": "home_bp_pitches_3d",
    "sdv_away_bp_pitches_3d": "away_bp_pitches_3d",
    "sdv_home_bp_pa_7d": "home_bp_pa_7d",
    "sdv_away_bp_pa_7d": "away_bp_pa_7d",
    "sdv_home_bp_woba_allowed_7d": "home_bp_woba_allowed_7d",
    "sdv_away_bp_woba_allowed_7d": "away_bp_woba_allowed_7d",
    "sdv_home_bp_k_rate_7d": "home_bp_k_rate_7d",
    "sdv_away_bp_k_rate_7d": "away_bp_k_rate_7d",
    "sdv_home_bp_bb_rate_7d": "home_bp_bb_rate_7d",
    "sdv_away_bp_bb_rate_7d": "away_bp_bb_rate_7d",
    "sdv_home_bp_hard_rate_7d": "home_bp_hard_rate_7d",
    "sdv_away_bp_hard_rate_7d": "away_bp_hard_rate_7d",
}

AUDIT_COLUMNS = [
    "game_date",
    "game_id",
    "gamePk",
    "home_team",
    "away_team",
    "sdv_as_of_date",
]

DRATINGS_COLUMNS = [
    "dratings_home_prob",
    "dratings_away_prob",
    "dratings_home_projected_runs",
    "dratings_away_projected_runs",
    "dratings_total_projected_runs",
]

SAFE_GAME_FEATURES = [
    "venue_id",
    "day_night",
]

TARGET_COLUMNS = [
    "target_home_runs",
    "target_away_runs",
]

FORBIDDEN_OUTPUT_TOKENS = [
    "dk_",
    "moneyline_american",
    "moneyline_decimal",
    "run_line_american",
    "run_line_decimal",
    "total_over_american",
    "total_under_american",
    "implied_prob",
    "break_even_prob",
    "juice",
    "graded",
    "bet_result",
    "selected_bet",
    "final_total",
    "final_scores_generated_at",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _log(message: str, level: str = "INFO") -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {message.rstrip()}\n")


def fail(message: str) -> None:
    _log(message, "ERROR")
    raise RuntimeError(message)


def duplicate_columns(columns) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []

    for col in columns:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)

    return dupes


def read_csv_checked(
    path: Path,
    required: list[str],
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        fail(f"{label} missing required file: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
    )

    dupes = duplicate_columns(
        list(df.columns)
    )

    if dupes:
        fail(
            f"{label} duplicate columns: {dupes}"
        )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        fail(
            f"{label} missing required columns: {missing}"
        )

    return df


def _normalize_game_id(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
    )


def _normalize_gamepk(
    series: pd.Series,
) -> pd.Series:
    raw = (
        series
        .astype("string")
        .str.strip()
    )

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


def _normalize_date(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series
        .astype("string")
        .str.replace("_", "-", regex=False),
        errors="coerce",
    ).dt.normalize()


def _assert_unique_key(
    df: pd.DataFrame,
    key: str,
    label: str,
    summary: dict,
) -> None:
    if key not in df.columns:
        return

    values = (
        df[key]
        .astype("string")
        .str.strip()
    )

    nonblank = (
        values.notna()
        & (values != "")
    )

    duplicated = (
        nonblank
        & values.duplicated(
            keep=False
        )
    )

    count = int(
        duplicated.sum()
    )

    if count:
        metric = (
            "duplicate_game_id"
            if key == "game_id"
            else "duplicate_gamePk"
        )

        summary[metric] += count

        sample = (
            values.loc[duplicated]
            .head(10)
            .tolist()
        )

        fail(
            f"{label} contains duplicate {key}; "
            f"duplicate_rows={count}; "
            f"sample={sample}"
        )


def _drop_blank_game_id_rows(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    if "game_id" not in df.columns:
        return df

    values = _normalize_game_id(
        df["game_id"]
    )

    keep = (
        values.notna()
        & (values != "")
    )

    dropped = int(
        (~keep).sum()
    )

    if dropped:
        _log(
            f"{label} dropped rows with blank game_id: {dropped}",
            "WARN",
        )

    return df.loc[
        keep
    ].copy()


def _drop_blank_gamepk_rows(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    if "gamePk" not in df.columns:
        return df

    values = _normalize_gamepk(
        df["gamePk"]
    )

    keep = (
        values.notna()
        & (values != "")
    )

    dropped = int(
        (~keep).sum()
    )

    if dropped:
        _log(
            f"{label} dropped rows with blank gamePk: {dropped}",
            "WARN",
        )

    return df.loc[
        keep
    ].copy()


def _prepare_source_keys(
    df: pd.DataFrame,
    label: str,
    summary: dict,
) -> pd.DataFrame:
    df = df.copy()

    if "game_id" in df.columns:
        df["game_id"] = _normalize_game_id(
            df["game_id"]
        )

        _assert_unique_key(
            df,
            "game_id",
            label,
            summary,
        )

    if "gamePk" in df.columns:
        df["gamePk"] = _normalize_gamepk(
            df["gamePk"]
        )

        _assert_unique_key(
            df,
            "gamePk",
            label,
            summary,
        )

    return df


def _assert_secondary_game_id_match(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    label: str,
) -> None:
    if (
        left_col not in df.columns
        or right_col not in df.columns
    ):
        return

    left = _normalize_game_id(
        df[left_col]
    )

    right = _normalize_game_id(
        df[right_col]
    )

    comparable = (
        left.notna()
        & right.notna()
        & (left != "")
        & (right != "")
    )

    mismatch = (
        comparable
        & (left != right)
    )

    if mismatch.any():
        sample_cols = [
            col
            for col in [
                "gamePk",
                left_col,
                right_col,
            ]
            if col in df.columns
        ]

        sample = (
            df.loc[
                mismatch,
                sample_cols,
            ]
            .head(10)
            .to_dict("records")
        )

        fail(
            f"{label} game_id consistency check failed; "
            f"bad_rows={int(mismatch.sum())}; "
            f"sample={sample}"
        )


def _assert_secondary_gamepk_match(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    label: str,
) -> None:
    if (
        left_col not in df.columns
        or right_col not in df.columns
    ):
        return

    left = _normalize_gamepk(
        df[left_col]
    )

    right = _normalize_gamepk(
        df[right_col]
    )

    comparable = (
        left.notna()
        & right.notna()
        & (left != "")
        & (right != "")
    )

    mismatch = (
        comparable
        & (left != right)
    )

    if mismatch.any():
        sample_cols = [
            col
            for col in [
                "game_id",
                left_col,
                right_col,
            ]
            if col in df.columns
        ]

        sample = (
            df.loc[
                mismatch,
                sample_cols,
            ]
            .head(10)
            .to_dict("records")
        )

        fail(
            f"{label} gamePk consistency check failed; "
            f"bad_rows={int(mismatch.sum())}; "
            f"sample={sample}"
        )


def _assert_team_consistency(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    label: str,
) -> None:
    if (
        left_col not in df.columns
        or right_col not in df.columns
    ):
        return

    left = (
        df[left_col]
        .astype("string")
        .str.strip()
    )

    right = (
        df[right_col]
        .astype("string")
        .str.strip()
    )

    comparable = (
        left.notna()
        & right.notna()
        & (left != "")
        & (right != "")
    )

    mismatch = (
        comparable
        & (left != right)
    )

    if mismatch.any():
        sample = (
            df.loc[
                mismatch,
                [
                    "game_id",
                    left_col,
                    right_col,
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        fail(
            f"{label} team consistency check failed; "
            f"bad_rows={int(mismatch.sum())}; "
            f"sample={sample}"
        )


def _resolve_game_datetime(
    games: pd.DataFrame,
) -> pd.Series:
    if "game_time" not in games.columns:
        return pd.Series(
            pd.NaT,
            index=games.index,
            dtype="datetime64[ns]",
        )

    date_text = (
        games["game_date"]
        .astype("string")
        .str.replace(
            "_",
            "-",
            regex=False,
        )
        .str.strip()
    )

    time_text = (
        games["game_time"]
        .astype("string")
        .str.strip()
    )

    return pd.to_datetime(
        date_text + " " + time_text,
        errors="coerce",
    )


def _safe_weather_frame(
    weather_path: Path,
    games: pd.DataFrame,
    summary: dict,
) -> pd.DataFrame | None:
    if not weather_path.exists():
        _log(
            f"weather missing; no weather features joined: {weather_path}",
            "INFO",
        )
        return None

    weather = pd.read_csv(
        weather_path,
        dtype=str,
        encoding="utf-8-sig",
    )

    dupes = duplicate_columns(
        list(weather.columns)
    )

    if dupes:
        fail(
            f"weather duplicate columns: {dupes}"
        )

    if "gamePk" not in weather.columns:
        _log(
            f"weather ignored because gamePk is missing: {weather_path}",
            "WARN",
        )
        return None

    weather = _prepare_source_keys(
        weather,
        f"weather {weather_path.name}",
        summary,
    )

    provenance_col = next(
        (
            col
            for col in WEATHER_PROVENANCE_COLUMNS
            if col in weather.columns
        ),
        None,
    )

    if provenance_col is None:
        _log(
            "weather ignored because no pregame-generation provenance "
            f"timestamp exists: {weather_path.name}",
            "WARN",
        )
        return None

    keep_features = [
        col
        for col in WEATHER_FEATURES
        if col in weather.columns
    ]

    if not keep_features:
        _log(
            "weather ignored because no approved environment features exist: "
            f"{weather_path.name}",
            "WARN",
        )
        return None

    weather_gamepk = _normalize_gamepk(
        weather["gamePk"]
    )

    weather = weather.loc[
        weather_gamepk.notna()
        & (weather_gamepk != "")
    ].copy()

    game_times = games[
        ["gamePk"]
    ].copy()

    game_times["_game_start"] = (
        _resolve_game_datetime(games)
    )

    game_gamepk = _normalize_gamepk(
        game_times["gamePk"]
    )

    game_times = game_times.loc[
        game_gamepk.notna()
        & (game_gamepk != "")
    ].copy()

    check = game_times.merge(
        weather[
            ["gamePk", provenance_col]
            + keep_features
        ],
        on="gamePk",
        how="left",
        validate="one_to_one",
    )

    generated = pd.to_datetime(
        check[provenance_col],
        errors="coerce",
    )

    game_start = pd.to_datetime(
        check["_game_start"],
        errors="coerce",
    )

    unsafe = (
        generated.isna()
        | game_start.isna()
        | (generated >= game_start)
    )

    matched_weather = (
        check[provenance_col]
        .notna()
    )

    unsafe_matched = (
        unsafe
        & matched_weather
    )

    if unsafe_matched.any():
        sample = (
            check.loc[
                unsafe_matched,
                [
                    "gamePk",
                    provenance_col,
                    "_game_start",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        fail(
            "weather provenance is not safely pregame for "
            f"{weather_path.name}; "
            f"bad_rows={int(unsafe_matched.sum())}; "
            f"sample={sample}"
        )

    return weather[
        ["gamePk"]
        + keep_features
    ].copy()


def discover_dates(summary: dict) -> list[str]:
    prediction_dates: list[str] = []

    for path in sorted(
        PRED_DIR.glob("*_MLB.csv")
    ):
        stem = path.stem

        if not stem.endswith("_MLB"):
            continue

        date_str = stem[:-4]

        if date_str:
            prediction_dates.append(
                date_str
            )

    final_dates: set[str] = set()

    final_suffix = "_final_scores_MLB"

    for path in sorted(
        FINAL_DIR.glob("*_final_scores_MLB.csv")
    ):
        stem = path.stem

        if not stem.endswith(
            final_suffix
        ):
            continue

        date_str = stem[
            :-len(final_suffix)
        ]

        if date_str:
            final_dates.add(
                date_str
            )

    if not final_dates:
        fail(
            f"No final-score files found in {FINAL_DIR}"
        )

    eligible_dates = [
        date_str
        for date_str in prediction_dates
        if date_str in final_dates
    ]

    missing_final_dates = [
        date_str
        for date_str in prediction_dates
        if date_str not in final_dates
    ]

    summary["missing_final_score_file"] += len(
        missing_final_dates
    )

    if missing_final_dates:
        _log(
            "Skipping prediction dates with no matching final-score file: "
            + ", ".join(missing_final_dates),
            "INFO",
        )

    return eligible_dates


def _filter_dates(
    dates: list[str],
    from_date: str | None,
    to_date: str | None,
) -> list[str]:
    if (
        not from_date
        and not to_date
    ):
        return dates

    if (
        not from_date
        or not to_date
    ):
        fail(
            "--from-date and --to-date must be provided together"
        )

    start = pd.Timestamp(
        str(from_date).replace(
            "_",
            "-",
        )
    ).normalize()

    end = pd.Timestamp(
        str(to_date).replace(
            "_",
            "-",
        )
    ).normalize()

    if start > end:
        fail(
            "--from-date must be <= --to-date"
        )

    selected: list[str] = []

    for date_str in dates:
        parsed = pd.Timestamp(
            date_str.replace(
                "_",
                "-",
            )
        ).normalize()

        if start <= parsed <= end:
            selected.append(
                date_str
            )

    return selected


def _rename_prediction_features(
    pred: pd.DataFrame,
) -> pd.DataFrame:
    return pred.rename(
        columns={
            "home_prob": "dratings_home_prob",
            "away_prob": "dratings_away_prob",
            "home_projected_runs": "dratings_home_projected_runs",
            "away_projected_runs": "dratings_away_projected_runs",
            "total_projected_runs": "dratings_total_projected_runs",
        }
    )



def _log_unresolved_final_gamepks(date_str, count):
    if count:
        _log(
            f"final_scores {date_str} unresolved gamePk rows: {count}",
            "WARN",
        )


def _warn_final_score_join_misses(date_str, joined, mask):
    if not mask.any():
        return
    sample = joined.loc[
        mask,
        ["game_id", "gamePk", "game_date", "home_team", "away_team"],
    ].head(10).to_dict("records")
    _log(
        f"{date_str} final-score join misses: {int(mask.sum())}; sample={sample}",
        "WARN",
    )


def _validate_matched_final_rows(date_str, joined, mask):
    if not mask.any():
        return
    sample = joined.loc[
        mask,
        ["game_id", "gamePk", "game_status", "final_home_score", "final_away_score"],
    ].head(10).to_dict("records")
    fail(
        f"{date_str} matched final-score rows are invalid; "
        f"bad_rows={int(mask.sum())}; sample={sample}"
    )


def _validate_training_sdv_leakage(date_str, joined, mask):
    if not mask.any():
        return
    sample = joined.loc[
        mask,
        ["game_id", "gamePk", "game_date", "sdv_as_of_date"],
    ].head(10).to_dict("records")
    fail(
        f"{date_str} SDV leakage validation failed; "
        "sdv_as_of_date must be < game_date; "
        f"bad_rows={int(mask.sum())}; sample={sample}"
    )


def _merge_training_weather(joined, weather):
    if weather is None:
        return joined
    return joined.merge(weather, on="gamePk", how="left", validate="many_to_one")



def _audit_dratings_probabilities(df: pd.DataFrame, summary: dict, label: str) -> None:
    home = pd.to_numeric(df["dratings_home_prob"], errors="coerce")
    away = pd.to_numeric(df["dratings_away_prob"], errors="coerce")
    home_blank = df["dratings_home_prob"].isna() | (df["dratings_home_prob"].astype(str).str.strip() == "")
    away_blank = df["dratings_away_prob"].isna() | (df["dratings_away_prob"].astype(str).str.strip() == "")
    missing_both = home_blank & away_blank
    incomplete = home_blank ^ away_blank
    invalid = (
        (~home_blank & (~np.isfinite(home) | (home < 0) | (home > 1)))
        | (~away_blank & (~np.isfinite(away) | (away < 0) | (away > 1)))
        | (~missing_both & ~incomplete & ((home + away - 1).abs() > 0.002))
    )
    summary["missing_dratings_prob_rows"] = (
        summary.get("missing_dratings_prob_rows", 0) + int(missing_both.sum())
    )
    summary["invalid_dratings_prob_rows"] = (
        summary.get("invalid_dratings_prob_rows", 0)
        + int((invalid | incomplete).sum())
    )
    if missing_both.any():
        sample = df.loc[missing_both, ["game_id"]].head(10).to_dict("records")
        _log(f"{label} missing both DRatings probability features: "
             f"rows={int(missing_both.sum())}; sample={sample}", "WARN")
    if (invalid | incomplete).any():
        sample = df.loc[invalid | incomplete, [
            "game_id", "dratings_home_prob", "dratings_away_prob",
        ]].head(10).to_dict("records")
        fail(f"{label} invalid or partial DRatings probability pair; "
             f"bad_rows={int((invalid | incomplete).sum())}; sample={sample}")

def _coerce_training_output_columns(output):
    numeric_columns = (
        DRATINGS_COLUMNS
        + list(SDV_FEATURE_MAP.values())
        + TARGET_COLUMNS
        + WEATHER_FEATURES
    )
    for col in numeric_columns:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")
    output["game_date"] = pd.to_datetime(
        output["game_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    output["sdv_as_of_date"] = pd.to_datetime(
        output["sdv_as_of_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return output


def build_date_training_rows(
    date_str: str,
    summary: dict,
) -> pd.DataFrame:
    pred_path = (
        PRED_DIR
        / f"{date_str}_MLB.csv"
    )

    games_path = (
        GAMES_DIR
        / f"{date_str}_games.csv"
    )

    sdv_path = (
        SDV_DIR
        / f"{date_str}_sportsdataverse.csv"
    )

    final_path = (
        FINAL_DIR
        / f"{date_str}_final_scores_MLB.csv"
    )

    weather_path = (
        WEATHER_DIR
        / f"{date_str}_weather.csv"
    )

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

    final = read_csv_checked(
        final_path,
        FINAL_REQUIRED,
        f"final_scores {date_str}",
    )

    pred = _prepare_source_keys(
        pred,
        f"predictions {date_str}",
        summary,
    )

    games = _prepare_source_keys(
        games,
        f"games {date_str}",
        summary,
    )

    sdv = _prepare_source_keys(
        sdv,
        f"sportsdataverse {date_str}",
        summary,
    )

    final = _prepare_source_keys(
        final,
        f"final_scores {date_str}",
        summary,
    )

    summary["rows_loaded"] += len(
        pred
    )

    pred = _rename_prediction_features(
        pred
    )
    _audit_dratings_probabilities(pred, summary, f"predictions {date_str}")

    pred_keep = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
    ] + DRATINGS_COLUMNS

    game_keep = [
        col
        for col in [
            "gamePk",
            "game_id",
            "game_date",
            "game_time",
            "home_team",
            "away_team",
            "venue_id",
            "day_night",
        ]
        if col in games.columns
    ]

    joined = pred[
        pred_keep
    ].merge(
        games[
            game_keep
        ],
        on="game_id",
        how="inner",
        suffixes=(
            "_pred",
            "_games",
        ),
        validate="one_to_one",
    )

    summary["rows_joined"] += len(
        joined
    )

    if len(joined) != len(pred):
        missing = (
            len(pred)
            - len(joined)
        )

        fail(
            f"{date_str} prediction->games join lost {missing} rows; "
            "game_id spine must resolve one-to-one"
        )

    _assert_team_consistency(
        joined,
        "home_team_pred",
        "home_team_games",
        f"{date_str} home team",
    )

    _assert_team_consistency(
        joined,
        "away_team_pred",
        "away_team_games",
        f"{date_str} away team",
    )

    joined["game_date"] = (
        joined["game_date_games"]
    )

    joined["home_team"] = (
        joined["home_team_games"]
    )

    joined["away_team"] = (
        joined["away_team_games"]
    )

    sdv_keep = [
        "gamePk",
        "game_id",
        "sdv_as_of_date",
        "sdv_status",
        "sdv_home_sp_found",
        "sdv_away_sp_found",
    ] + list(
        SDV_FEATURE_MAP.keys()
    )

    sdv_join = sdv[
        sdv_keep
    ].copy()

    sdv_join = _drop_blank_gamepk_rows(
        sdv_join,
        f"sportsdataverse {date_str} join",
    )

    sdv_join = sdv_join.rename(
        columns={
            "game_id": "game_id_sdv",
        }
    )

    joined = joined.merge(
        sdv_join,
        on="gamePk",
        how="left",
        validate="one_to_one",
    )

    missing_sdv = (
        joined["sdv_as_of_date"]
        .isna()
    )

    summary["missing_sdv"] += int(
        missing_sdv.sum()
    )

    _assert_secondary_game_id_match(
        joined,
        "game_id",
        "game_id_sdv",
        f"{date_str} games->sportsdataverse",
    )

    final_keep = [
        "gamePk",
        "game_id",
        "game_status",
        "final_home_score",
        "final_away_score",
    ]

    final_gamepk = _normalize_gamepk(
        final["gamePk"]
    )

    unresolved_final_gamepk = (
        final_gamepk.isna()
        | (final_gamepk == "")
    )

    unresolved_gamepk_count = int(
        unresolved_final_gamepk.sum()
    )

    summary["unresolved_gamePk"] += (
        unresolved_gamepk_count
    )

    unresolved_game_ids = set(
        _normalize_game_id(
            final.loc[
                unresolved_final_gamepk,
                "game_id",
            ]
        )
        .dropna()
        .loc[
            lambda s: s != ""
        ]
        .tolist()
    )

    _log_unresolved_final_gamepks(date_str, unresolved_gamepk_count)

    final_join = final.loc[
        ~unresolved_final_gamepk,
        final_keep,
    ].copy()

    final_join["_final_score_joined"] = "1"

    final_join = final_join.rename(
        columns={
            "game_id": "game_id_final",
        }
    )

    joined = joined.merge(
        final_join,
        on="gamePk",
        how="left",
        validate="one_to_one",
    )

    final_join_missing = (
        joined["_final_score_joined"]
        .isna()
    )

    unresolved_gamepk_join_miss = (
        final_join_missing
        & joined["game_id"].isin(
            unresolved_game_ids
        )
    )

    failed_final_score_join = (
        final_join_missing
        & ~unresolved_gamepk_join_miss
    )

    summary["failed_final_score_join"] += int(
        failed_final_score_join.sum()
    )

    _warn_final_score_join_misses(date_str, joined, failed_final_score_join)

    _assert_secondary_game_id_match(
        joined,
        "game_id",
        "game_id_final",
        f"{date_str} games->final_scores",
    )

    final_home = pd.to_numeric(
        joined["final_home_score"],
        errors="coerce",
    )

    final_away = pd.to_numeric(
        joined["final_away_score"],
        errors="coerce",
    )

    final_status = (
        joined["game_status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid_matched_final = (
        ~final_join_missing
        & (
            final_status.ne("final")
            | final_home.isna()
            | final_away.isna()
            | ~np.isfinite(final_home)
            | ~np.isfinite(final_away)
            | (final_home < 0)
            | (final_away < 0)
        )
    )

    _validate_matched_final_rows(date_str, joined, invalid_matched_final)

    invalid_final = (
        final_join_missing
        | invalid_matched_final
    )

    joined["target_home_runs"] = (
        final_home
    )

    joined["target_away_runs"] = (
        final_away
    )

    game_date = _normalize_date(
        joined["game_date"]
    )

    sdv_as_of_date = _normalize_date(
        joined["sdv_as_of_date"]
    )

    leakage = (
        ~missing_sdv
        & (
            game_date.isna()
            | sdv_as_of_date.isna()
            | (
                sdv_as_of_date
                >= game_date
            )
        )
    )

    summary[
        "leakage_rejections"
    ] += int(
        leakage.sum()
    )

    _validate_training_sdv_leakage(date_str, joined, leakage)

    keep = (
        ~missing_sdv
        & ~invalid_final
    )

    joined = joined.loc[
        keep
    ].copy()

    joined = joined.rename(
        columns=SDV_FEATURE_MAP
    )

    weather = _safe_weather_frame(
        weather_path,
        games,
        summary,
    )

    joined = _merge_training_weather(joined, weather)

    output_columns = (
        AUDIT_COLUMNS
        + DRATINGS_COLUMNS
        + [
            col
            for col in SAFE_GAME_FEATURES
            if col in joined.columns
        ]
        + list(
            SDV_FEATURE_MAP.values()
        )
        + [
            col
            for col in WEATHER_FEATURES
            if col in joined.columns
        ]
        + TARGET_COLUMNS
    )

    output = joined[
        output_columns
    ].copy()

    output = _coerce_training_output_columns(output)

    return output


def validate_final_output(
    df: pd.DataFrame,
) -> None:
    dupes = duplicate_columns(
        list(df.columns)
    )

    if dupes:
        fail(
            f"training output has duplicate columns: {dupes}"
        )

    missing = [
        col
        for col in (
            AUDIT_COLUMNS
            + DRATINGS_COLUMNS
            + list(
                SDV_FEATURE_MAP.values()
            )
            + TARGET_COLUMNS
        )
        if col not in df.columns
    ]

    if missing:
        fail(
            f"training output missing required columns: {missing}"
        )

    forbidden = [
        col
        for col in df.columns
        if any(
            token in col.lower()
            for token in FORBIDDEN_OUTPUT_TOKENS
        )
    ]

    if forbidden:
        fail(
            "training output contains forbidden feature columns: "
            f"{forbidden}"
        )

    game_date = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    as_of = pd.to_datetime(
        df["sdv_as_of_date"],
        errors="coerce",
    )

    bad_leakage = (
        game_date.isna()
        | as_of.isna()
        | (as_of >= game_date)
    )

    if bad_leakage.any():
        sample = (
            df.loc[
                bad_leakage,
                [
                    "game_id",
                    "gamePk",
                    "game_date",
                    "sdv_as_of_date",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        fail(
            "final training output failed SDV as-of validation; "
            f"bad_rows={int(bad_leakage.sum())}; "
            f"sample={sample}"
        )

    for col in TARGET_COLUMNS:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        bad = (
            values.isna()
            | ~np.isfinite(values)
            | (values < 0)
        )

        if bad.any():
            sample = (
                df.loc[
                    bad,
                    [
                        "game_id",
                        "gamePk",
                        col,
                    ],
                ]
                .head(10)
                .to_dict("records")
            )

            fail(
                f"final training output invalid target {col}; "
                f"bad_rows={int(bad.sum())}; "
                f"sample={sample}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from-date",
        help=(
            "Optional inclusive start date "
            "(YYYY-MM-DD or YYYY_MM_DD)."
        ),
    )

    parser.add_argument(
        "--to-date",
        help=(
            "Optional inclusive end date "
            "(YYYY-MM-DD or YYYY_MM_DD)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with LOG_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== build_run_training_set RUN {_now()} ===\n"
        )

    summary = {
        "rows_loaded": 0,
        "rows_joined": 0,
        "missing_sdv": 0,
        "missing_final_score_file": 0,
        "unresolved_gamePk": 0,
        "failed_final_score_join": 0,
        "duplicate_game_id": 0,
        "duplicate_gamePk": 0,
        "leakage_rejections": 0,
        "missing_dratings_prob_rows": 0,
        "invalid_dratings_prob_rows": 0,
        "rows_written": 0,
    }

    try:
        dates = discover_dates(summary)

        dates = _filter_dates(
            dates,
            args.from_date,
            args.to_date,
        )

        if not dates:
            fail(
                "No prediction dates with matching final-score files "
                "found for the requested range"
            )

        _log(
            f"dates_to_process={len(dates)}"
        )

        all_rows: list[
            pd.DataFrame
        ] = []

        for date_str in dates:
            _log(
                f"processing date={date_str}"
            )

            frame = build_date_training_rows(
                date_str,
                summary,
            )

            if not frame.empty:
                all_rows.append(
                    frame
                )

        if all_rows:
            training = pd.concat(
                all_rows,
                ignore_index=True,
                sort=False,
            )

        else:
            training = pd.DataFrame(
                columns=(
                    AUDIT_COLUMNS
                    + DRATINGS_COLUMNS
                    + SAFE_GAME_FEATURES
                    + list(
                        SDV_FEATURE_MAP.values()
                    )
                    + TARGET_COLUMNS
                )
            )

        validate_final_output(
            training
        )

        training = training.sort_values(
            [
                "game_date",
                "game_id",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

        summary["rows_written"] = len(
            training
        )

        training.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        for key in [
            "rows_loaded",
            "rows_joined",
            "missing_sdv",
            "missing_final_score_file",
            "unresolved_gamePk",
            "failed_final_score_join",
            "duplicate_game_id",
            "duplicate_gamePk",
            "leakage_rejections",
            "missing_dratings_prob_rows",
            "invalid_dratings_prob_rows",
            "rows_written",
        ]:
            _log(
                f"{key}={summary[key]}"
            )

        _log(
            f"WROTE {OUTPUT_FILE}"
        )

        print(
            "build_run_training_set complete. "
            f"rows_written={summary['rows_written']} "
            f"missing_sdv={summary['missing_sdv']} "
            f"missing_final_score_file={summary['missing_final_score_file']} "
            f"unresolved_gamePk={summary['unresolved_gamePk']} "
            f"failed_final_score_join={summary['failed_final_score_join']} "
            f"leakage_rejections={summary['leakage_rejections']}"
        )

    except Exception as exc:
        _log(
            f"FATAL: {exc}\n{traceback.format_exc()}",
            "ERROR",
        )

        for key in [
            "rows_loaded",
            "rows_joined",
            "missing_sdv",
            "missing_final_score_file",
            "unresolved_gamePk",
            "failed_final_score_join",
            "duplicate_game_id",
            "duplicate_gamePk",
            "leakage_rejections",
            "missing_dratings_prob_rows",
            "invalid_dratings_prob_rows",
            "rows_written",
        ]:
            _log(
                f"{key}={summary[key]}"
            )

        print(
            f"build_run_training_set failed: {exc}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()