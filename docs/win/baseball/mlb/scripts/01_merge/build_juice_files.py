#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/01_merge/build_juice_files.py

import glob
import math
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

INPUT_DIR = Path("docs/win/baseball/mlb/01_merge")
OUTPUT_DIR = Path("docs/win/baseball/mlb/02_juice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/baseball/mlb/errors/01_merge")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_juice_files.txt"

PROB_TOLERANCE = 1e-6

# Fixed, price-independent probability contraction. Retrospective game-level
# and selected-moneyline/run-line comparisons favored contraction; this is a
# conservative, reversible adjustment, not a certified live calibration fit.
# Leave totals and their modeled push probabilities unchanged.
BINARY_PROBABILITY_CALIBRATION_SLOPE = 0.50


def calibrate_binary_probabilities(home_probability, away_probability):
    """Contract complementary binary probabilities toward 0.5.

    Preserve the higher-probability side, strict complementarity, and all
    downstream probability/EV/Kelly identities. Do not change raw run means.
    """
    home = float(home_probability)
    away = float(away_probability)
    if (
        not np.isfinite(home)
        or not np.isfinite(away)
        or not 0.0 <= home <= 1.0
        or not 0.0 <= away <= 1.0
        or abs(home + away - 1.0) > PROB_TOLERANCE
    ):
        raise ValueError("invalid pre-calibration binary probabilities")
    adjusted_home = 0.5 + BINARY_PROBABILITY_CALIBRATION_SLOPE * (home - 0.5)
    return adjusted_home, 1.0 - adjusted_home

LEGACY_OFFICIAL_PROBABILITY_COLUMNS = [
    "home_normalized_prob_moneyline",
    "away_normalized_prob_moneyline",
    "home_normalized_prob_run_line",
    "away_normalized_prob_run_line",
    "over_normalized_prob_total",
    "under_normalized_prob_total",
]

RUN_PROJECTION_COLUMNS = [
    "dratings_home_projected_runs",
    "dratings_away_projected_runs",
    "dratings_total_projected_runs",
    "model_home_runs",
    "model_away_runs",
    "model_total_runs",
]

CONTEXT_COLS = [
    "gamePk",
    "home_team_id", "away_team_id", "venue_id",
    "roof_type", "turf_type",
    "home_pitcher_id", "away_pitcher_id",
    "home_pitcher_hand", "away_pitcher_hand",
    "home_sp_xwoba", "away_sp_xwoba",
    "home_sp_k_pct", "away_sp_k_pct",
    "home_sp_bb_pct", "away_sp_bb_pct",
    "home_sp_barrel_pct", "away_sp_barrel_pct",
    "home_sp_whiff_pct", "away_sp_whiff_pct",
    "home_sp_sample_flag", "away_sp_sample_flag",
    "home_lineup_xwoba", "home_lineup_barrel_pct", "home_lineup_hard_hit_pct",
    "home_lineup_k_pct", "home_lineup_bb_pct", "home_lineup_exit_velo",
    "home_lineup_frv", "home_lineup_brv", "home_catcher_framing",
    "home_low_sample_count", "home_n_left", "home_n_right", "home_n_switch",
    "away_lineup_xwoba", "away_lineup_barrel_pct", "away_lineup_hard_hit_pct",
    "away_lineup_k_pct", "away_lineup_bb_pct", "away_lineup_exit_velo",
    "away_lineup_frv", "away_lineup_brv", "away_catcher_framing",
    "away_low_sample_count", "away_n_left", "away_n_right", "away_n_switch",
    "park_factor", "park_wOBAcon", "park_xwOBAcon", "park_HR", "park_R",
    "park_factor_B", "park_wOBAcon_B", "park_xwOBAcon_B", "park_HR_B", "park_R_B",
    "weather_applicable", "weather_time",
    "temp_f", "wind_mph", "wind_dir",
    "precip_in", "humidity", "will_it_rain", "wind_blowing_out",
    "air_pressure_at_sea_level", "dew_point_f", "symbol_code",
    "home_batters_found", "away_batters_found",
    "home_sp_found", "away_sp_found",
    "sp_data_available", "lineup_data_available",
]

BASE_REQUIRED = [
    "game_id", "sport", "league", "game_date", "game_time", "home_team", "away_team",
    "away_run_line", "home_run_line", "total",
    "home_pitcher", "away_pitcher",
] + RUN_PROJECTION_COLUMNS + CONTEXT_COLS

MONEYLINE_REQUIRED_COLUMNS = BASE_REQUIRED + [
    "away_dk_moneyline_american", "home_dk_moneyline_american",
    "away_dk_moneyline_decimal", "home_dk_moneyline_decimal",
]

RUN_LINE_REQUIRED_COLUMNS = BASE_REQUIRED + [
    "away_dk_run_line_american", "home_dk_run_line_american",
    "away_dk_run_line_decimal", "home_dk_run_line_decimal",
]

TOTAL_REQUIRED_COLUMNS = BASE_REQUIRED + [
    "dk_total_over_american", "dk_total_under_american",
    "dk_total_over_decimal", "dk_total_under_decimal",
]


def _now():
    return datetime.now(UTC).isoformat()


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {msg}\n")


def american_to_decimal(odds):
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None

    if math.isnan(odds) or odds == 0:
        return None

    return 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))


def parse_slate_date_and_market(file_path: str):
    stem = Path(file_path).stem
    if stem.endswith("_mlb_moneyline"):
        return stem.replace("_mlb_moneyline", ""), "moneyline"
    if stem.endswith("_mlb_run_line"):
        return stem.replace("_mlb_run_line", ""), "run_line"
    if stem.endswith("_mlb_total"):
        return stem.replace("_mlb_total", ""), "total"
    return None, None


def duplicate_columns(columns):
    seen = set()
    dupes = []
    for col in columns:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)
    return dupes


def validate_no_duplicate_columns(df, label):
    dupes = duplicate_columns(list(df.columns))
    if dupes:
        raise ValueError(f"{label} duplicate columns: {dupes}")


def validate_schema(df, required_columns, label):
    validate_no_duplicate_columns(df, label)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")

    legacy = [
        c
        for c in LEGACY_OFFICIAL_PROBABILITY_COLUMNS
        if c in df.columns
    ]
    if legacy:
        raise ValueError(
            f"{label} contains obsolete official probability columns: {legacy}"
        )


def coerce_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _validate_prob_series(df, cols, label):
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = (
            values.isna()
            | ~np.isfinite(values)
            | (values < 0)
            | (values > 1)
        )

        if bad.any():
            sample = (
                df.loc[bad, ["game_id", col]]
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                f"{label} invalid probability column {col}; "
                f"bad_rows={int(bad.sum())}; sample={sample}"
            )


def _validate_pair(df, a_col, b_col, label):
    _validate_prob_series(df, [a_col, b_col], label)

    a = pd.to_numeric(df[a_col], errors="coerce")
    b = pd.to_numeric(df[b_col], errors="coerce")

    bad = ((a + b - 1.0).abs() > PROB_TOLERANCE)

    if bad.any():
        sample = (
            df.loc[bad, ["game_id", a_col, b_col]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{label} probability pair does not sum to 1 within "
            f"{PROB_TOLERANCE}; bad_rows={int(bad.sum())}; sample={sample}"
        )


def _validate_totals(df, label):
    cols = [
        "over_model_prob_total_win",
        "over_model_prob_total_loss",
        "under_model_prob_total_win",
        "under_model_prob_total_loss",
        "total_model_prob_push",
    ]

    _validate_prob_series(df, cols, label)

    probabilities = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in cols
    }

    identities = (
        (
            probabilities["over_model_prob_total_win"]
            + probabilities["over_model_prob_total_loss"]
            + probabilities["total_model_prob_push"]
            - 1.0
        ).abs(),
        (
            probabilities["under_model_prob_total_win"]
            + probabilities["under_model_prob_total_loss"]
            + probabilities["total_model_prob_push"]
            - 1.0
        ).abs(),
        (
            probabilities["under_model_prob_total_win"]
            - probabilities["over_model_prob_total_loss"]
        ).abs(),
        (
            probabilities["under_model_prob_total_loss"]
            - probabilities["over_model_prob_total_win"]
        ).abs(),
    )

    bad = identities[0] > PROB_TOLERANCE
    for difference in identities[1:]:
        bad |= difference > PROB_TOLERANCE

    if bad.any():
        sample_cols = ["game_id"] + cols
        sample = (
            df.loc[bad, sample_cols]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{label} totals probability contract failed; "
            f"bad_rows={int(bad.sum())}; sample={sample}"
        )


def write_csv_checked(df, out_path, market):
    validate_no_duplicate_columns(
        df,
        f"{out_path} output",
    )

    if market == "moneyline":
        _validate_pair(
            df,
            "home_model_prob_moneyline",
            "away_model_prob_moneyline",
            str(out_path),
        )

    elif market == "run_line":
        _validate_pair(
            df,
            "home_model_prob_run_line",
            "away_model_prob_run_line",
            str(out_path),
        )

    elif market == "total":
        _validate_totals(
            df,
            str(out_path),
        )

    else:
        raise ValueError(
            f"Unknown market {market}"
        )

    legacy = [
        c
        for c in LEGACY_OFFICIAL_PROBABILITY_COLUMNS
        if c in df.columns
    ]

    if legacy:
        raise ValueError(
            f"{out_path} contains obsolete official probability columns: "
            f"{legacy}"
        )

    df.to_csv(
        out_path,
        index=False,
    )


def moneyline_probabilities(
    model_home_runs,
    model_away_runs,
):
    p_home_raw = (
        1.0
        - skellam.cdf(
            0,
            model_home_runs,
            model_away_runs,
        )
    )

    p_away_raw = skellam.cdf(
        -1,
        model_home_runs,
        model_away_runs,
    )

    p_tie = skellam.pmf(
        0,
        model_home_runs,
        model_away_runs,
    )

    resolved = (
        p_home_raw
        + p_away_raw
    )

    if (
        not np.isfinite(resolved)
        or resolved <= 0
    ):
        raise ValueError(
            "invalid moneyline resolved probability mass"
        )

    return (
        p_home_raw / resolved,
        p_away_raw / resolved,
        p_tie,
    )


def run_line_probabilities(
    model_home_runs,
    model_away_runs,
    home_line,
    away_line,
):
    if (
        not np.isfinite(home_line)
        or not np.isfinite(away_line)
    ):
        raise ValueError(
            "missing run line"
        )

    if (
        abs(home_line + away_line)
        > PROB_TOLERANCE
    ):
        raise ValueError(
            "run lines are not complementary: "
            f"home={home_line} away={away_line}"
        )

    if sorted([
        round(home_line, 6),
        round(away_line, 6),
    ]) != [-1.5, 1.5]:
        raise ValueError(
            "unsupported run-line pair: "
            f"home={home_line} away={away_line}"
        )

    threshold = (
        math.floor(-home_line)
        + 1
    )

    p_home = (
        1.0
        - skellam.cdf(
            threshold - 1,
            model_home_runs,
            model_away_runs,
        )
    )

    p_away = 1.0 - p_home

    if (
        not np.isfinite(p_home)
        or not np.isfinite(p_away)
        or p_home < 0.0
        or p_home > 1.0
        or p_away < 0.0
        or p_away > 1.0
        or abs(p_home + p_away - 1.0) > PROB_TOLERANCE
    ):
        raise ValueError(
            "invalid raw Skellam run-line probabilities: "
            f"home={p_home} away={p_away}"
        )

    return (
        p_home,
        p_away,
    )


def totals_probabilities(
    model_home_runs,
    model_away_runs,
    total_line,
):
    if (
        not np.isfinite(model_home_runs)
        or not np.isfinite(model_away_runs)
    ):
        raise ValueError(
            "missing model run projection"
        )

    if (
        model_home_runs < 0
        or model_away_runs < 0
    ):
        raise ValueError(
            "negative model run projection"
        )

    if not np.isfinite(total_line):
        raise ValueError(
            "missing total line"
        )

    lambda_total = (
        model_home_runs
        + model_away_runs
    )

    frac = abs(
        total_line
        - round(total_line)
    )

    if frac < 1e-9:
        k = int(
            round(total_line)
        )

        p_under = poisson.cdf(
            k - 1,
            lambda_total,
        )

        p_push = poisson.pmf(
            k,
            lambda_total,
        )

        p_over = (
            1.0
            - poisson.cdf(
                k,
                lambda_total,
            )
        )

    elif abs(frac - 0.5) < 1e-9:
        k = math.floor(
            total_line
        )

        p_under = poisson.cdf(
            k,
            lambda_total,
        )

        p_push = 0.0

        p_over = (
            1.0
            - p_under
        )

    else:
        raise ValueError(
            f"unsupported total line: "
            f"{total_line}"
        )

    return (
        p_over,
        p_under,
        p_push,
    )


def _log_row_skip(
    summary,
    market,
    file_path,
    idx,
    row,
    error,
):
    game_id = row.get(
        "game_id",
        "",
    )

    game_date = row.get(
        "game_date",
        "",
    )

    away_team = row.get(
        "away_team",
        "",
    )

    home_team = row.get(
        "home_team",
        "",
    )

    log(
        f"ROW ISSUE SKIPPED | "
        f"market={market} | "
        f"file={file_path} | "
        f"idx={idx} | "
        f"game_id={game_id} | "
        f"game_date={game_date} | "
        f"away_team={away_team} | "
        f"home_team={home_team} | "
        f"issue={error}"
    )

    summary["row_issues"] += 1


def _filter_valid_run_projection_rows(
    df,
    file_path,
    market,
    summary,
):
    valid_indices = []

    for idx, row in df.iterrows():
        issues = []

        for col in RUN_PROJECTION_COLUMNS:
            value = row.get(col)

            if pd.isna(value):
                issues.append(
                    f"missing {col}"
                )
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                issues.append(
                    f"nonnumeric {col}={value}"
                )
                continue

            if not np.isfinite(
                numeric_value
            ):
                issues.append(
                    f"non-finite {col}={value}"
                )
                continue

            if numeric_value < 0:
                issues.append(
                    f"negative {col}={value}"
                )

        if not issues:
            try:
                model_home = float(
                    row["model_home_runs"]
                )
                model_away = float(
                    row["model_away_runs"]
                )
                model_total = float(
                    row["model_total_runs"]
                )

                if abs(
                    model_total
                    - (
                        model_home
                        + model_away
                    )
                ) > 1e-6:
                    issues.append(
                        "model_total_runs mismatch: "
                        f"home={model_home} "
                        f"away={model_away} "
                        f"total={model_total}"
                    )

            except Exception as exc:
                issues.append(
                    "model run projection "
                    f"validation failed: {exc}"
                )

        if issues:
            _log_row_skip(
                summary=summary,
                market=market,
                file_path=file_path,
                idx=idx,
                row=row,
                error="; ".join(issues),
            )
            continue

        valid_indices.append(idx)

    return df.loc[
        valid_indices
    ].copy()


def _prepare(
    file_path,
    required_columns,
    numeric_cols,
    market,
    summary,
):
    df = pd.read_csv(
        file_path
    )

    if df.empty:
        log(
            f"EMPTY FILE SKIPPED | "
            f"market={market} | "
            f"file={file_path}"
        )
        summary["empty"] += 1
        return df

    validate_schema(
        df,
        required_columns,
        str(file_path),
    )

    coerce_numeric(
        df,
        list(
            dict.fromkeys(
                RUN_PROJECTION_COLUMNS
                + numeric_cols
            )
        ),
    )

    df = (
        _filter_valid_run_projection_rows(
            df=df,
            file_path=file_path,
            market=market,
            summary=summary,
        )
    )

    return df


def process_moneyline(
    file_path,
    summary,
):
    df = _prepare(
        file_path=file_path,
        required_columns=MONEYLINE_REQUIRED_COLUMNS,
        numeric_cols=[
            "away_run_line",
            "home_run_line",
            "total",
            "away_dk_moneyline_american",
            "home_dk_moneyline_american",
            "away_dk_moneyline_decimal",
            "home_dk_moneyline_decimal",
        ],
        market="moneyline",
        summary=summary,
    )

    if df.empty:
        log(
            f"SKIPPED moneyline "
            f"{file_path}: no valid rows"
        )
        return

    valid_indices = []
    home_probs = []
    away_probs = []
    raw_home_probs = []
    raw_away_probs = []
    ties = []

    for i, r in df.iterrows():
        try:
            hp_raw, ap_raw, tp = (
                moneyline_probabilities(
                    r["model_home_runs"],
                    r["model_away_runs"],
                )
            )
            hp, ap = calibrate_binary_probabilities(hp_raw, ap_raw)

        except Exception as e:
            _log_row_skip(
                summary=summary,
                market="moneyline",
                file_path=file_path,
                idx=i,
                row=r,
                error=e,
            )
            continue

        valid_indices.append(i)
        home_probs.append(hp)
        away_probs.append(ap)
        raw_home_probs.append(hp_raw)
        raw_away_probs.append(ap_raw)
        ties.append(tp)

    if not valid_indices:
        log(
            f"SKIPPED moneyline "
            f"{file_path}: no valid rows"
        )
        return

    ml = df.loc[
        valid_indices
    ].copy()

    ml[
        "away_dk_decimal_moneyline"
    ] = ml[
        "away_dk_moneyline_american"
    ].apply(
        american_to_decimal
    )

    ml[
        "home_dk_decimal_moneyline"
    ] = ml[
        "home_dk_moneyline_american"
    ].apply(
        american_to_decimal
    )

    ml["home_model_prob_moneyline_raw"] = raw_home_probs
    ml["away_model_prob_moneyline_raw"] = raw_away_probs

    ml[
        "home_model_prob_moneyline"
    ] = home_probs

    ml[
        "away_model_prob_moneyline"
    ] = away_probs

    ml[
        "model_prob_moneyline_tie_raw"
    ] = ties

    ml[
        "home_model_fair_decimal_moneyline"
    ] = (
        1.0
        / ml[
            "home_model_prob_moneyline"
        ]
    )

    ml[
        "away_model_fair_decimal_moneyline"
    ] = (
        1.0
        / ml[
            "away_model_prob_moneyline"
        ]
    )

    slate_date, market = (
        parse_slate_date_and_market(
            file_path
        )
    )

    if (
        not slate_date
        or market != "moneyline"
    ):
        raise ValueError(
            f"FILENAME ERROR: "
            f"{file_path}"
        )

    out = (
        OUTPUT_DIR
        / f"{slate_date}_mlb_moneyline.csv"
    )

    write_csv_checked(
        ml,
        out,
        market,
    )

    log(
        f"WROTE {out} "
        f"({len(ml)} rows)"
    )

    summary["files_written"] += 1
    summary["rows_written"] += len(ml)


def process_run_line(
    file_path,
    summary,
):
    df = _prepare(
        file_path=file_path,
        required_columns=RUN_LINE_REQUIRED_COLUMNS,
        numeric_cols=[
            "away_run_line",
            "home_run_line",
            "total",
            "away_dk_run_line_american",
            "home_dk_run_line_american",
            "away_dk_run_line_decimal",
            "home_dk_run_line_decimal",
        ],
        market="run_line",
        summary=summary,
    )

    if df.empty:
        log(
            f"SKIPPED run_line "
            f"{file_path}: no valid rows"
        )
        return

    valid_indices = []
    home_probs = []
    away_probs = []
    raw_home_probs = []
    raw_away_probs = []

    for i, r in df.iterrows():
        try:
            hp_raw, ap_raw = (
                run_line_probabilities(
                    r["model_home_runs"],
                    r["model_away_runs"],
                    r["home_run_line"],
                    r["away_run_line"],
                )
            )
            hp, ap = calibrate_binary_probabilities(hp_raw, ap_raw)

        except Exception as e:
            _log_row_skip(
                summary=summary,
                market="run_line",
                file_path=file_path,
                idx=i,
                row=r,
                error=e,
            )
            continue

        valid_indices.append(i)
        home_probs.append(hp)
        away_probs.append(ap)
        raw_home_probs.append(hp_raw)
        raw_away_probs.append(ap_raw)

    if not valid_indices:
        log(
            f"SKIPPED run_line "
            f"{file_path}: no valid rows"
        )
        return

    rl = df.loc[
        valid_indices
    ].copy()

    rl[
        "home_dk_run_line_decimal"
    ] = rl[
        "home_dk_run_line_american"
    ].apply(
        american_to_decimal
    )

    rl[
        "away_dk_run_line_decimal"
    ] = rl[
        "away_dk_run_line_american"
    ].apply(
        american_to_decimal
    )

    rl["home_model_prob_run_line_raw"] = raw_home_probs
    rl["away_model_prob_run_line_raw"] = raw_away_probs

    rl[
        "home_model_prob_run_line"
    ] = home_probs

    rl[
        "away_model_prob_run_line"
    ] = away_probs

    rl[
        "home_model_fair_decimal_run_line"
    ] = (
        1.0
        / rl[
            "home_model_prob_run_line"
        ]
    )

    rl[
        "away_model_fair_decimal_run_line"
    ] = (
        1.0
        / rl[
            "away_model_prob_run_line"
        ]
    )

    slate_date, market = (
        parse_slate_date_and_market(
            file_path
        )
    )

    if (
        not slate_date
        or market != "run_line"
    ):
        raise ValueError(
            f"FILENAME ERROR: "
            f"{file_path}"
        )

    out = (
        OUTPUT_DIR
        / f"{slate_date}_mlb_run_line.csv"
    )

    write_csv_checked(
        rl,
        out,
        market,
    )

    log(
        f"WROTE {out} "
        f"({len(rl)} rows)"
    )

    summary["files_written"] += 1
    summary["rows_written"] += len(rl)


def process_total(
    file_path,
    summary,
):
    df = _prepare(
        file_path=file_path,
        required_columns=TOTAL_REQUIRED_COLUMNS,
        numeric_cols=[
            "away_run_line",
            "home_run_line",
            "total",
            "dk_total_over_american",
            "dk_total_under_american",
            "dk_total_over_decimal",
            "dk_total_under_decimal",
        ],
        market="total",
        summary=summary,
    )

    if df.empty:
        log(
            f"SKIPPED total "
            f"{file_path}: no valid rows"
        )
        return

    valid_indices = []
    over_win = []
    over_loss = []
    under_win = []
    under_loss = []
    pushes = []

    for i, r in df.iterrows():
        try:
            (
                p_over,
                p_under,
                p_push,
            ) = totals_probabilities(
                r["model_home_runs"],
                r["model_away_runs"],
                r["total"],
            )

        except Exception as e:
            _log_row_skip(
                summary=summary,
                market="total",
                file_path=file_path,
                idx=i,
                row=r,
                error=e,
            )
            continue

        valid_indices.append(i)
        over_win.append(p_over)
        over_loss.append(p_under)
        under_win.append(p_under)
        under_loss.append(p_over)
        pushes.append(p_push)

    if not valid_indices:
        log(
            f"SKIPPED total "
            f"{file_path}: no valid rows"
        )
        return

    tot = df.loc[
        valid_indices
    ].copy()

    tot[
        "dk_total_over_decimal"
    ] = tot[
        "dk_total_over_american"
    ].apply(
        american_to_decimal
    )

    tot[
        "dk_total_under_decimal"
    ] = tot[
        "dk_total_under_american"
    ].apply(
        american_to_decimal
    )

    tot[
        "over_model_prob_total_win"
    ] = over_win

    tot[
        "over_model_prob_total_loss"
    ] = over_loss

    tot[
        "under_model_prob_total_win"
    ] = under_win

    tot[
        "under_model_prob_total_loss"
    ] = under_loss

    tot[
        "total_model_prob_push"
    ] = pushes

    # Push-aware fair decimal:
    # 1 + (p_loss / p_win).
    # For half-run totals, p_push == 0
    # and this reduces to 1 / p_win.
    tot[
        "fair_total_over_decimal"
    ] = (
        1.0
        + tot[
            "over_model_prob_total_loss"
        ]
        / tot[
            "over_model_prob_total_win"
        ]
    )

    tot[
        "fair_total_under_decimal"
    ] = (
        1.0
        + tot[
            "under_model_prob_total_loss"
        ]
        / tot[
            "under_model_prob_total_win"
        ]
    )

    slate_date, market = (
        parse_slate_date_and_market(
            file_path
        )
    )

    if (
        not slate_date
        or market != "total"
    ):
        raise ValueError(
            f"FILENAME ERROR: "
            f"{file_path}"
        )

    out = (
        OUTPUT_DIR
        / f"{slate_date}_mlb_total.csv"
    )

    write_csv_checked(
        tot,
        out,
        market,
    )

    log(
        f"WROTE {out} "
        f"({len(tot)} rows)"
    )

    summary["files_written"] += 1
    summary["rows_written"] += len(tot)


def main():
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== build_juice_files RUN "
            f"{_now()} ===\n"
        )

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "empty": 0,
        "schema_errors": 0,
        "row_issues": 0,
        "errors": 0,
    }

    log(
        "MODEL PROBABILITIES ARE PRICE-INDEPENDENT: "
        "sportsbook odds are not probability inputs"
    )
    log(
        "MONEYLINE AND RUN-LINE PROBABILITIES USE FIXED 0.50 "
        "CONTRACTION TOWARD EVEN; TOTALS ARE UNCHANGED"
    )

    for f in OUTPUT_DIR.glob(
        "*.csv"
    ):
        f.unlink()

    try:
        groups = [
            (
                "moneyline",
                sorted(
                    glob.glob(
                        str(
                            INPUT_DIR
                            / "*_mlb_moneyline.csv"
                        )
                    )
                ),
                process_moneyline,
            ),
            (
                "run_line",
                sorted(
                    glob.glob(
                        str(
                            INPUT_DIR
                            / "*_mlb_run_line.csv"
                        )
                    )
                ),
                process_run_line,
            ),
            (
                "total",
                sorted(
                    glob.glob(
                        str(
                            INPUT_DIR
                            / "*_mlb_total.csv"
                        )
                    )
                ),
                process_total,
            ),
        ]

        for (
            market,
            files,
            processor,
        ) in groups:
            log(
                f"{market} files: "
                f"{len(files)}"
            )

            for file_path in files:
                try:
                    processor(
                        file_path,
                        summary,
                    )

                except ValueError as e:
                    log(
                        f"SCHEMA/CONTRACT ERROR "
                        f"{market} "
                        f"{file_path}: "
                        f"{e}\n"
                        f"{traceback.format_exc()}"
                    )
                    summary[
                        "schema_errors"
                    ] += 1

                except Exception as e:
                    log(
                        f"ERROR "
                        f"{market} "
                        f"{file_path}: "
                        f"{e}\n"
                        f"{traceback.format_exc()}"
                    )
                    summary[
                        "errors"
                    ] += 1

        status = (
            "SUCCESS"
            if (
                summary["errors"] == 0
                and summary["schema_errors"] == 0
            )
            else "COMPLETED WITH ERRORS"
        )

        log(
            "--- SUMMARY ---"
        )

        for key, value in summary.items():
            log(
                f"{key}={value}"
            )

        log(
            f"STATUS: {status}"
        )

        if (
            summary["errors"] > 0
            or summary["schema_errors"] > 0
        ):
            print(
                "build_juice_files completed "
                "with errors. "
                f"errors={summary['errors']} "
                f"schema_errors="
                f"{summary['schema_errors']} "
                f"row_issues="
                f"{summary['row_issues']}"
            )
            sys.exit(1)

        print(
            "build_juice_files complete. "
            f"files_written="
            f"{summary['files_written']} "
            f"rows_written="
            f"{summary['rows_written']} "
            f"row_issues="
            f"{summary['row_issues']} "
            f"schema_errors="
            f"{summary['schema_errors']} "
            f"errors="
            f"{summary['errors']}"
        )

    except Exception as e:
        log(
            f"FATAL ERROR: "
            f"{e}\n"
            f"{traceback.format_exc()}"
        )
        log(
            "STATUS: FAILED"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()