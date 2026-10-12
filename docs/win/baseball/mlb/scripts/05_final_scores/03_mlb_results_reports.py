#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/03_mlb_results_reports.py

import shutil
import pandas as pd
from pathlib import Path

###############################################################
######################## PATH CONFIG ##########################
###############################################################

INPUT_FILE = Path("docs/win/baseball/mlb/05_final_scores/intermediate/work_mlb.csv")

SUMMARY_DIR  = Path("docs/win/baseball/mlb/05_final_scores")
REPORTS_DIR  = Path("docs/win/baseball/mlb/05_final_scores/reports")
OVERVIEW_DIR = Path("docs/win/baseball/mlb/05_final_scores/reports/overview")
ML_DIR       = Path("docs/win/baseball/mlb/05_final_scores/reports/moneyline")
RL_DIR       = Path("docs/win/baseball/mlb/05_final_scores/reports/run_line")
TOT_DIR      = Path("docs/win/baseball/mlb/05_final_scores/reports/totals")

for startup_dir in [SUMMARY_DIR, OVERVIEW_DIR, ML_DIR, RL_DIR, TOT_DIR]:
    startup_dir.mkdir(parents=True, exist_ok=True)

LEAGUE = "MLB"

###############################################################
######################## HELPERS ##############################
###############################################################

def clear_report_outputs():
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)

    for d in [OVERVIEW_DIR, ML_DIR, RL_DIR, TOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def to_float(value):
    try:
        v = float(value)
        import math
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def units_won(odds, result):
    result = str(result).strip().title()

    if result == "Push":
        return 0.0

    if result == "Loss":
        return -1.0

    if result != "Win":
        return None

    odds = to_float(odds)
    if odds is None:
        return None

    return odds / 100.0 if odds >= 0 else 100.0 / abs(odds)


def write_metric_definitions():
    definitions = pd.DataFrame([
        {
            "metric": "Win_Pct",
            "definition": "wins / (wins + losses)",
            "push_handling": "pushes excluded from denominator",
        },
        {
            "metric": "Win_Pct_All_Bets",
            "definition": "wins / (wins + losses + pushes)",
            "push_handling": "pushes included in denominator",
        },
        {
            "metric": "ROI_Excluding_Pushes",
            "definition": "units / (wins + losses)",
            "push_handling": "pushes excluded from denominator",
        },
        {
            "metric": "ROI_Including_Pushes",
            "definition": "units / (wins + losses + pushes)",
            "push_handling": "pushes included in denominator",
        },
    ])

    write_csv(definitions, OVERVIEW_DIR / "mlb_report_metric_definitions.csv")


###############################################################
######################## ENRICH ###############################
###############################################################

def enrich(df):
    work = df.copy()

    normalizers = {
        "market_type": "lower",
        "bet_side": "lower",
        "bet_result": "title",
    }

    for column, operation in normalizers.items():
        values = (
            work[column]
            .astype(str)
            .str.strip()
        )
        work[column] = (
            values.str.lower()
            if operation == "lower"
            else values.str.title()
        )

    if "side_group" not in work.columns:
        def _sg(row):
            mt = row["market_type"]
            bs = row["bet_side"]

            if mt in {"moneyline", "run_line"}:
                return (
                    "HOME"
                    if bs == "home"
                    else (
                        "AWAY"
                        if bs == "away"
                        else ""
                    )
                )

            if mt == "total":
                return (
                    "OVER"
                    if bs == "over"
                    else (
                        "UNDER"
                        if bs == "under"
                        else ""
                    )
                )

            return ""

        work["side_group"] = work.apply(
            _sg,
            axis=1,
        )

    work["bet_units"] = work.apply(
        lambda row: units_won(
            row.get("dk_odds_american"),
            row["bet_result"],
        ),
        axis=1,
    )

    return work


###############################################################
######################## AGGREGATE ############################
###############################################################

def build_metric_row(sub):
    wins   = int((sub["bet_result"] == "Win").sum())
    losses = int((sub["bet_result"] == "Loss").sum())
    pushes = int((sub["bet_result"] == "Push").sum())

    bets_excluding_pushes = wins + losses
    bets_including_pushes = wins + losses + pushes

    win_pct = (
        round(wins / bets_excluding_pushes, 4)
        if bets_excluding_pushes > 0
        else 0.0
    )

    win_pct_all_bets = (
        round(wins / bets_including_pushes, 4)
        if bets_including_pushes > 0
        else 0.0
    )

    unit_vals = pd.to_numeric(sub["bet_units"], errors="coerce").dropna()
    total_units = round(float(unit_vals.sum()), 4) if not unit_vals.empty else 0.0

    roi_excluding_pushes = (
        round(total_units / bets_excluding_pushes, 4)
        if bets_excluding_pushes > 0
        else 0.0
    )

    roi_including_pushes = (
        round(total_units / bets_including_pushes, 4)
        if bets_including_pushes > 0
        else 0.0
    )

    ev_vals = pd.to_numeric(sub["ev"], errors="coerce").dropna()
    avg_ev = round(float(ev_vals.mean()), 4) if not ev_vals.empty else None

    od_vals = pd.to_numeric(sub["dk_odds_american"], errors="coerce").dropna()
    avg_odds = round(float(od_vals.mean()), 1) if not od_vals.empty else None

    return {
        "Win": wins,
        "Loss": losses,
        "Push": pushes,
        "Total": bets_including_pushes,
        "bets_excluding_pushes": bets_excluding_pushes,
        "bets_including_pushes": bets_including_pushes,
        "Win_Pct": win_pct,
        "Win_Pct_All_Bets": win_pct_all_bets,
        "units": total_units,
        "ROI_Excluding_Pushes": roi_excluding_pushes,
        "ROI_Including_Pushes": roi_including_pushes,
        "avg_ev": avg_ev,
        "avg_odds": avg_odds,
    }


def empty_metric_columns(prefix_cols):
    return prefix_cols + [
        "Win",
        "Loss",
        "Push",
        "Total",
        "bets_excluding_pushes",
        "bets_including_pushes",
        "Win_Pct",
        "Win_Pct_All_Bets",
        "units",
        "ROI_Excluding_Pushes",
        "ROI_Including_Pushes",
        "avg_ev",
        "avg_odds",
    ]


def agg(df, group_cols, variable_label=None):
    """
    Win_Pct excludes pushes:
        wins / (wins + losses)

    Win_Pct_All_Bets includes pushes:
        wins / (wins + losses + pushes)

    ROI_Excluding_Pushes:
        units / (wins + losses)

    ROI_Including_Pushes:
        units / (wins + losses + pushes)
    """
    prefix_cols = ["league", "market_type"] + (["variable"] if variable_label else [])

    if df.empty:
        return pd.DataFrame(columns=empty_metric_columns(prefix_cols))

    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}
        for i, col in enumerate(group_cols):
            label = "variable" if (variable_label and i == len(group_cols) - 1) else col
            row[label] = keys[i]

        row.update(build_metric_row(sub))
        rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = list(out.columns[:len(group_cols)])
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def agg_simple(df, group_cols):
    """league + market_type tally — used for top-level summary."""
    if df.empty:
        return pd.DataFrame(columns=empty_metric_columns(["league", "market_type"]))

    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(build_metric_row(sub))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


###############################################################
###################### REPORT BUILDERS ########################
###############################################################

def build_top_summary(df):
    df["league"] = LEAGUE
    out = agg_simple(df, ["league", "market_type"])
    write_csv(out, SUMMARY_DIR / "mlb_summary_overall.csv")


def build_overview(df):
    df["league"] = LEAGUE

    write_metric_definitions()

    write_csv(agg(df, ["league"]), OVERVIEW_DIR / "mlb_summary_overall.csv")

    write_csv(
        agg(df, ["league", "market_type"], variable_label="market_type"),
        OVERVIEW_DIR / "mlb_summary_by_market.csv"
    )

    write_csv(
        agg(df, ["league", "side_group"], variable_label="side_group"),
        OVERVIEW_DIR / "mlb_summary_by_side_group.csv"
    )

    date_df = agg(df, ["league", "game_date"], variable_label="game_date")
    date_df = date_df.sort_values("variable")
    date_df["cumulative_units"] = date_df["units"].cumsum().round(4)
    write_csv(date_df, OVERVIEW_DIR / "mlb_summary_by_date.csv")

    if "day_night" in df.columns:
        write_csv(
            agg(df, ["league", "day_night"], variable_label="day_night"),
            OVERVIEW_DIR / "mlb_summary_by_day_night.csv"
        )

    if "low_confidence" in df.columns:
        write_csv(
            agg(df, ["league", "low_confidence"], variable_label="low_confidence"),
            OVERVIEW_DIR / "mlb_summary_by_low_confidence.csv"
        )

    log_cols = [
        "game_date",
        "game_id",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "line",
        "dk_odds_american",
        "ev",
        "kelly",
        "model_prob",
        "low_confidence",
        "day_night",
        "bet_result",
        "bet_units",
        "gamePk",
        "gameNumber",
        "game_status",
    ]

    available = [c for c in log_cols if c in df.columns]
    write_csv(df[available].copy(), OVERVIEW_DIR / "mlb_bet_log.csv")


def write_market_bucket(
    src,
    bucket_col,
    fname,
    output_dir,
    home_away=False,
):
    if src.empty or bucket_col not in src.columns:
        return

    bucketed = src[
        src[bucket_col] != "UNBUCKETED"
    ].copy()

    group_columns = [
        "league",
        "market_type",
    ]
    if home_away:
        group_columns.append(
            "side_group"
        )
    group_columns.append(
        bucket_col
    )

    result = agg(
        bucketed,
        group_columns,
        variable_label=bucket_col,
    )
    write_csv(
        result,
        output_dir / fname,
    )


def build_market_reports(
    df,
    market_type,
    reports,
    output_dir,
    summary_sides,
):
    market_rows = df[
        df["market_type"] == market_type
    ].copy()

    if market_rows.empty:
        return

    market_rows["league"] = LEAGUE

    for bucket_col, fname in reports:
        write_market_bucket(
            market_rows,
            bucket_col,
            fname,
            output_dir,
        )

    summary_rows = market_rows[
        market_rows["side_group"].isin(
            summary_sides
        )
    ]

    for bucket_col, fname in reports:
        write_market_bucket(
            summary_rows,
            bucket_col,
            fname.replace(
                ".csv",
                "_home_away_summary.csv",
            ),
            output_dir,
            home_away=True,
        )


def build_moneyline(df):
    reports = (
        ("ev_bucket", "mlb_moneyline_by_ev.csv"),
        ("odds_bucket", "mlb_moneyline_by_odds.csv"),
        ("kelly_bucket", "mlb_moneyline_by_kelly.csv"),
        ("win_prob_bucket", "mlb_moneyline_by_win_prob.csv"),
    )
    build_market_reports(
        df,
        "moneyline",
        reports,
        ML_DIR,
        ("HOME", "AWAY"),
    )


def build_run_line(df):
    reports = (
        ("ev_bucket", "mlb_run_line_by_ev.csv"),
        ("odds_bucket", "mlb_run_line_by_odds.csv"),
        ("kelly_bucket", "mlb_run_line_by_kelly.csv"),
        ("win_prob_bucket", "mlb_run_line_by_win_prob.csv"),
        ("run_line_side", "mlb_run_line_by_side.csv"),
    )
    build_market_reports(
        df,
        "run_line",
        reports,
        RL_DIR,
        ("HOME", "AWAY"),
    )


def build_totals(df):
    reports = (
        ("ev_bucket", "mlb_total_by_ev.csv"),
        ("odds_bucket", "mlb_total_by_odds.csv"),
        ("kelly_bucket", "mlb_total_by_kelly.csv"),
        ("win_prob_bucket", "mlb_total_by_win_prob.csv"),
        ("total_range_bucket", "mlb_total_by_total_range.csv"),
        ("side_group", "mlb_total_by_side.csv"),
    )
    build_market_reports(
        df,
        "total",
        reports,
        TOT_DIR,
        ("OVER", "UNDER"),
    )


###############################################################
######################## MAIN #################################
###############################################################

def run(completion_label: str = "MLB reports"):
    clear_report_outputs()

    if not INPUT_FILE.exists():
        print(f"ERROR: input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, dtype=str)

    # Morning historical reports must never include bets outside the CURRENT
    # markets_AM.yaml bands, even if an older intermediate CSV remains.
    # Use the same refiltering implementation as morning grading.
    if SUMMARY_DIR.name.lower() == "morning":
        from importlib.util import module_from_spec, spec_from_file_location
        morning_grade_path = Path(__file__).with_name("01_mlb_results_grade_AM.py")
        spec = spec_from_file_location("mlb_morning_report_refilter", morning_grade_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load morning refilter: {morning_grade_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        old_rows = len(df)
        df = module.refilter_historical_bets(df)
        print(f"MLB morning report refilter | before={old_rows} "
              f"after={len(df)} excluded={old_rows - len(df)}")

    df = enrich(df)

    build_top_summary(df)
    build_overview(df)
    build_moneyline(df)
    build_run_line(df)
    build_totals(df)

    print(f"{completion_label} complete.")


if __name__ == "__main__":
    run()
