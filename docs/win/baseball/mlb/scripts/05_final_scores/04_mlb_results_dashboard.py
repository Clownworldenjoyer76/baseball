#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/04_mlb_results_dashboard.py
#
# Builds the MLB dashboard data (read live by the React Baseball/MLB pages) with two source views:
#   - MLB                -> morning graded/report tree
#   - MLB · WITH LINEUPS -> pregame/non-morning graded/report tree
#
# Inputs:
#   docs/win/baseball/mlb/05_final_scores/morning/mlb_summary_overall.csv
#   docs/win/baseball/mlb/05_final_scores/morning/reports/{overview,moneyline,run_line,totals}/*.csv
#   docs/win/baseball/mlb/05_final_scores/mlb_summary_overall.csv
#   docs/win/baseball/mlb/05_final_scores/reports/{overview,moneyline,run_line,totals}/*.csv
#
# Output:
#   frontend/data/dashboards/baseball.json  ({"generated_at": <UTC ISO>, "data": <feed payload>})
#
# Log:
#   docs/win/baseball/mlb/errors/05_final_scores/04_mlb_results_dashboard.txt

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import json
import math
import os
import traceback

import pandas as pd

BASE = Path("docs/win/baseball/mlb/05_final_scores")
OUTPUT_FILE = Path("frontend/data/dashboards/baseball.json")
ERROR_DIR = Path("docs/win/baseball/mlb/errors/05_final_scores")
LOG_FILE = ERROR_DIR / "04_mlb_results_dashboard.txt"

FEEDS = {
    "mlb": {
        "label": "MLB",
        "root": BASE / "morning",
    },
    "mlb_lineups": {
        "label": "MLB WITH LINEUPS",
        "root": BASE,
    },
}

MARKETS = {
    "moneyline": {
        "label": "Moneyline",
        "directory": "moneyline",
        "file_key": "moneyline",
        "dimensions": ["ev", "odds", "kelly", "win_prob"],
    },
    "run_line": {
        "label": "Run Line",
        "directory": "run_line",
        "file_key": "run_line",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "side"],
    },
    "total": {
        "label": "Total",
        "directory": "totals",
        "file_key": "total",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "total_range", "side"],
    },
}

ERROR_DIR.mkdir(parents=True, exist_ok=True)

RUN_STARTED = datetime.now(UTC)
RUN_COUNTS = {
    "warning": 0,
    "error": 0,
    "input_files": 0,
    "input_rows": 0,
    "output_files": 0,
    "output_rows": 0,
}
INPUT_FILES_SEEN: set[str] = set()

with LOG_FILE.open("w", encoding="utf-8") as startup_log:
    startup_log.write("=== 04_mlb_results_dashboard ===\n")
    startup_log.write(f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def log(level: str, message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level} | {message}\n")


def warn(message: str) -> None:
    RUN_COUNTS["warning"] += 1
    log("WARNING", message)


def error(message: str) -> None:
    RUN_COUNTS["error"] += 1
    log("ERROR", message)


def log_input(path: Path, rows: int, exists: bool) -> None:
    key = str(path)
    if key in INPUT_FILES_SEEN:
        return
    INPUT_FILES_SEEN.add(key)
    RUN_COUNTS["input_files"] += 1
    RUN_COUNTS["input_rows"] += rows
    log("INFO", f"INPUT | file={path} | exists={int(exists)} | rows={rows}")


def log_output(path: Path, rows: int, bytes_written: int) -> None:
    RUN_COUNTS["output_files"] += 1
    RUN_COUNTS["output_rows"] += rows
    log("INFO", f"OUTPUT | file={path} | rows={rows} | bytes={bytes_written}")


def finish(status: str) -> None:
    ended = datetime.now(UTC)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(
            f"INPUT_SUMMARY | files={RUN_COUNTS['input_files']} | "
            f"rows={RUN_COUNTS['input_rows']}\n"
        )
        f.write(
            f"OUTPUT_SUMMARY | files={RUN_COUNTS['output_files']} | "
            f"rows={RUN_COUNTS['output_rows']}\n"
        )
        f.write(f"WARNING_COUNT: {RUN_COUNTS['warning']}\n")
        f.write(f"ERROR_COUNT: {RUN_COUNTS['error']}\n")
        f.write(f"END_TIMESTAMP_UTC: {ended.isoformat()}\n")
        f.write(f"STATUS: {status}\n")


def safe_read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        log_input(path, 0, False)
        if required:
            warn(f"Required dashboard input missing: {path}")
        else:
            log("INFO", f"Optional dashboard input missing: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        log_input(path, len(df), True)
        return df
    except Exception as exc:
        log_input(path, 0, True)
        warn(f"Unable to read dashboard input {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return [
        {key: _clean_value(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def first_row_dict(df: pd.DataFrame) -> dict:
    rows = df_to_records(df.head(1))
    return rows[0] if rows else {}


def collect_feed_data(feed_key: str) -> dict:
    cfg = FEEDS[feed_key]
    root: Path = cfg["root"]
    reports = root / "reports"
    overview = reports / "overview"

    data = {
        "feed": feed_key,
        "label": cfg["label"],
        "headline": first_row_dict(
            safe_read(overview / "mlb_summary_overall.csv", required=True)
        ),
        "by_market_summary": df_to_records(
            safe_read(root / "mlb_summary_overall.csv", required=True)
        ),
        "overview": {
            "by_market": df_to_records(safe_read(overview / "mlb_summary_by_market.csv")),
            "by_side_group": df_to_records(safe_read(overview / "mlb_summary_by_side_group.csv")),
            "by_date": df_to_records(safe_read(overview / "mlb_summary_by_date.csv")),
            "by_day_night": df_to_records(safe_read(overview / "mlb_summary_by_day_night.csv")),
            "by_low_confidence": df_to_records(
                safe_read(overview / "mlb_summary_by_low_confidence.csv")
            ),
        },
        "markets": {},
    }

    for market_key, market_cfg in MARKETS.items():
        market_dir = reports / market_cfg["directory"]
        file_key = market_cfg["file_key"]
        market_data = {"by": {}, "by_side": {}}

        for dimension in market_cfg["dimensions"]:
            base_name = f"mlb_{file_key}_by_{dimension}"
            market_data["by"][dimension] = df_to_records(
                safe_read(market_dir / f"{base_name}.csv")
            )
            market_data["by_side"][dimension] = df_to_records(
                safe_read(market_dir / f"{base_name}_home_away_summary.csv")
            )

        data["markets"][market_key] = market_data

    return data


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data) -> int:
    document = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": json_safe(data),
    }
    content = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)
    return len(content.encode("utf-8"))


def run() -> None:
    payload = {feed_key: collect_feed_data(feed_key) for feed_key in FEEDS}
    bytes_written = write_json(OUTPUT_FILE, payload)
    log_output(OUTPUT_FILE, 1, bytes_written)
    log("INFO", f"dashboard data -> {OUTPUT_FILE}")


def main() -> None:
    status = "FAILED"
    try:
        run()
        status = "SUCCESS"
    except Exception as exc:
        error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        trace = traceback.format_exc()
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(trace)
            if not trace.endswith("\n"):
                f.write("\n")
        raise
    finally:
        finish(status)


if __name__ == "__main__":
    main()
