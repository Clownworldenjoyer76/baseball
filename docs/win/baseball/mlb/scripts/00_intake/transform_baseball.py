#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/transform_baseball.py

import csv
import os
import tempfile
import traceback
import json
from datetime import datetime
from pathlib import Path

from intake_log import make_logger

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "transform_baseball.txt"
log = make_logger(LOG_FILE, "transform_baseball")

RAW_DIR = Path("docs/win/baseball/mlb/00_intake/drat_raw")
PRED_DIR = Path("docs/win/baseball/mlb/00_intake/predictions")
TEAM_MAP_FILE = Path("docs/win/baseball/mlb/maps/team_map_mlb.csv")
PRED_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_HEADER = [
    "game_id", "sport", "league", "game_date", "game_time",
    "home_team", "away_team", "home_pitcher", "away_pitcher",
    "home_prob", "away_prob", "away_projected_runs",
    "home_projected_runs", "total_projected_runs",
]


def parse_datetime(dt_str):
    dt = datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M %p")
    return dt, dt.strftime("%Y_%m_%d"), dt.strftime("%I:%M %p")


def clean_team(team_str):
    return team_str.split("(")[0].strip()


def pct_to_decimal(value):
    return str(round(float(value.replace("%", "")) / 100, 3))


def is_future_game(row):
    return isinstance(row, list) and len(row) == 11


def is_completed_game(row):
    return isinstance(row, list) and len(row) == 8


def is_summary_row(row):
    return isinstance(row, list) and bool(row) and str(row[0]).strip() in {"Sportsbooks", "DRatings"}


def load_team_aliases():
    """Match raw teams to existing CSVs normalized by name_normalization.py."""
    aliases = {}
    if not TEAM_MAP_FILE.exists():
        log(f"Team map missing; using literal team names for matching: {TEAM_MAP_FILE}")
        return aliases
    with TEAM_MAP_FILE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"league", "alias", "canonical_team"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid MLB team map: {TEAM_MAP_FILE}")
        for item in reader:
            if str(item.get("league", "")).strip().casefold() != "mlb":
                continue
            alias = str(item.get("alias", "")).strip().casefold()
            canonical = str(item.get("canonical_team", "")).strip().casefold()
            if alias and canonical:
                if alias in aliases and aliases[alias] != canonical:
                    raise ValueError(f"Conflicting MLB team alias: {alias}")
                aliases[alias] = canonical
    return aliases


def prediction_key(record, aliases):
    """Use time as part of game identity so doubleheaders remain distinct."""
    date = str(record.get("game_date", "")).strip()
    time_value = str(record.get("game_time", "")).strip()
    try:
        date = datetime.strptime(date, "%Y_%m_%d").strftime("%Y_%m_%d")
        time_value = datetime.strptime(time_value, "%I:%M %p").strftime("%I:%M %p")
    except ValueError as exc:
        raise ValueError(f"Invalid prediction game date/time: {record}") from exc

    home = str(record.get("home_team", "")).strip().casefold()
    away = str(record.get("away_team", "")).strip().casefold()
    if not home or not away:
        raise ValueError(f"Missing prediction team: {record}")
    return date, time_value, aliases.get(home, home), aliases.get(away, away)


def _future_prediction_row(row, game_date, game_time, home_team, away_team):
    try:
        pitchers = row[2].split("\n")
        probs = row[3].split("\n")
        runs = row[6].split("\n")
        return dict(zip(PREDICTION_HEADER, [
            "", "baseball", "mlb", game_date, game_time,
            home_team, away_team,
            pitchers[1].strip() if len(pitchers) > 1 else "",
            pitchers[0].strip(),
            pct_to_decimal(probs[1]) if len(probs) > 1 else "",
            pct_to_decimal(probs[0]),
            runs[0].strip(),
            runs[1].strip() if len(runs) > 1 else "", row[7],
        ]))
    except (IndexError, AttributeError, TypeError, ValueError):
        return None


def _transform_raw_row(row):
    if not isinstance(row, list) or len(row) < 2:
        return "skip", None, None
    if is_summary_row(row):
        return "summary", None, None
    try:
        _, date, game_time = parse_datetime(row[0])
        teams = row[1].split("\n")
        if len(teams) < 2:
            return "skip", None, None
        away = clean_team(teams[0])
        home = clean_team(teams[1])
    except (IndexError, AttributeError, TypeError, ValueError):
        return "parse_error", None, None
    if is_future_game(row):
        record = _future_prediction_row(row, date, game_time, home, away)
        return ("future", date, record) if record is not None else ("parse_error", None, None)
    if is_completed_game(row):
        return "completed", None, None
    return "unknown", None, None


def trusted_raw_file(file_path):
    path = Path(file_path).resolve()
    if not path.is_relative_to(RAW_DIR.resolve()) or not path.name.endswith("_mlb_raw.json"):
        raise ValueError(f"Refusing untrusted DRatings raw input: {file_path}")
    return path


def trusted_prediction_file(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PRED_DIR.resolve()) or path.suffix.lower() != ".csv":
        raise ValueError(f"Refusing untrusted prediction output: {path}")
    return path


def load_existing_predictions(merged, aliases):
    """Existing predictions always win over later snapshots."""
    for source in sorted(PRED_DIR.glob("*_MLB.csv")):
        path = trusted_prediction_file(source)
        expected_date = path.name[:-len("_MLB.csv")]
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PREDICTION_HEADER:
                raise ValueError(f"Unexpected prediction CSV schema; refusing overwrite: {path}")
            for line_number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed prediction CSV: {path}:{line_number}")
                key = prediction_key(row, aliases)
                if key[0] != expected_date:
                    raise ValueError(f"Prediction date differs from filename: {path}:{line_number}")
                if key in merged and merged[key] != row:
                    raise ValueError(f"Conflicting existing predictions; refusing overwrite: {path}:{line_number}")
                merged.setdefault(key, row)
    log(f"Existing predictions protected: {len(merged)}")


def process_file(file_path, merged, aliases):
    path = trusted_raw_file(file_path)
    log(f"Processing {path.name}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected raw JSON format: {path}")

    counters = {"future": 0, "completed": 0, "summary": 0,
                "parse_error": 0, "unknown": 0, "skip": 0, "preserved": 0}
    for row in raw:
        status, _, prediction = _transform_raw_row(row)
        counters[status] += 1
        if status != "future":
            continue
        key = prediction_key(prediction, aliases)
        if key in merged:
            counters["preserved"] += 1
            continue
        merged[key] = prediction

    log(f"  {counters}")


def write_csv(path, rows, files_written):
    path = trusted_prediction_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_rows = None
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PREDICTION_HEADER:
                raise ValueError(f"Unexpected prediction CSV schema: {path}")
            current_rows = list(reader)
    if current_rows == rows:
        log(f"UNCHANGED predictions -> {path} ({len(rows)} rows)")
        return

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_HEADER)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    files_written.append((str(path), len(rows)))
    log(f"WROTE predictions -> {path} ({len(rows)} rows)")


def main():
    files_written = []
    try:
        aliases = load_team_aliases()
        merged = {}
        load_existing_predictions(merged, aliases)
        raw_files = sorted(RAW_DIR.glob("*_mlb_raw.json"))
        log(f"Raw files found: {len(raw_files)}")
        log("Intake writes predictions only; completed rows remain in raw JSON for final scores")
        for path in raw_files:
            process_file(path, merged, aliases)

        by_date = {}
        for key, record in merged.items():
            by_date.setdefault(key[0], []).append(record)
        for date, rows in sorted(by_date.items()):
            write_csv(PRED_DIR / f"{date}_MLB.csv", rows, files_written)

        log("--- SUMMARY ---")
        log(f"Raw files processed: {len(raw_files)}")
        log(f"Files written: {len(files_written)}")
        log(f"Predictions preserved or recovered: {len(merged)}")
        log("STATUS: SUCCESS")
    except Exception as exc:
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise
    print("Baseball transform complete.")


if __name__ == "__main__":
    main()
