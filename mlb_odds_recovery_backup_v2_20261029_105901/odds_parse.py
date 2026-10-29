# docs/win/baseball/mlb/scripts/00_intake/odds_parse.py

import sys
import json
import csv
import traceback
from pathlib import Path

from intake_log import make_logger
from datetime import datetime
from zoneinfo import ZoneInfo

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "odds_parse.txt"

log = make_logger(LOG_FILE, "odds_parse")

# -----------------------
# INPUT HANDLING
# -----------------------

if len(sys.argv) > 1:
    INPUT_PATH = Path(sys.argv[1])
else:
    INPUT_PATH = Path("docs/win/baseball/mlb/odds")

if not INPUT_PATH.exists():
    log(f"FATAL ERROR: Input path does not exist: {INPUT_PATH}")
    log("STATUS: FAILED")
    raise FileNotFoundError(f"Input path does not exist: {INPUT_PATH}")

# -----------------------
# TIME CONVERSION
# -----------------------
def utc_to_est(utc_str):
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    est = dt.astimezone(ZoneInfo("America/New_York"))
    return est.strftime("%Y_%m_%d"), est.strftime("%H:%M:%S")

# -----------------------
# ODDS CONVERSION
# -----------------------
def decimal_to_american(decimal_odds):
    if decimal_odds is None:
        return None
    if decimal_odds >= 2:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))

# -----------------------
# PROCESS ONE FILE
# -----------------------
def _parse_game_markets(game, away_team, home_team):
    values = {
        "away_run_line": None, "home_run_line": None, "total": None,
        "away_rl_dec": None, "home_rl_dec": None,
        "over_dec": None, "under_dec": None,
        "away_ml_dec": None, "home_ml_dec": None,
    }
    markets = game["bookmakers"][0].get("markets", [])
    for market in markets:
        key = market["key"]
        outcomes = market.get("outcomes", [])
        if key == "h2h":
            for outcome in outcomes:
                if outcome["name"] == away_team:
                    values["away_ml_dec"] = outcome["price"]
                elif outcome["name"] == home_team:
                    values["home_ml_dec"] = outcome["price"]
        elif key == "spreads":
            for outcome in outcomes:
                if outcome["name"] == away_team:
                    values["away_run_line"] = outcome["point"]
                    values["away_rl_dec"] = outcome["price"]
                elif outcome["name"] == home_team:
                    values["home_run_line"] = outcome["point"]
                    values["home_rl_dec"] = outcome["price"]
        elif key == "totals":
            if outcomes:
                values["total"] = outcomes[0]["point"]
            for outcome in outcomes:
                if outcome["name"] == "Over":
                    values["over_dec"] = outcome["price"]
                elif outcome["name"] == "Under":
                    values["under_dec"] = outcome["price"]
    return values


def _parsed_odds_game_row(game):
    game_date, game_time = utc_to_est(game["commence_time"])
    away_team = game["away_team"]
    home_team = game["home_team"]
    values = _parse_game_markets(game, away_team, home_team)
    return game_date, [
        game.get("id"), "baseball", "mlb", game_date, game_time,
        home_team, away_team,
        values["away_run_line"], values["home_run_line"], values["total"],
        decimal_to_american(values["away_rl_dec"]),
        decimal_to_american(values["home_rl_dec"]),
        decimal_to_american(values["over_dec"]),
        decimal_to_american(values["under_dec"]),
        decimal_to_american(values["away_ml_dec"]),
        decimal_to_american(values["home_ml_dec"]),
        values["away_rl_dec"], values["home_rl_dec"],
        values["over_dec"], values["under_dec"],
        values["away_ml_dec"], values["home_ml_dec"],
    ]


def _write_grouped_odds_rows(grouped_rows, files_written):
    header = [
        "game_id","sport","league","game_date","game_time","home_team","away_team",
        "away_run_line","home_run_line","total",
        "away_dk_run_line_american","home_dk_run_line_american",
        "dk_total_over_american","dk_total_under_american",
        "away_dk_moneyline_american","home_dk_moneyline_american",
        "away_dk_run_line_decimal","home_dk_run_line_decimal",
        "dk_total_over_decimal","dk_total_under_decimal",
        "away_dk_moneyline_decimal","home_dk_moneyline_decimal",
    ]
    base_output_dir = Path("docs/win/baseball/mlb/00_intake/sportsbook")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    for game_date, rows in grouped_rows.items():
        output_path = base_output_dir / f"{game_date}_MLB.csv"
        with open(output_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        files_written.append((str(output_path), len(rows)))
        log(f"  WROTE {output_path} ({len(rows)} games)")


def process_file(file_path, files_written):
    allowed_root = Path(
        "docs/win/baseball/mlb/odds"
    ).resolve()

    safe_file_path = Path(
        file_path
    ).resolve()

    if not safe_file_path.is_relative_to(
        allowed_root
    ):
        raise ValueError(
            "Refusing odds input outside trusted "
            f"directory: {file_path}"
        )

    if safe_file_path.suffix.lower() != ".json":
        raise ValueError(
            f"Refusing non-JSON odds input: {file_path}"
        )

    file_path = safe_file_path

    log(f"Processing {file_path.name}")

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)
    grouped_rows = {}
    games_parsed = 0
    games_skipped = 0
    for game in data:
        if not game.get("bookmakers"):
            games_skipped += 1
            continue
        game_date, row = _parsed_odds_game_row(game)
        grouped_rows.setdefault(game_date, []).append(row)
        games_parsed += 1
    _write_grouped_odds_rows(grouped_rows, files_written)
    log(f"  games_parsed={games_parsed}, games_skipped={games_skipped}")

def main():
    files_written = []

    try:
        if INPUT_PATH.is_file():
            process_file(INPUT_PATH, files_written)
        elif INPUT_PATH.is_dir():
            files = list(INPUT_PATH.glob("*.json"))
            if not files:
                log(f"No JSON files found in {INPUT_PATH}")
                log("STATUS: SUCCESS (nothing to do)")
                return
            for file in sorted(files):
                process_file(file, files_written)
        else:
            raise ValueError(f"Invalid input path: {INPUT_PATH}")

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} games)")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

if __name__ == "__main__":
    main()
