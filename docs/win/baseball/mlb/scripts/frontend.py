#!/usr/bin/env python3
"""Export dated MLB frontend CSV files for every available games date.

Run from any working directory:
    python docs/win/baseball/mlb/scripts/frontend.py

Per date, writes:
    frontend/data/mlb/YYYY_MM_DD_mlb_games.csv  (one row per scheduled game)
    frontend/data/mlb/YYYY_MM_DD_mlb_picks.csv  (unchanged selection rows)

Only docs/win/baseball/mlb/04_select/YYYY_MM_DD_MLB.csv supplies picks.
No JSON is produced, and source files are never modified.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
MLB = ROOT / "docs/win/baseball/mlb"
OUT = ROOT / "frontend/data/mlb"
MARKETS = ("moneyline", "run_line", "total")

WEATHER_FIELDS = (
    "weather_applicable", "weather_time", "temp_f", "wind_mph",
    "wind_dir", "precip_in", "humidity", "will_it_rain",
    "wind_blowing_out", "air_pressure_at_sea_level", "dew_point_f",
    "symbol_code",
)

# Only used when a particular date has no 04_select/YYYY_MM_DD_MLB.csv.
PICK_HEADERS = (
    "game_id", "sport", "game_date", "game_time", "league", "away_team",
    "home_team", "home_batters_found", "away_batters_found", "home_sp_found",
    "away_sp_found", "market_type", "bet_side", "market", "side", "line",
    "take_bet", "dk_odds_american", "dk_odds_decimal", "model_prob",
    "prob_used_for_selection", "prob_for_ev", "prob_for_kelly",
    "ev_probability_source", "kelly_probability_source", "ev", "kelly",
    "selection_reason", "low_confidence",
)

GAME_FIRST_COLUMNS = (
    "gamePk", "game_id", "schedule_game_id", "market_game_id",
    "game_date", "game_time", "schedule_game_time", "sport", "league",
    "home_team", "away_team", "home_pitcher", "away_pitcher",
)


def clean(value: str | None) -> str:
    if value is None:
        return ""
    result = value.strip()
    return "" if result.lower() in {"nan", "none", "null"} else result


def read_csv(path: Path, *, required: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")
        return [], []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers or any(not h for h in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid or duplicate CSV headers in {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({k: clean(v) for k, v in row.items()})
    return headers, rows


def unique_index(rows: list[dict[str, str]], key: str, source: Path) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            raise ValueError(f"Blank {key} in {source}")
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {source}")
        result[value] = row
    return result


def equivalent(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        a_number, b_number = float(a), float(b)
        return (
            math.isfinite(a_number)
            and math.isfinite(b_number)
            and math.isclose(a_number, b_number, rel_tol=1e-10, abs_tol=1e-8)
        )
    except ValueError:
        return False


def add_fields(target: dict[str, str], source_row: dict[str, str], *,
               source: Path, strict: bool = False) -> None:
    for field, value in source_row.items():
        if field in {"gamePk", "game_id", "game_date"} or not value:
            continue
        existing = target.get(field, "")
        if not existing:
            target[field] = value
        elif strict and not equivalent(existing, value):
            raise ValueError(
                f"Conflicting {field} for gamePk={target['gamePk']}: "
                f"{existing!r} / {value!r} in {source}"
            )


def warn_unmatched(key: str, value: str, source: Path) -> None:
    print(f"WARNING: Skipping unmatched {key}={value} in {source.relative_to(ROOT)}")


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build(date: str, *, pitchers: dict, teams: dict, venues: dict) -> None:
    schedule_path = MLB / f"00_intake/games/{date}_games.csv"
    schedule_headers, schedule_rows = read_csv(schedule_path, required=True)
    column_order = list(GAME_FIRST_COLUMNS)

    def include_columns(fields: list[str]) -> None:
        for field in fields:
            if field not in column_order:
                column_order.append(field)

    include_columns(schedule_headers)
    schedule_by_pk = unique_index(schedule_rows, "gamePk", schedule_path)
    if not schedule_rows:
        raise ValueError(f"Empty schedule in {schedule_path}")

    games_by_pk: dict[str, dict[str, str]] = {}
    games_by_id: dict[str, dict[str, str]] = {}

    for game_pk, schedule in schedule_by_pk.items():
        schedule_id = schedule.get("game_id", "")
        if not schedule_id or schedule_id in games_by_id:
            raise ValueError(f"Missing or duplicate schedule game_id in {schedule_path}: {schedule_id!r}")
        game = {
            "gamePk": game_pk,
            "game_id": schedule_id,
            "schedule_game_id": schedule_id,
            "market_game_id": "",
            "game_date": date,
            "schedule_game_time": schedule.get("game_time", ""),
        }
        games_by_pk[game_pk] = game
        games_by_id[schedule_id] = game

    # Each market provides distinct odds; shared fields must agree.
    for market in MARKETS:
        source = MLB / f"01_merge/{date}_mlb_{market}.csv"
        fields, rows = read_csv(source, required=True)
        include_columns(fields)
        for market_id, row in unique_index(rows, "game_id", source).items():
            game_pk = row.get("gamePk", "")
            game = games_by_pk.get(game_pk) if game_pk else games_by_id.get(market_id)
            if game is None:
                warn_unmatched("gamePk" if game_pk else "game_id", game_pk or market_id, source)
                continue
            previous_match = games_by_id.get(market_id)
            if previous_match is not None and previous_match is not game:
                raise ValueError(f"Market game_id={market_id} maps to two games in {source}")
            if game["market_game_id"] and game["market_game_id"] != market_id:
                raise ValueError(f"Conflicting market game_id for gamePk={game['gamePk']} in {source}")
            game["market_game_id"] = market_id
            game["game_id"] = market_id
            games_by_id[market_id] = game
            add_fields(game, row, source=source, strict=True)

    # The schedule provides any fields absent from the merged market records.
    # Its original game time is retained separately as schedule_game_time.
    for game_pk, schedule in schedule_by_pk.items():
        add_fields(games_by_pk[game_pk], schedule, source=schedule_path)

    # Supplemental data fills missing fields only; unmatched rows cannot
    # create fictitious scheduled games.
    context_source = MLB / f"00_intake/mlb_raw/{date}_game_context.csv"
    context_fields, context_rows = read_csv(context_source)
    include_columns(context_fields)
    for game_pk, row in unique_index(context_rows, "gamePk", context_source).items():
        game = games_by_pk.get(game_pk)
        if game is None:
            warn_unmatched("gamePk", game_pk, context_source)
            continue
        add_fields(game, row, source=context_source)

    for source in (
        MLB / f"00_intake/predictions/pred_with_game_id/{date}_MLB.csv",
        MLB / f"00_intake/sportsbook/{date}_MLB.csv",
    ):
        fields, rows = read_csv(source)
        include_columns(fields)
        for game_id, row in unique_index(rows, "game_id", source).items():
            game = games_by_id.get(game_id)
            if game is None:
                warn_unmatched("game_id", game_id, source)
                continue
            add_fields(game, row, source=source)

    ev_source = MLB / f"03_edges/ev_kelly/{date}_mlb_moneyline.csv"
    ev_fields, ev_rows = read_csv(ev_source)
    include_columns(ev_fields)
    for game_id, row in unique_index(ev_rows, "game_id", ev_source).items():
        game = games_by_id.get(game_id)
        if game is None:
            warn_unmatched("game_id", game_id, ev_source)
            continue
        add_fields(game, row, source=ev_source)

    weather_source = MLB / f"data/weather/{date}_weather.csv"
    weather_fields, weather_rows = read_csv(weather_source)
    include_columns(weather_fields)
    for game_pk, row in unique_index(weather_rows, "gamePk", weather_source).items():
        game = games_by_pk.get(game_pk)
        if game is None:
            warn_unmatched("gamePk", game_pk, weather_source)
            continue
        for field in WEATHER_FIELDS:
            if row.get(field, ""):
                game[field] = row[field]

    for game in games_by_pk.values():
        venue = venues.get(game.get("venue_id", ""), {})
        if not game.get("venue_name"):
            game["venue_name"] = venue.get("venue_name", "")
        for side in ("home", "away"):
            team = f"{side}_team"
            pitcher = f"{side}_pitcher"
            if not game.get(team):
                game[team] = teams.get(game.get(f"{side}_team_id", ""), {}).get("name", "")
            if not game.get(pitcher):
                game[pitcher] = pitchers.get(game.get(f"{side}_pitcher_id", ""), {}).get("full_name", "")
            if game.get(pitcher, "").upper() == "UNDECIDED UNDECIDED":
                game[pitcher] = "Undecided"

        # Aliases match the original exporter and never use DRatings
        # probabilities as a replacement for calibrated model probabilities.
        game["home_prob"] = game.get("home_model_prob_moneyline", "")
        game["away_prob"] = game.get("away_model_prob_moneyline", "")
        game["home_projected_runs"] = game.get("model_home_runs", "")
        game["away_projected_runs"] = game.get("model_away_runs", "")
        game["total_projected_runs"] = game.get("model_total_runs", "")

    games = sorted(
        games_by_pk.values(),
        key=lambda g: (g.get("game_time", ""), g["gamePk"]),
    )
    headers = column_order
    for game in games:
        for field in game:
            if field not in headers:
                headers.append(field)

    picks_source = MLB / f"04_select/{date}_MLB.csv"
    pick_headers, picks = read_csv(picks_source)
    if not pick_headers:
        pick_headers = list(PICK_HEADERS)
    elif "game_id" not in pick_headers:
        raise ValueError(f"Selected bets CSV missing game_id column: {picks_source}")
    # Preserve every source pick exactly; a missing file means header-only CSV.
    for pick in picks:
        if pick.get("game_id", "") not in games_by_id:
            print(
                f"WARNING: Selected bet references unmatched game_id="
                f"{pick.get('game_id', '')} in {picks_source.relative_to(ROOT)}"
            )

    game_dest = OUT / f"{date}_mlb_games.csv"
    picks_dest = OUT / f"{date}_mlb_picks.csv"
    write_csv(game_dest, headers, games)
    write_csv(picks_dest, pick_headers, picks)
    print(f"WROTE {game_dest.relative_to(ROOT)}: {len(games)} games")
    print(f"WROTE {picks_dest.relative_to(ROOT)}: {len(picks)} selected bets")


def main() -> None:
    games_dir = MLB / "00_intake/games"
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Missing schedule directory: {games_dir}")
    dates = sorted({
        match.group(1)
        for path in games_dir.glob("*_games.csv")
        if (match := re.fullmatch(r"(\d{4}_\d{2}_\d{2})_games\.csv", path.name))
    })
    if not dates:
        raise FileNotFoundError(f"No dated MLB games CSVs in {games_dir}")

    def load_map(filename: str, key: str) -> dict:
        path = MLB / "maps" / filename
        _, rows = read_csv(path, required=True)
        return unique_index(rows, key, path)

    pitchers = load_map("mlb_pitcher_ids.csv", "pitcher_id")
    teams = load_map("mlb_team_ids.csv", "team_id")
    venues = load_map("mlb_venue_ids.csv", "venue_id")
    for date in dates:
        build(date, pitchers=pitchers, teams=teams, venues=venues)
    print(f"Export complete: {len(dates)} dates, {len(dates) * 2} CSV files")


if __name__ == "__main__":
    main()
