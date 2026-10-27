#!/usr/bin/env python3
"""Export MLB frontend JSON for every historical date with a games CSV.

Run:
    python docs/win/baseball/mlb/scripts/frontend.py

Writes frontend/data/mlb/YYYY_MM_DD_mlb_frontend.json for every date.
Each game has zero or more selected_bets. Pipeline inputs are unchanged.
"""

from __future__ import annotations

import csv
import json
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
    "symbol_code", "gust_mph", "chance_of_rain",
)


def clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if not value or value.lower() in {"nan", "none", "null"} else value


def read_csv(path: Path, *, required: bool = False) -> list[dict[str, str | None]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")
        return []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns or len(columns) != len(set(columns)) or any(not c for c in columns):
            raise ValueError(f"Invalid CSV headers: {path}")
        rows = []
        for number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Extra CSV fields at {path}:{number}")
            rows.append({k: clean(v) for k, v in row.items()})
        return rows


def required_key(row: dict, key: str, path: Path) -> str:
    value = row.get(key)
    if not value:
        raise ValueError(f"Missing {key} in {path}: {row}")
    return value


def index_unique(rows: list[dict], key: str, path: Path) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = required_key(row, key, path)
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {path}")
        result[value] = row
    return result


def equivalent(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
        return math.isfinite(x) and math.isfinite(y) and math.isclose(
            x, y, rel_tol=1e-10, abs_tol=1e-8
        )
    except ValueError:
        return False


def merge_fields(game: dict, row: dict, source: Path, *, strict: bool) -> None:
    for key, value in row.items():
        if key in {"game_id", "gamePk", "game_date"} or value is None:
            continue
        existing = game.get(key)
        if existing is None:
            game[key] = value
        elif strict and not equivalent(str(existing), str(value)):
            raise ValueError(
                f"Conflicting {key} for game_id={game['game_id']}: "
                f"{existing!r} vs {value!r} ({source})"
            )


def build(date: str) -> Path:
    schedule_path = MLB / f"00_intake/games/{date}_games.csv"
    schedule = read_csv(schedule_path, required=True)
    games: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    schedules: dict[str, dict] = {}

    for row in schedule:
        game_pk = required_key(row, "gamePk", schedule_path)
        schedule_id = required_key(row, "game_id", schedule_path)
        if game_pk in games or schedule_id in by_id:
            raise ValueError(f"Duplicate schedule gamePk/game_id: {game_pk}/{schedule_id}")
        game = {
            "gamePk": game_pk,
            "game_id": schedule_id,
            "schedule_game_id": schedule_id,
            "game_date": date,
            "schedule_game_time": row.get("game_time"),
            "selected_bets": [],
        }
        games[game_pk] = game
        schedules[game_pk] = row
        by_id[schedule_id] = game

    # A market file is required for a completed pregame pipeline run.
    for market in MARKETS:
        source = MLB / f"01_merge/{date}_mlb_{market}.csv"
        rows = read_csv(source, required=True)
        index_unique(rows, "game_id", source)
        for row in rows:
            market_id = required_key(row, "game_id", source)
            game_pk = row.get("gamePk")
            if game_pk:
                game = games.get(game_pk)
                if game is None:
                    raise ValueError(f"Unknown gamePk={game_pk} in {source}")
            else:
                game = by_id.get(market_id)
                if game is None:
                    raise ValueError(f"Cannot match game_id={market_id} in {source}")
            other = by_id.get(market_id)
            if other is not None and other is not game:
                raise ValueError(f"game_id={market_id} maps to different games in {source}")
            if game.get("market_game_id") not in (None, market_id):
                raise ValueError(f"Multiple market game_id values for gamePk={game['gamePk']}")
            game["market_game_id"] = market_id
            game["game_id"] = market_id
            by_id[market_id] = game
            merge_fields(game, row, source, strict=True)

    # Backfill schedule data without overwriting complete pipeline values.
    # The schedule and prediction feeds can use different time conventions.
    for game_pk, row in schedules.items():
        merge_fields(games[game_pk], row, schedule_path, strict=False)

    # Supplementary inputs only fill absent fields: completed market outputs
    # take precedence over earlier raw predictions, context and sportsbook.
    context_path = MLB / f"00_intake/mlb_raw/{date}_game_context.csv"
    context = read_csv(context_path)
    for game_pk, row in index_unique(context, "gamePk", context_path).items():
        if game_pk not in games:
            raise ValueError(f"Unknown context gamePk={game_pk} in {context_path}")
        merge_fields(games[game_pk], row, context_path, strict=False)

    pred_path = MLB / f"00_intake/predictions/pred_with_game_id/{date}_MLB.csv"
    book_path = MLB / f"00_intake/sportsbook/{date}_MLB.csv"
    for source in (pred_path, book_path):
        for game_id, row in index_unique(read_csv(source), "game_id", source).items():
            game = by_id.get(game_id)
            if game is None:
                print(
                    f"WARNING: Skipping unmatched supplemental game_id={game_id} "
                    f"in {source.relative_to(ROOT)}"
                )
                continue
            merge_fields(game, row, source, strict=False)

    # Preserve calculated moneyline model probabilities and EV/Kelly fields.
    ev_path = MLB / f"03_edges/ev_kelly/{date}_mlb_moneyline.csv"
    # Older dates predate the moneyline EV export. Include all dates anyway.
    ev_rows = read_csv(ev_path)
    for game_id, row in index_unique(ev_rows, "game_id", ev_path).items():
        game = by_id.get(game_id)
        if game is None:
            raise ValueError(f"Unknown moneyline EV game_id={game_id} in {ev_path}")
        merge_fields(game, row, ev_path, strict=False)

    # The standalone weather source is authoritative for weather fields.
    weather_path = MLB / f"data/weather/{date}_weather.csv"
    weather = read_csv(weather_path)
    for game_pk, row in index_unique(weather, "gamePk", weather_path).items():
        if game_pk not in games:
            raise ValueError(f"Unknown weather gamePk={game_pk} in {weather_path}")
        game = games[game_pk]
        for key in WEATHER_FIELDS:
            if row.get(key) is not None:
                game[key] = row[key]

    # Use the repository maps to provide display names where absent.
    pitcher_path = MLB / "maps/mlb_pitcher_ids.csv"
    team_path = MLB / "maps/mlb_team_ids.csv"
    venue_path = MLB / "maps/mlb_venue_ids.csv"
    pitchers = index_unique(read_csv(pitcher_path, required=True), "pitcher_id", pitcher_path)
    teams = index_unique(read_csv(team_path, required=True), "team_id", team_path)
    venues = index_unique(read_csv(venue_path, required=True), "venue_id", venue_path)
    for game in games.values():
        venue = venues.get(game.get("venue_id"), {})
        if not game.get("venue_name"):
            game["venue_name"] = venue.get("venue_name")
        for side in ("home", "away"):
            if not game.get(f"{side}_team"):
                game[f"{side}_team"] = teams.get(game.get(f"{side}_team_id"), {}).get("name")
            if not game.get(f"{side}_pitcher"):
                game[f"{side}_pitcher"] = pitchers.get(game.get(f"{side}_pitcher_id"), {}).get("full_name")
            if (game.get(f"{side}_pitcher") or "").upper() == "UNDECIDED UNDECIDED":
                game[f"{side}_pitcher"] = "Undecided"

        # Never present DRatings as the model's calibrated probabilities.
        game["home_prob"] = game.get("home_model_prob_moneyline")
        game["away_prob"] = game.get("away_model_prob_moneyline")
        game["home_projected_runs"] = game.get("model_home_runs")
        game["away_projected_runs"] = game.get("model_away_runs")
        game["total_projected_runs"] = game.get("model_total_runs")

    # The selected CSV is intentionally absent on dates with no picks.
    picks_path = MLB / f"04_select/{date}_MLB.csv"
    seen_picks = set()
    pick_count = 0
    for pick in read_csv(picks_path):
        game_id = required_key(pick, "game_id", picks_path)
        game = by_id.get(game_id)
        if game is None:
            raise ValueError(f"Unknown selected game_id={game_id} in {picks_path}")
        identity = (
            game["gamePk"], pick.get("market_type"),
            pick.get("bet_side"), pick.get("line"),
        )
        if identity in seen_picks:
            raise ValueError(f"Duplicate selected bet {identity} in {picks_path}")
        seen_picks.add(identity)
        game["selected_bets"].append(pick)
        pick_count += 1

    results = sorted(
        games.values(),
        key=lambda g: (g.get("game_time") or "", g["gamePk"]),
    )
    payload = {
        "schema_version": 1,
        "game_date": date,
        "game_count": len(results),
        "selected_bet_count": pick_count,
        "games": results,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{date}_mlb_frontend.json"
    temporary = destination.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"WROTE {destination.relative_to(ROOT)}: {len(results)} games, {pick_count} selected bets")
    return destination


def main() -> None:
    games_dir = MLB / "00_intake/games"
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Missing games directory: {games_dir}")

    dates = sorted({
        match.group(1)
        for path in games_dir.glob("*_games.csv")
        if (match := re.fullmatch(r"(\d{4}_\d{2}_\d{2})_games\.csv", path.name))
    })
    if not dates:
        raise FileNotFoundError(f"No dated games CSVs found in {games_dir}")

    for date in dates:
        build(date)
    print(f"Export complete: {len(dates)} dated MLB frontend JSON files")


if __name__ == "__main__":
    main()
