#!/usr/bin/env python3
"""Build one complete MLB CSV per date for the homepage and Games Today.

Run from the repository root:
    python build_mlb_games_today_updated.py
    python build_mlb_games_today_updated.py --date 2026_04_17

Preserves all 01_merge fields. Adds model moneyline data, names, game metadata,
weather, and explicit frontend aliases. Never substitutes DRatings for model
win probabilities or projected runs.
"""

import argparse
import csv
import re
from pathlib import Path

INPUT_DIR = Path("docs/win/baseball/mlb/01_merge")
OUTPUT_DIR = Path("frontend/data/index_home/baseball")
MLB_ROOT = Path("docs/win/baseball/mlb")
EV_DIR = MLB_ROOT / "03_edges/ev_kelly"
GAMES_DIR = MLB_ROOT / "00_intake/games"
WEATHER_DIR = MLB_ROOT / "data/weather"
MAPS_DIR = MLB_ROOT / "maps"
MARKETS = ("total", "moneyline", "run_line")
FILENAME = re.compile(r"^(\d{4}_\d{2}_\d{2})_mlb_(total|moneyline|run_line)\.csv$")

MODEL_FIELDS = (
    "model_home_runs", "model_away_runs", "model_total_runs",
    "home_model_prob_moneyline", "away_model_prob_moneyline",
)
EXTRA_FIELDS = (
    "doubleheader", "gameNumber", "day_night", "venue_name",
    "gust_mph", "chance_of_rain",
    "home_prob", "away_prob", "home_projected_runs",
    "away_projected_runs", "total_projected_runs",
)
WEATHER_FIELDS = (
    "weather_applicable", "weather_time", "temp_f", "wind_mph",
    "wind_dir", "gust_mph", "precip_in", "humidity", "chance_of_rain",
    "will_it_rain", "wind_blowing_out",
)


def read_csv(path: Path, required=()):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)) or any(not x for x in fields):
            raise ValueError(f"Missing or duplicate headers: {path}")
        missing = sorted(set(required) - set(fields))
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
        rows = []
        for line_number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Extra CSV cells at {path}:{line_number}")
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return fields, rows


def indexed(rows, key, path):
    result = {}
    for n, row in enumerate(rows, 2):
        value = row.get(key, "")
        if not value:
            raise ValueError(f"Empty {key}: {path}:{n}")
        if value in result:
            raise ValueError(f"Duplicate {key}={value}: {path}:{n}")
        result[value] = row
    return result


def load_optional_index(path, key):
    if not path.is_file():
        print(f"WARNING: optional source missing: {path}")
        return {}
    _, rows = read_csv(path, (key,))
    return indexed(rows, key, path)


def add_field(columns, name):
    if name not in columns:
        columns.append(name)


def merge_value(out, field, incoming, path, game_id):
    existing = out.get(field, "")
    if existing and incoming and existing != incoming:
        raise ValueError(
            f"Conflicting values for game_id={game_id}, {field}: "
            f"{existing!r} versus {incoming!r} ({path})"
        )
    if incoming and not existing:
        out[field] = incoming


def normalized_pitcher(raw):
    name = " ".join(str(raw or "").strip().split())
    return "Undecided" if name.casefold() == "undecided undecided" else name


def optional_map(path, id_column):
    return load_optional_index(path, id_column)


def from_map(row, name_field, id_field, mapping, candidates):
    current = (row.get(name_field) or "").strip()
    if current:
        return current
    lookup = mapping.get(row.get(id_field, ""), {})
    return next((lookup[x] for x in candidates if lookup.get(x)), "")


def merge_date(date, pitcher_map, venue_map, team_map):
    columns = []
    merged = {}
    for market in MARKETS:
        path = INPUT_DIR / f"{date}_mlb_{market}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
        fields, rows = read_csv(path, ("game_id",))
        for field in fields:
            add_field(columns, field)
        for row in rows:
            game_id = row["game_id"]
            if not game_id:
                raise ValueError(f"Blank game_id: {path}")
            out = merged.setdefault(game_id, {})
            for field, value in row.items():
                merge_value(out, field, value, path, game_id)

    for field in (*MODEL_FIELDS, *EXTRA_FIELDS):
        add_field(columns, field)

    # Full moneyline EV CSV also includes model-calibrated win probabilities.
    ev_path = EV_DIR / f"{date}_mlb_moneyline.csv"
    ev_by_id = {}
    if ev_path.is_file():
        fields, rows = read_csv(ev_path, ("game_id", *MODEL_FIELDS))
        ev_by_id = indexed(rows, "game_id", ev_path)
        # Preserve EV-specific fields for use by future frontend pages.
        for field in fields:
            add_field(columns, field)
    else:
        print(f"WARNING: model probability source missing: {ev_path}")

    # Schedule game_id may differ from market game_id: use verified gamePk.
    schedule_path = GAMES_DIR / f"{date}_games.csv"
    schedule_by_pk = load_optional_index(schedule_path, "gamePk")
    weather_path = WEATHER_DIR / f"{date}_weather.csv"
    weather_by_pk = load_optional_index(weather_path, "gamePk")
    missing_probs = []
    missing_schedule = 0
    missing_weather = 0
    missing_venues = 0

    for game_id, out in merged.items():
        # The model runs in the EV source must agree with the merged market files.
        ev = ev_by_id.get(game_id, {})
        if ev:
            for field in MODEL_FIELDS[:3]:
                old, new = out.get(field, ""), ev.get(field, "")
                if old and new:
                    try:
                        matches = abs(float(old) - float(new)) < 1e-8
                    except ValueError:
                        matches = old == new
                    if not matches:
                        raise ValueError(f"Model run conflict: {date} {game_id} {field}: {old} vs {new}")
                if new:
                    out[field] = new
            for field, value in ev.items():
                # Keep 01_merge values for shared non-model fields.
                if value and (field not in out or not out[field]):
                    out[field] = value
        # Model probabilities are never taken from dratings_* or a market quote.
        out["home_prob"] = ev.get("home_model_prob_moneyline", "")
        out["away_prob"] = ev.get("away_model_prob_moneyline", "")
        if not out["home_prob"] or not out["away_prob"]:
            missing_probs.append(game_id)
        out["home_projected_runs"] = out.get("model_home_runs", "")
        out["away_projected_runs"] = out.get("model_away_runs", "")
        out["total_projected_runs"] = out.get("model_total_runs", "")

        game_pk = out.get("gamePk", "")
        schedule = schedule_by_pk.get(game_pk, {})
        if schedule:
            for field in ("doubleheader", "gameNumber", "day_night"):
                out[field] = schedule.get(field, "")
            for field in ("home_team", "away_team", "home_team_id", "away_team_id", "venue_id",
                          "home_pitcher_id", "away_pitcher_id"):
                if not out.get(field) and schedule.get(field):
                    out[field] = schedule[field]
        else:
            missing_schedule += 1

        # Prefer displayed full names to IDs, using maps only when necessary.
        out["home_team"] = from_map(out, "home_team", "home_team_id", team_map, ("name", "team_name"))
        out["away_team"] = from_map(out, "away_team", "away_team_id", team_map, ("name", "team_name"))
        for side in ("home", "away"):
            field = f"{side}_pitcher"
            out[field] = normalized_pitcher(
                from_map(out, field, f"{side}_pitcher_id", pitcher_map, ("full_name",))
            )
        out["venue_name"] = from_map(out, "venue_name", "venue_id", venue_map, ("venue_name",))
        if not out["venue_name"]:
            missing_venues += 1

        weather = weather_by_pk.get(game_pk, {})
        if weather:
            for field in WEATHER_FIELDS:
                if weather.get(field, "") != "":
                    out[field] = weather[field]
        else:
            missing_weather += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / f"{date}_mlb_games_today.csv"
    # Atomic replace: a validation/write failure will not leave a partial CSV.
    temp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(merged.values())
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    print(f"WROTE {dest}: {len(merged)} games, {len(columns)} columns")
    if missing_probs:
        print(f"WARNING: {len(missing_probs)} game(s) have blank model win probabilities (no DRatings fallback)")
    if missing_schedule:
        print(f"WARNING: {missing_schedule} game(s) have no matching schedule gamePk")
    if missing_weather:
        print(f"WARNING: {missing_weather} game(s) have no matching weather gamePk")
    if missing_venues:
        print(f"WARNING: {missing_venues} game(s) have no venue name match")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Build only this YYYY_MM_DD date")
    args = parser.parse_args()
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory missing: {INPUT_DIR}")
    if args.date and not re.fullmatch(r"\d{4}_\d{2}_\d{2}", args.date):
        parser.error("--date must be YYYY_MM_DD")

    pitcher_map = optional_map(MAPS_DIR / "mlb_pitcher_ids.csv", "pitcher_id")
    venue_map = optional_map(MAPS_DIR / "mlb_venue_ids.csv", "venue_id")
    team_map = optional_map(MAPS_DIR / "mlb_team_ids.csv", "team_id")
    dates = [args.date] if args.date else sorted({
        m.group(1) for path in INPUT_DIR.iterdir()
        if path.is_file() and (m := FILENAME.fullmatch(path.name))
    })
    if not dates:
        raise FileNotFoundError(f"No dated MLB input CSVs in {INPUT_DIR}")
    for date in dates:
        merge_date(date, pitcher_map, venue_map, team_map)


if __name__ == "__main__":
    main()
