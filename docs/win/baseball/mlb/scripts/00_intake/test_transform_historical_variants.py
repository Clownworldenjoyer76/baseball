"""Regression checks for pregame DRatings row shapes and historical deduplication."""

import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("transform_baseball.py")
spec = importlib.util.spec_from_file_location("transform_baseball", SCRIPT)
transform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform)


def test_ten_field_pregame_row_is_accepted():
    row = [
        "04/11/2026 02:20 PM", "Pittsburgh Pirates\nChicago Cubs",
        "", "43.7%\n56.3%", "", "", "3.09\n3.74", "6.83", "", "",
    ]
    status, date, prediction = transform._transform_raw_row(row)
    assert status == "future"
    assert date == "2026_04_11"
    assert prediction["home_prob"] == "0.563"
    assert prediction["away_prob"] == "0.437"
    assert prediction["home_projected_runs"] == "3.74"


def test_same_game_time_variants_keep_official_time(tmp_path, monkeypatch):
    games = tmp_path / "games"
    games.mkdir()
    with (games / "2026_10_26_games.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "game_id", "game_date", "game_time", "home_team", "away_team"
        ])
        writer.writeheader()
        writer.writerow({
            "game_id": "401908044", "game_date": "2026-10-26",
            "game_time": "20:00:00", "home_team": "Tampa Bay Rays",
            "away_team": "Milwaukee Brewers",
        })
    monkeypatch.setattr(transform, "GAMES_DIR", games)
    transform._SCHEDULE_CACHE.clear()
    def row(time):
        return {
            "game_date": "2026_10_26", "game_time": time,
            "home_team": "Tampa Bay Rays",
            "away_team": "Milwaukee Brewers",
        }
    merged = {
        transform.prediction_key(row("04:00 PM"), {}): row("04:00 PM"),
        transform.prediction_key(row("08:00 PM"), {}): row("08:00 PM"),
    }
    transform.deduplicate_verified_schedule_variants(merged, {})
    assert len(merged) == 1
    assert next(iter(merged.values()))["game_time"] == "08:00 PM"


def test_distinct_doubleheader_game_ids_are_not_collapsed(tmp_path, monkeypatch):
    games = tmp_path / "games"
    games.mkdir()
    with (games / "2026_07_19_games.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "game_id", "game_date", "game_time", "home_team", "away_team"
        ])
        writer.writeheader()
        for gid, time in [("72971354", "12:35:00"), ("63303685", "19:20:00")]:
            writer.writerow({
                "game_id": gid, "game_date": "2026-07-19",
                "game_time": time, "home_team": "New York Yankees",
                "away_team": "Los Angeles Dodgers",
            })
    monkeypatch.setattr(transform, "GAMES_DIR", games)
    transform._SCHEDULE_CACHE.clear()
    def row(time):
        return {
            "game_date": "2026_07_19", "game_time": time,
            "home_team": "New York Yankees",
            "away_team": "Los Angeles Dodgers",
        }
    merged = {
        transform.prediction_key(row("12:35 PM"), {}): row("12:35 PM"),
        transform.prediction_key(row("07:20 PM"), {}): row("07:20 PM"),
    }
    transform.deduplicate_verified_schedule_variants(merged, {})
    assert len(merged) == 2
