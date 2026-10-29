#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/baseball_drat_scraper.py

import csv
import json
import os
import random
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pytz
from playwright.sync_api import sync_playwright

URLS = {
    "mlb": "https://www.dratings.com/predictor/mlb-baseball-predictions/",
}

UTC = pytz.utc
ET = pytz.timezone("America/New_York")
GAMES_DIR = Path("docs/win/baseball/mlb/00_intake/games")
BOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")
TEAM_MAP_FILE = Path("docs/win/baseball/mlb/maps/team_map_mlb.csv")
RAW_DIR = Path("docs/win/baseball/mlb/00_intake/drat_raw")
SOURCE_TIME_MODE = os.environ.get("DRATINGS_SOURCE_TIMEZONE", "auto").strip().lower()
if SOURCE_TIME_MODE not in {"auto", "utc", "et"}:
    raise ValueError("DRATINGS_SOURCE_TIMEZONE must be auto, utc or et")
TIME_TOLERANCE_MINUTES = 90
_SCHEDULE_CACHE = {}
_ALIAS_CACHE = None

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "baseball_drat_scraper.txt"

with LOG_FILE.open("w", encoding="utf-8") as startup_log:
    startup_log.write(f"=== baseball_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def _team_aliases():
    """Map DRatings and normalized team spellings to one identity."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    aliases = {}
    if TEAM_MAP_FILE.exists():
        with TEAM_MAP_FILE.open(newline="", encoding="utf-8-sig") as handle:
            for entry in csv.DictReader(handle):
                if entry.get("league", "").lower() != "mlb":
                    continue
                a = _simple_team(entry.get("alias", ""))
                c = _simple_team(entry.get("canonical_team", ""))
                if a and c:
                    aliases[a] = c
    _ALIAS_CACHE = aliases
    return aliases


def _simple_team(team):
    import re
    return re.sub(r"[^a-z0-9]", "", str(team).split("(")[0].lower())


def _team_key(team):
    basic = _simple_team(team)
    return _team_aliases().get(basic, basic)


def _schedule_for(date_value):
    """Read only pre-existing authoritative games (or sportsbook fallback)."""
    stamp = date_value.strftime("%Y_%m_%d")
    if stamp in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[stamp]
    games = GAMES_DIR / f"{stamp}_games.csv"
    book = BOOK_DIR / f"{stamp}_MLB.csv"
    selected = games if games.exists() else book
    result = []
    if selected.exists():
        with selected.open(newline="", encoding="utf-8-sig") as handle:
            for entry in csv.DictReader(handle):
                try:
                    d = datetime.strptime(entry["game_date"].replace("_", "-"), "%Y-%m-%d")
                    clock = entry["game_time"]
                    parsed = None
                    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
                        try:
                            parsed = datetime.strptime(clock.strip(), fmt)
                            break
                        except ValueError:
                            pass
                    if parsed is None:
                        continue
                    when = d.replace(hour=parsed.hour, minute=parsed.minute, second=parsed.second)
                    result.append({
                        "time_et": when, "home": _team_key(entry["home_team"]),
                        "away": _team_key(entry["away_team"]),
                        "game_id": entry.get("game_id", ""),
                        "source": "games" if selected == games else "sportsbook",
                    })
                except (KeyError, ValueError):
                    continue
    _SCHEDULE_CACHE[stamp] = result
    return result


def normalize_source_time(source_text, away_team, home_team, *, scraped_at_utc=None):
    """Choose UTC vs Eastern only when independent game schedule identifies it.

    The DRatings webpage provides a timezone-naive clock. Never guess a four-hour
    shift purely from its displayed hour. Without schedule evidence, keep the
    legacy UTC interpretation and mark the row unverified in the audit.
    """
    original = datetime.strptime(source_text.strip(), "%m/%d/%Y %I:%M %p")
    utc_et = UTC.localize(original).astimezone(ET).replace(tzinfo=None)
    local_et = original
    interpretations = {"utc": utc_et, "et": local_et}
    candidates = []
    for mode, value in interpretations.items():
        for delta_days in (-1, 0, 1):
            for entry in _schedule_for(value.date() + timedelta(days=delta_days)):
                if (entry["away"], entry["home"]) != (_team_key(away_team), _team_key(home_team)):
                    continue
                diff = abs((value - entry["time_et"]).total_seconds()) / 60
                if diff <= TIME_TOLERANCE_MINUTES:
                    candidates.append((diff, mode, entry))
    choice = None
    evidence = None
    if SOURCE_TIME_MODE in ("utc", "et"):
        choice = SOURCE_TIME_MODE
        evidence = f"explicit_{SOURCE_TIME_MODE}"
    elif candidates:
        matches = {(mode, e["game_id"], e["time_et"]) for _, mode, e in candidates}
        valid_modes = {mode for mode, _, _ in matches}
        if len(valid_modes) == 1 and len({(gid, dt) for _, gid, dt in matches}) == 1:
            choice = next(iter(valid_modes))
            evidence = "schedule_verified"
    if choice is None:
        choice = "utc"  # backward-compatible fallback, never called verified
        evidence = "unverified_legacy_utc_fallback"
    chosen = interpretations[choice]
    schedule_match = next((e for _, mode, e in candidates if mode == choice), None)
    return chosen.strftime("%m/%d/%Y %I:%M %p"), {
        "source_time": source_text.strip(), "normalized_time_et": chosen.strftime("%m/%d/%Y %I:%M %p"),
        "source_interpretation": choice, "evidence": evidence,
        "game_id": (schedule_match or {}).get("game_id", ""),
        "game_start_et": (
            schedule_match["time_et"].strftime("%Y-%m-%d %H:%M:%S")
            if schedule_match is not None else ""
        ),
        "away_team": away_team, "home_team": home_team,
        "scraped_at_utc": (scraped_at_utc or datetime.now(UTC)).isoformat(),
    }


def convert_utc_to_et(date_time_str: str) -> str:
    """Legacy helper for callers that explicitly know an input is UTC."""
    try:
        dt = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        return UTC.localize(dt).astimezone(ET).strftime("%m/%d/%Y %I:%M %p")
    except (TypeError, ValueError, OverflowError):
        return date_time_str


def _save_time_audit(date_value, records):
    if not records:
        return
    directory = RAW_DIR / "time_audit"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date_value}_mlb_time_audit.json"
    # Keep each scrape's evidence, even when the original prediction is retained.
    prior = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            prior = json.load(handle)
        if not isinstance(prior, list):
            raise ValueError(f"Invalid existing time audit: {path}")
    combined = prior + records
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(combined, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")
    all_rows = []
    audits = []
    captured = datetime.now(UTC)
    for table in page.query_selector_all("table"):
        for row in table.query_selector_all("tbody tr"):
            cells = [cell.inner_text().strip() for cell in row.query_selector_all("td")]
            if cells and len(cells) in (8, 11):
                try:
                    teams = cells[1].split("\n")
                    if len(teams) != 2:
                        all_rows.append(cells)
                        continue
                    away, home = (t.split("(")[0].strip() for t in teams)
                    original = cells[0].replace("\n", " ")
                    cells[0], audit = normalize_source_time(
                        original, away, home, scraped_at_utc=captured,
                    )
                    if audit["evidence"].startswith("unverified"):
                        log(f"UNVERIFIED SOURCE TIME: {away} @ {home} "
                            f"source={original} fallback_ET={cells[0]}")
                    elif audit["source_interpretation"] == "et":
                        log(f"SOURCE ALREADY ET: {away} @ {home} "
                            f"source={original} verified_ET={cells[0]}")
                    if len(cells) == 11 and audit["game_start_et"]:
                        game_start = ET.localize(
                            datetime.strptime(audit["game_start_et"], "%Y-%m-%d %H:%M:%S")
                        ).astimezone(UTC)
                        if captured >= game_start:
                            audit["decision"] = "skipped_after_first_pitch"
                            audits.append(audit)
                            log(f"SKIPPED AFTER FIRST PITCH: {away} @ {home} "
                                f"game_id={audit['game_id']} "
                                f"official_start_et={audit['game_start_et']}")
                            continue
                    audit["decision"] = "retained"
                    audits.append(audit)
                except (TypeError, ValueError, IndexError, AttributeError) as exc:
                    log(f"SOURCE TIME PARSE WARNING: {cells[:2]}: {exc}")
                all_rows.append(cells)
            elif cells:
                all_rows.append(cells)
    return all_rows, audits


def game_key(row):
    """Keep prediction and result rows separate; time distinguishes doubleheaders."""
    if not isinstance(row, list) or len(row) not in (8, 11):
        return None
    try:
        when = datetime.strptime(row[0].strip(), "%m/%d/%Y %I:%M %p")
        teams = row[1].split("\n")
        if len(teams) != 2:
            return None
        away, home = (part.split("(")[0].strip().casefold() for part in teams)
        if not away or not home:
            return None
        return len(row), when.strftime("%Y_%m_%d %H:%M"), away, home
    except (TypeError, AttributeError, ValueError, IndexError):
        return None


def scored_final(row):
    if not isinstance(row, list) or len(row) != 8:
        return False
    try:
        scores = row[5].split("\n")
        away, home = int(scores[0].strip()), int(scores[1].strip())
        return away >= 0 and home >= 0 and (away != 0 or home != 0)
    except (TypeError, ValueError, AttributeError):
        return False


def merge_rows(existing, incoming):
    """Preserve first pregame prediction; add completed rows independently."""
    merged = []
    positions = {}
    other_rows = set()

    for source, rows in (("existing", existing), ("new", incoming)):
        for row in rows:
            key = game_key(row)
            if key is None:
                signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
                if signature not in other_rows:
                    merged.append(row)
                    other_rows.add(signature)
                continue

            if key not in positions:
                positions[key] = len(merged)
                merged.append(row)
                continue

            previous = merged[positions[key]]
            if previous == row:
                continue
            if key[0] == 8 and not scored_final(previous) and scored_final(row):
                merged[positions[key]] = row
                log(f"REPLACED incomplete score placeholder: {key}")
            else:
                log(f"PRESERVED first {key[0]}-column row; differing {source} row: {key}")

    return merged


def save_merged_rows(path, scraped_rows):
    if not isinstance(scraped_rows, list):
        raise ValueError("Scraper did not return a list")

    valid_new = sum(game_key(row) is not None for row in scraped_rows)
    if not valid_new:
        log(f"NO VALID GAMES: left {path} unchanged")
        return 0, False

    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, list):
            raise ValueError(f"Refusing to replace invalid raw JSON list: {path}")

    merged = merge_rows(existing, scraped_rows)
    if merged == existing:
        log(f"UNCHANGED {path} ({len(merged)} rows)")
        return len(merged), False

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(merged, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    log(f"WROTE {path} ({len(merged)} rows; {valid_new} valid scraped games)")
    return len(merged), True


def main():
    files_written = []
    failures = 0
    try:
        date = datetime.now(ET).strftime("%Y_%m_%d")
        raw_dir = Path("docs/win/baseball/mlb/00_intake/drat_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    )
                })

                for sport, url in URLS.items():
                    log(f"Scraping {sport.upper()}")
                    try:
                        raw, time_audit = scrape_page(page, url)
                        path = raw_dir / f"{date}_{sport}_raw.json"
                        count, written = save_merged_rows(path, raw)
                        _save_time_audit(date, time_audit)
                        if written:
                            files_written.append((str(path), count))
                    except Exception as exc:
                        failures += 1
                        log(f"ERROR scraping {sport}: {exc}\n{traceback.format_exc()}")
                    time.sleep(random.uniform(2, 4))
            finally:
                browser.close()

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log(f"STATUS: {'FAILED' if failures else 'SUCCESS'}")
        if failures:
            raise RuntimeError(f"DRatings scrape failed for {failures} sport(s)")
    except Exception as exc:
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("Baseball drat scraper complete.")


if __name__ == "__main__":
    main()
