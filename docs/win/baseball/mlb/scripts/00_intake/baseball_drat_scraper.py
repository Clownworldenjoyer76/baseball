#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/baseball_drat_scraper.py

import json
import os
import random
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import pytz
from playwright.sync_api import sync_playwright

URLS = {
    "mlb": "https://www.dratings.com/predictor/mlb-baseball-predictions/",
}

UTC = pytz.utc
ET = pytz.timezone("America/New_York")
ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "baseball_drat_scraper.txt"

with LOG_FILE.open("w", encoding="utf-8") as startup_log:
    startup_log.write(f"=== baseball_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        return UTC.localize(dt).astimezone(ET).strftime("%m/%d/%Y %I:%M %p")
    except (TypeError, ValueError, OverflowError):
        return date_time_str


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")
    all_rows = []
    for table in page.query_selector_all("table"):
        for row in table.query_selector_all("tbody tr"):
            cells = [cell.inner_text().strip() for cell in row.query_selector_all("td")]
            if cells:
                try:
                    cells[0] = convert_utc_to_et(cells[0].replace("\n", " "))
                except (AttributeError, TypeError, IndexError):
                    pass
                all_rows.append(cells)
    return all_rows


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
                        raw = scrape_page(page, url)
                        path = raw_dir / f"{date}_{sport}_raw.json"
                        count, written = save_merged_rows(path, raw)
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
