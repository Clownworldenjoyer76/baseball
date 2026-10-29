#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/game_id_pred.py
#
# Injects game_id into prediction files using:
#   docs/win/baseball/mlb/00_intake/games/{date}_games.csv
#
# Input:
#   docs/win/baseball/mlb/00_intake/predictions/{date}_MLB.csv
#   docs/win/baseball/mlb/00_intake/games/{date}_games.csv
#   docs/win/baseball/mlb/00_intake/sportsbook/{date}_MLB.csv
#
# Output:
#   docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/{date}_MLB.csv
#   docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/baseline/{date}_MLB.csv
#
# Rejections:
#   docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/rejections/{date}_unmatched_predictions.csv
#
# Matching rules:
#   1. A prediction row must have a matching sportsbook row to be eligible for game_id assignment.
#   2. Prediction rows without a matching sportsbook row are non-fatal rejections.
#   3. Non-fatal rejected rows are written to rejection CSV and omitted from pred_with_game_id output.
#   4. Eligible prediction rows are matched to games rows by same home/away teams.
#   5. Never reuse a games row / game_id.
#   6. If same matchup has exactly one eligible prediction row and exactly one games row:
#      match them even if source times differ, and log the time difference.
#   7. If same matchup has multiple eligible prediction rows and the same number of games rows:
#      pair by chronological order. This handles doubleheaders.
#   8. Otherwise, match to the closest unused games row time within threshold.
#
# Step 2 hardening:
#   - Never write blank game_id to pred_with_game_id output.
#   - Non-fatal reject prediction rows when the matching sportsbook row is missing.
#   - Non-fatal reject prediction rows when the games file is missing/empty for current or future dates.
#   - Hard-fail if a past-date games file is missing/empty.
#   - Hard-fail if an eligible prediction row cannot be assigned a game_id for a past date with games data.
#   - Hard-fail if duplicate game_id appears in output.
#   - Hard-fail if matched eligible output row count differs from eligible prediction row count.
#   - Write rejected prediction rows to rejection CSV.
#   - Print rejected prediction rows to stdout so GitHub Actions logs show the reason.
#   - Print fatal exception/traceback directly to stdout so failed GitHub runs show the actual error.

import csv
from collections import Counter
import math
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Never
from zoneinfo import ZoneInfo


PRED_DIR = Path("docs/win/baseball/mlb/00_intake/predictions")
GAMES_DIR = Path("docs/win/baseball/mlb/00_intake/games")
BOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")

OUT_DIR = Path("docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id")
REJECTION_DIR = OUT_DIR / "rejections"
BASELINE_DIR = OUT_DIR / "baseline"
TIME_AUDIT_DIR = OUT_DIR / "time_audit"
ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REJECTION_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
TIME_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "game_id_pred.txt"

MAX_TIME_DIFF_MINUTES = 90
TZ_OFFSET_TOLERANCE_MINUTES = 15  # allow schedule revisions, never blanket-shift
TIME_AUDIT_HEADER = [
    "game_date", "game_id", "source_csv_row", "home_team", "away_team",
    "source_game_time", "verified_game_time", "diff_minutes",
    "time_resolution", "evidence",
]

NY = ZoneInfo("America/New_York")

OUTPUT_HEADER = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
]

REJECTION_HEADER = OUTPUT_HEADER + [
    "reject_reason",
    "reject_action",
    "fatal",
    "sportsbook_present",
    "sportsbook_match_detail",
    "candidate_games",
    "source_csv_row",
]

REQUIRED_PRED_COLS = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
]

REQUIRED_GAMES_COLS = [
    "gamePk",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "gameNumber",
]

REQUIRED_BOOK_COLS = [
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]


# ─────────────────────────────────────────────
# LOGGING / FAILURE
# ─────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def print_block(title: str, lines: list[str]):
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)
    for line in lines:
        print(line)
    print("=" * 80)
    print("")


def fail(msg: str) -> Never:
    log(f"FATAL VALIDATION ERROR: {msg}", "ERROR")
    print_block("FATAL VALIDATION ERROR IN game_id_pred.py", [msg])
    raise RuntimeError(msg)


# ─────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────

def duplicate_columns(header):
    counts = Counter(header)
    return list(
        dict.fromkeys(
            col
            for col in header
            if counts[col] > 1
        )
    )


def assert_no_duplicate_columns(header, label):
    dupes = duplicate_columns(header)

    if dupes:
        fail(f"{label} has duplicate columns: {dupes}")


def assert_required_columns(path: Path, header: list[str], required_cols: list[str], label: str):
    missing = [col for col in required_cols if col not in header]

    if missing:
        fail(f"{label} missing required columns in {path}: {missing}")


def load_csv(path: Path, required_cols=None, label=None, required_file=False) -> list[dict]:
    if not path.exists():
        msg = f"MISSING: {path}"

        if required_file:
            fail(msg)

        log(msg, "WARN")
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        assert_no_duplicate_columns(header, f"{label or path} input")

        if required_cols:
            assert_required_columns(path, header, required_cols, label or str(path))

        _qodana_return_value = list(reader)
    return _qodana_return_value


def write_csv(path: Path, header: list[str], rows: list[dict]):
    safe_path = Path(path).resolve()
    allowed_root = OUT_DIR.resolve()

    if not safe_path.is_relative_to(
        allowed_root
    ):
        raise ValueError(
            "Refusing prediction output outside "
            f"trusted directory: {path}"
        )

    if not (
        safe_path.name.endswith(".csv")
        or safe_path.name.endswith(".csv.tmp")
    ):
        raise ValueError(
            f"Refusing unexpected prediction output: {path}"
        )

    path = safe_path

    assert_no_duplicate_columns(
        header,
        f"{path} output",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=header,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    log(
        f"WROTE: {path} | rows={len(rows)}"
    )


def write_output_csv(date_str: str, header: list[str], rows: list[dict], summary: dict) -> Path:
    out_path = OUT_DIR / f"{date_str}_MLB.csv"
    write_csv(out_path, header, rows)
    summary["files_written"] += 1
    return out_path


def clear_stale_rejection_file(rejection_path: Path) -> None:
    """Remove a prior rejection CSV after a clean successful run for that date."""
    if rejection_path.exists():
        rejection_path.unlink()
        log(f"REMOVED STALE REJECTION FILE: {rejection_path}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces for team name matching."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_int(value, default=0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def clean_date(value: str) -> str:
    return str(value or "").strip().replace("_", "-")


def parse_date_value(value: str):
    value = clean_date(value)

    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def current_cutoff_date():
    env_date = os.environ.get("DATE", "")
    parsed_env_date = parse_date_value(env_date)

    if parsed_env_date is not None:
        return parsed_env_date

    return datetime.now(NY).date()


def is_current_or_future_date(date_str: str) -> bool:
    parsed_date = parse_date_value(date_str)

    if parsed_date is None:
        fail(f"Could not parse date for current/future validation: {date_str}")

    return parsed_date >= current_cutoff_date()


def validate_row_date(date_str: str, row: dict, label: str, csv_row: int | None = None) -> None:
    expected_date = parse_date_value(date_str)
    actual_raw = row.get("game_date", "")
    actual_date = parse_date_value(actual_raw)
    row_text = "" if csv_row is None else f" csv_row={csv_row}"

    if expected_date is None:
        fail(f"Could not parse filename date for {label}: {date_str}")

    if actual_date is None:
        fail(
            f"{date_str} | {label} has invalid game_date={actual_raw!r}"
            f"{row_text}"
        )

    if actual_date != expected_date:
        fail(
            f"{date_str} | {label} game_date mismatch{row_text}: "
            f"row_game_date={actual_raw!r} expected_date={expected_date.isoformat()}"
        )


def parse_prediction_datetime(date_str: str, time_str: str):
    date_clean = clean_date(date_str)
    time_clean = re.sub(r"\s+", " ", str(time_str or "").strip()).upper()

    if not date_clean or not time_clean:
        return None

    formats = [
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    value = f"{date_clean} {time_clean}"

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def parse_games_datetime(date_str: str, time_str: str):
    date_clean = clean_date(date_str)
    time_clean = str(time_str or "").strip()

    if not date_clean or not time_clean:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M:%S %p",
    ]

    value = f"{date_clean} {time_clean}"

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def parse_book_datetime(date_str: str, time_str: str):
    return parse_games_datetime(date_str, time_str)


def minutes_between(a: datetime | None, b: datetime | None) -> float | None:
    if a is None or b is None:
        return None

    return abs((a - b).total_seconds()) / 60.0


def dt_sort_value(dt):
    if dt is None:
        return math.inf

    return dt.timestamp()


def make_output_row(pred_entry: dict, game_id: str) -> dict:
    p = pred_entry["row"]

    return {
        "game_id": game_id,
        "sport": p.get("sport", ""),
        "league": p.get("league", ""),
        "game_date": p.get("game_date", ""),
        "game_time": (
            pred_entry.get("verified_game_time", p.get("game_time", ""))
            if game_id else p.get("game_time", "")
        ),
        "home_team": p.get("home_team", ""),
        "away_team": p.get("away_team", ""),
        "home_pitcher": p.get("home_pitcher", ""),
        "away_pitcher": p.get("away_pitcher", ""),
        "home_prob": p.get("home_prob", ""),
        "away_prob": p.get("away_prob", ""),
        "away_projected_runs": p.get("away_projected_runs", ""),
        "home_projected_runs": p.get("home_projected_runs", ""),
        "total_projected_runs": p.get("total_projected_runs", ""),
    }


def make_rejection_row(
    pred_entry: dict,
    reason: str,
    candidate_games: str = "",
    fatal: bool = True,
    sportsbook_present: bool = False,
    sportsbook_match_detail: str = "",
) -> dict:
    row = make_output_row(pred_entry, "")
    row["reject_reason"] = reason
    row["reject_action"] = "hard_failure" if fatal else "omit_and_continue"
    row["fatal"] = "1" if fatal else "0"
    row["sportsbook_present"] = "1" if sportsbook_present else "0"
    row["sportsbook_match_detail"] = sportsbook_match_detail
    row["candidate_games"] = candidate_games
    row["source_csv_row"] = str(pred_entry.get("csv_row", ""))
    return row


def matchup_label(row: dict) -> str:
    return f"{row.get('away_team', '')} @ {row.get('home_team', '')}"


def describe_game_entry(game_entry: dict) -> str:
    g = game_entry["row"]
    diff_fields = [
        f"game_id={g.get('game_id', '')}",
        f"gamePk={g.get('gamePk', '')}",
        f"time={g.get('game_time', '')}",
        f"gameNumber={g.get('gameNumber', '')}",
        f"away={g.get('away_team', '')}",
        f"home={g.get('home_team', '')}",
    ]
    return "|".join(diff_fields)


def describe_book_entry(book_entry: dict) -> str:
    b = book_entry["row"]
    diff_fields = [
        f"game_id={b.get('game_id', '')}",
        f"time={b.get('game_time', '')}",
        f"away={b.get('away_team', '')}",
        f"home={b.get('home_team', '')}",
    ]
    return "|".join(diff_fields)


def describe_candidates(scored: list[tuple[float | None, dict]]) -> str:
    parts = []

    for diff, game_entry in scored:
        diff_text = "NA" if diff is None else f"{diff:.1f}"
        parts.append(f"{describe_game_entry(game_entry)}|diff_minutes={diff_text}")

    return "; ".join(parts)


def describe_book_candidates(scored: list[tuple[float | None, dict]]) -> str:
    parts = []

    for diff, book_entry in scored:
        diff_text = "NA" if diff is None else f"{diff:.1f}"
        parts.append(f"{describe_book_entry(book_entry)}|diff_minutes={diff_text}")

    return "; ".join(parts)


def print_rejection_rows(date_str: str, rejection_path: Path, rejection_rows: list[dict]) -> None:
    print("")
    print("=" * 80)
    print(f"REJECTED PREDICTION ROWS | date={date_str}")
    print(f"rejection_csv={rejection_path.as_posix()}")
    print(f"count={len(rejection_rows)}")
    print("=" * 80)

    for idx, row in enumerate(rejection_rows, start=1):
        print(f"REJECTED #{idx}")
        print(f"  reject_reason={row.get('reject_reason', '')}")
        print(f"  reject_action={row.get('reject_action', '')}")
        print(f"  fatal={row.get('fatal', '')}")
        print(f"  sportsbook_present={row.get('sportsbook_present', '')}")
        print(f"  sportsbook_match_detail={row.get('sportsbook_match_detail', '')}")
        print(f"  source_csv_row={row.get('source_csv_row', '')}")
        print(f"  game_date={row.get('game_date', '')}")
        print(f"  game_time={row.get('game_time', '')}")
        print(f"  away_team={row.get('away_team', '')}")
        print(f"  home_team={row.get('home_team', '')}")
        print(f"  away_pitcher={row.get('away_pitcher', '')}")
        print(f"  home_pitcher={row.get('home_pitcher', '')}")
        print(f"  away_prob={row.get('away_prob', '')}")
        print(f"  home_prob={row.get('home_prob', '')}")
        print(f"  candidate_games={row.get('candidate_games', '')}")
        print("-" * 80)

    print("=" * 80)
    print("END REJECTED PREDICTION ROWS")
    print("=" * 80)
    print("")


# ─────────────────────────────────────────────
# GROUP BUILDERS
# ─────────────────────────────────────────────

def build_prediction_groups(pred_rows: list[dict], date_str: str) -> tuple[dict, list]:
    groups = {}
    key_order = []

    for idx, p in enumerate(pred_rows):
        csv_row = idx + 2
        validate_row_date(date_str, p, "prediction row", csv_row)
        key = (norm(p.get("home_team", "")), norm(p.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"prediction row has blank team at csv_row={csv_row}: "
                f"away={p.get('away_team', '')} home={p.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []
            key_order.append(key)

        groups[key].append({
            "row": p,
            "key": key,
            "index": idx,
            "csv_row": csv_row,
            "dt": parse_prediction_datetime(
                p.get("game_date", ""),
                p.get("game_time", ""),
            ),
        })

    return groups, key_order


def build_games_groups(games_rows: list[dict], date_str: str) -> dict:
    groups = {}
    seen_game_ids = {}
    seen_gamepks = {}

    for idx, g in enumerate(games_rows):
        csv_row = idx + 2

        validate_row_date(date_str, g, "games row", csv_row)
        game_id = (g.get("game_id") or "").strip()
        game_pk = (g.get("gamePk") or "").strip()

        if not game_id:
            fail(f"{date_str} | games row has blank game_id at csv_row={csv_row}: {g}")

        if not game_pk:
            fail(f"{date_str} | games row has blank gamePk at csv_row={csv_row}: {g}")

        if game_id in seen_game_ids:
            fail(
                f"{date_str} | duplicate games game_id={game_id} "
                f"first_csv_row={seen_game_ids[game_id]} second_csv_row={csv_row}"
            )

        if game_pk in seen_gamepks:
            fail(
                f"{date_str} | duplicate games gamePk={game_pk} "
                f"first_csv_row={seen_gamepks[game_pk]} second_csv_row={csv_row}"
            )

        seen_game_ids[game_id] = csv_row
        seen_gamepks[game_pk] = csv_row

        key = (norm(g.get("home_team", "")), norm(g.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"{date_str} | games row has blank team at csv_row={csv_row}: "
                f"away={g.get('away_team', '')} home={g.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []

        groups[key].append({
            "row": g,
            "key": key,
            "index": idx,
            "csv_row": csv_row,
            "dt": parse_games_datetime(
                g.get("game_date", ""),
                g.get("game_time", ""),
            ),
            "game_number": parse_int(g.get("gameNumber", "1"), 1),
            "used": False,
        })

    return groups


def build_book_groups(book_rows: list[dict], date_str: str) -> dict:
    groups = {}
    seen_book_game_ids = {}

    for idx, b in enumerate(book_rows):
        csv_row = idx + 2
        validate_row_date(date_str, b, "sportsbook row", csv_row)
        game_id = (b.get("game_id") or "").strip()

        if not game_id:
            fail(f"{date_str} | sportsbook row has blank game_id at csv_row={csv_row}: {b}")

        if game_id in seen_book_game_ids:
            fail(
                f"{date_str} | duplicate sportsbook game_id={game_id} "
                f"first_csv_row={seen_book_game_ids[game_id]} second_csv_row={csv_row}"
            )

        seen_book_game_ids[game_id] = csv_row

        key = (norm(b.get("home_team", "")), norm(b.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"{date_str} | sportsbook row has blank team at csv_row={csv_row}: "
                f"away={b.get('away_team', '')} home={b.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []

        groups[key].append({
            "row": b,
            "key": key,
            "index": idx,
            "csv_row": csv_row,
            "dt": parse_book_datetime(
                b.get("game_date", ""),
                b.get("game_time", ""),
            ),
            "used": False,
        })

    return groups


# ─────────────────────────────────────────────
# SPORTSBOOK PRESENCE
# ─────────────────────────────────────────────


def _explicit_book_id_matches(date_str, label, preds, books, presence):
    """Honor recovered game IDs before time-based matching; never guess another ID."""
    unmatched = []
    for pred in preds:
        supplied_id = (pred["row"].get("game_id") or "").strip()
        if not supplied_id:
            unmatched.append(pred)
            continue
        candidates = [
            book for book in books
            if book["row"].get("game_id", "").strip() == supplied_id
            and not book["used"]
        ]
        if len(candidates) == 1:
            _record_sportsbook_match(
                date_str, label, pred, candidates[0], presence, "explicit_game_id"
            )
        else:
            log(
                f"{date_str} | UNMATCHED EXPLICIT GAME ID: {label} "
                f"pred_game_id={supplied_id} csv_row={pred['csv_row']} "
                f"matching_unused_sportsbook_rows={len(candidates)}", "WARN",
            )
    return unmatched


def _time_audit_record(date_str, entry, official, diff, resolution, evidence):
    return {
        "game_date": date_str,
        "game_id": official.get("game_id", ""),
        "source_csv_row": entry["csv_row"],
        "home_team": entry["row"].get("home_team", ""),
        "away_team": entry["row"].get("away_team", ""),
        "source_game_time": entry["row"].get("game_time", ""),
        "verified_game_time": official.get("game_time", ""),
        "diff_minutes": round(diff, 1),
        "time_resolution": resolution,
        "evidence": evidence,
    }


def _verify_prediction_times(date_str, pred_groups, book_groups, games_rows):
    """Correct only independently-verified UTC/ET offsets in matching state.

    For a blank game ID, a singleton prediction + games + sportsbook with
    matching authoritative IDs is required. Explicit IDs can be verified
    across doubleheaders. Original prediction CSVs are never modified.
    """
    if not games_rows:
        return []
    games_groups = build_games_groups(games_rows, date_str)
    events = []
    date = parse_date_value(date_str)
    expected_offset = abs(
        int(datetime(date.year, date.month, date.day, 12,
                     tzinfo=NY).utcoffset().total_seconds() / 60)
    )
    for key, predictions in pred_groups.items():
        games = games_groups.get(key, [])
        books = book_groups.get(key, [])
        for pred in predictions:
            supplied_id = (pred["row"].get("game_id") or "").strip()
            evidence = ""
            game = None
            if supplied_id:
                verified_games = [
                    g for g in games if g["row"].get("game_id", "").strip() == supplied_id
                ]
                verified_books = [
                    b for b in books if b["row"].get("game_id", "").strip() == supplied_id
                ]
                if len(verified_games) == 1 and len(verified_books) == 1:
                    game = verified_games[0]
                    evidence = "explicit_game_id_verified_in_games_and_sportsbook"
            elif len(predictions) == len(games) == len(books) == 1:
                if games[0]["row"].get("game_id", "") == books[0]["row"].get("game_id", ""):
                    game = games[0]
                    evidence = "unique_matchup_verified_same_id_in_games_and_sportsbook"
            if game is None or pred.get("dt") is None or game.get("dt") is None:
                continue
            diff = minutes_between(pred["dt"], game["dt"])
            if diff is None or diff <= MAX_TIME_DIFF_MINUTES:
                continue
            official = game["row"]
            if abs(diff - expected_offset) <= TZ_OFFSET_TOLERANCE_MINUTES:
                # Store canonical schedule time *only* in enriched output.
                pred["verified_game_time"] = game["dt"].strftime("%I:%M %p")
                pred["dt"] = game["dt"]
                resolution = "verified_timezone_offset_corrected"
                log(
                    f"{date_str} | VERIFIED TIMEZONE CORRECTION: "
                    f"{matchup_label(pred['row'])} csv_row={pred['csv_row']} "
                    f"source={pred['row'].get('game_time', '')} "
                    f"official={official.get('game_time', '')} "
                    f"game_id={official.get('game_id', '')} diff_minutes={round(diff, 1)} "
                    f"evidence={evidence}", "WARN",
                )
            else:
                resolution = "unverified_discrepancy_no_correction"
                log(
                    f"{date_str} | UNVERIFIED TIME DISCREPANCY: "
                    f"{matchup_label(pred['row'])} csv_row={pred['csv_row']} "
                    f"source={pred['row'].get('game_time', '')} "
                    f"official={official.get('game_time', '')} "
                    f"game_id={official.get('game_id', '')} diff_minutes={round(diff, 1)} "
                    f"evidence={evidence}; manual review required", "WARN",
                )
            events.append(_time_audit_record(
                date_str, pred, official, diff, resolution, evidence,
            ))
    return events


def _write_time_audit(date_str, records):
    """Only write when the entire date has successfully completed."""
    path = TIME_AUDIT_DIR / f"{date_str}_time_audit.csv"
    if records:
        write_csv(path, TIME_AUDIT_HEADER, records)
    elif path.exists():
        # Prior reports are historical evidence: never delete them automatically.
        log(f"TIME AUDIT PRESERVED (no new discrepancies): {path}")


def _initialize_sportsbook_presence(presence, preds):
    for pred_entry in preds:
        presence[pred_entry["index"]] = {"present": False, "detail": ""}


def _record_sportsbook_match(
    date_str, label, pred_entry, book_entry, presence, match_kind,
):
    book_entry["used"] = True
    presence[pred_entry["index"]] = {
        "present": True,
        "detail": describe_book_entry(book_entry),
    }
    diff = minutes_between(
        pred_entry.get("dt"),
        book_entry.get("dt"),
    )
    diff_text = (
        ""
        if diff is None
        else f" diff_minutes={round(diff, 1)}"
    )
    log(
        f"{date_str} | SPORTSBOOK MATCH {match_kind}: {label} "
        f"pred_time={pred_entry['row'].get('game_time', '')} "
        f"book_time={book_entry['row'].get('game_time', '')}{diff_text}"
    )


def _sportsbook_one_to_one(date_str, label, preds, unused_books, presence):
    if len(preds) != 1 or len(unused_books) != 1:
        return False

    _record_sportsbook_match(
        date_str,
        label,
        preds[0],
        unused_books[0],
        presence,
        "one-to-one",
    )
    return True


def _sportsbook_order_match(date_str, label, preds, unused_books, presence):
    if not (1 < len(preds) == len(unused_books)):
        return False

    sorted_preds = sorted(
        preds, key=lambda item: (dt_sort_value(item.get("dt")), item["index"])
    )
    sorted_books = sorted(
        unused_books, key=lambda item: (dt_sort_value(item.get("dt")), item["index"])
    )
    log(
        f"{date_str} | SPORTSBOOK ORDER MATCH duplicate matchup: {label} "
        f"pred_count={len(sorted_preds)} book_count={len(sorted_books)}"
    )

    for pred_entry, book_entry in zip(sorted_preds, sorted_books):
        _record_sportsbook_match(
            date_str,
            label,
            pred_entry,
            book_entry,
            presence,
            "order",
        )

    return True


def _closest_unused_book(pred_entry, books):
    available = [book for book in books if not book["used"]]
    scored = [
        (minutes_between(pred_entry.get("dt"), book.get("dt")), book)
        for book in available
    ]
    valid = [item for item in scored if item[0] is not None]
    if not valid:
        return None, None, scored
    selected_diff, selected = min(valid, key=lambda item: item[0])
    if selected_diff > MAX_TIME_DIFF_MINUTES:
        return None, selected_diff, scored
    return selected, selected_diff, scored


def _sportsbook_closest_matches(date_str, label, preds, books, presence):
    sorted_preds = sorted(
        preds, key=lambda item: (dt_sort_value(item.get("dt")), item["index"])
    )
    for pred_entry in sorted_preds:
        selected, selected_diff, scored = _closest_unused_book(pred_entry, books)
        if selected is None:
            if not [book for book in books if not book["used"]]:
                log(
                    f"{date_str} | sportsbook no unused row for prediction: {label} "
                    f"pred_time={pred_entry['row'].get('game_time', '')}",
                    "WARN",
                )
            else:
                log(
                    f"{date_str} | sportsbook missing within time threshold: {label} "
                    f"pred_time={pred_entry['row'].get('game_time', '')} "
                    f"candidate_books={describe_book_candidates(scored)}",
                    "WARN",
                )
            continue
        selected["used"] = True
        presence[pred_entry["index"]] = {
            "present": True,
            "detail": describe_book_entry(selected),
        }
        diff_text = "" if selected_diff is None else f" diff_minutes={round(selected_diff, 1)}"
        log(
            f"{date_str} | SPORTSBOOK MATCH closest: {label} "
            f"pred_time={pred_entry['row'].get('game_time', '')} "
            f"book_time={selected['row'].get('game_time', '')}{diff_text}"
        )


def _match_sportsbook_group(date_str, preds, books, presence):
    label = matchup_label(preds[0]["row"])
    if not books:
        log(
            f"{date_str} | sportsbook missing for prediction matchup: "
            f"{label} pred_count={len(preds)}",
            "WARN",
        )
        return
    preds = _explicit_book_id_matches(date_str, label, preds, books, presence)
    if not preds:
        return
    unused_books = [book for book in books if not book["used"]]
    if _sportsbook_one_to_one(date_str, label, preds, unused_books, presence):
        return
    if _sportsbook_order_match(date_str, label, preds, unused_books, presence):
        return
    _sportsbook_closest_matches(date_str, label, preds, books, presence)














def build_sportsbook_presence(date_str: str, pred_groups: dict, pred_key_order: list, book_groups: dict) -> dict:
    presence = {}
    for key in pred_key_order:
        preds = pred_groups.get(key, [])
        if not preds:
            continue
        _initialize_sportsbook_presence(presence, preds)
        _match_sportsbook_group(date_str, preds, book_groups.get(key, []), presence)
    return presence

def validate_output_rows(date_str: str, eligible_count: int, output_rows: list[dict], allow_preserved: bool = False):
    if allow_preserved:
        if len(output_rows) < eligible_count:
            fail(
                f"{date_str} | eligible output row count below refreshed eligible count: "
                f"eligible_prediction_rows={eligible_count} output_rows={len(output_rows)}"
            )
    elif len(output_rows) != eligible_count:
        fail(
            f"{date_str} | eligible output row count mismatch: "
            f"eligible_prediction_rows={eligible_count} output_rows={len(output_rows)}"
        )

    seen_game_ids = {}

    for idx, row in enumerate(output_rows):
        csv_row = idx + 2
        validate_row_date(date_str, row, "output row", csv_row)
        game_id = (row.get("game_id") or "").strip()
        home_team = norm(row.get("home_team", ""))
        away_team = norm(row.get("away_team", ""))

        if not home_team or not away_team:
            fail(
                f"{date_str} | output row has blank team at csv_row={csv_row}: "
                f"away={row.get('away_team', '')} home={row.get('home_team', '')}"
            )

        if not game_id:
            fail(f"{date_str} | output row has blank game_id at csv_row={csv_row}: {row}")

        if game_id in seen_game_ids:
            fail(
                f"{date_str} | duplicate output game_id={game_id} "
                f"first_csv_row={seen_game_ids[game_id]} second_csv_row={csv_row}"
            )

        seen_game_ids[game_id] = csv_row


def load_same_day_baseline(date_str: str) -> list[dict]:
    """Load the durable cumulative same-day matched-prediction baseline."""
    baseline_path = BASELINE_DIR / f"{date_str}_MLB.csv"
    if baseline_path.exists():
        return load_csv(baseline_path, OUTPUT_HEADER, "same-day matched prediction baseline")

    # Migration fallback: seed once from the previously committed mutable output.
    # After the baseline exists, later refreshes never use the mutable output as persistence.
    existing_path = OUT_DIR / f"{date_str}_MLB.csv"
    if existing_path.exists():
        rows = load_csv(existing_path, OUTPUT_HEADER, "existing matched prediction seed")
        if rows:
            log(
                f"{date_str} | seeding durable same-day baseline from existing matched output: "
                f"rows={len(rows)}"
            )
            return rows

    return []


def validate_same_day_row_identity(
    date_str: str,
    row: dict,
    game_row: dict,
    row_label: str,
) -> None:
    row_home = norm(row.get("home_team", ""))
    row_away = norm(row.get("away_team", ""))
    games_home = norm(game_row.get("home_team", ""))
    games_away = norm(game_row.get("away_team", ""))

    if row_home != games_home or row_away != games_away:
        game_id = (row.get("game_id") or "").strip()
        fail(
            f"{date_str} | {row_label} identity mismatch "
            f"for game_id={game_id}: "
            f"{row_label}={row.get('away_team', '')} @ "
            f"{row.get('home_team', '')} "
            f"games={game_row.get('away_team', '')} @ "
            f"{game_row.get('home_team', '')}"
        )

    row_date = parse_date_value(row.get("game_date", ""))
    games_date = parse_date_value(game_row.get("game_date", ""))

    if (
        row_date is None
        or games_date is None
        or row_date != games_date
    ):
        game_id = (row.get("game_id") or "").strip()
        fail(
            f"{date_str} | {row_label} date mismatch "
            f"for game_id={game_id}: "
            f"{row_label}_game_date={row.get('game_date', '')} "
            f"games_game_date={game_row.get('game_date', '')}"
        )


def _same_day_games_by_id(date_str, games_rows):
    if not games_rows:
        fail(f"{date_str} | refusing to update same-day baseline from empty games rows")
    build_games_groups(games_rows, date_str)
    return {
        (row.get("game_id") or "").strip(): row
        for row in games_rows
        if (row.get("game_id") or "").strip()
    }


def _merge_preserved_baseline_rows(date_str, baseline_rows, games_by_id):
    merged = {}
    for row in baseline_rows:
        game_id = (row.get("game_id") or "").strip()
        if not game_id or game_id not in games_by_id:
            continue
        validate_same_day_row_identity(
            date_str, row, games_by_id[game_id], "baseline"
        )
        merged[game_id] = {col: row.get(col, "") for col in OUTPUT_HEADER}
    return merged


def _merge_refreshed_baseline_rows(date_str, refreshed_rows, games_by_id, merged):
    refreshed_ids = set()
    valid_game_ids = set(games_by_id)
    for row in refreshed_rows:
        game_id = (row.get("game_id") or "").strip()
        if not game_id:
            fail(
                f"{date_str} | refreshed row has blank game_id before baseline merge: {row}"
            )
        if game_id not in valid_game_ids:
            fail(
                f"{date_str} | refreshed game_id missing from authoritative games file "
                f"before baseline merge: game_id={game_id}"
            )
        validate_same_day_row_identity(
            date_str, row, games_by_id[game_id], "refreshed"
        )
        refreshed_ids.add(game_id)
        merged[game_id] = {col: row.get(col, "") for col in OUTPUT_HEADER}
    if len(refreshed_ids) != len(refreshed_rows):
        fail(
            f"{date_str} | refreshed rows collapsed during baseline merge: "
            f"refreshed_rows={len(refreshed_rows)} unique_game_ids={len(refreshed_ids)}"
        )
    return refreshed_ids


def _ordered_same_day_rows(games_rows, merged):
    order = []
    seen = set()
    for row in games_rows:
        game_id = (row.get("game_id") or "").strip()
        if game_id in merged and game_id not in seen:
            order.append(game_id)
            seen.add(game_id)
    return [merged[game_id] for game_id in order]


def _write_same_day_baseline(date_str, rows):
    baseline_path = BASELINE_DIR / f"{date_str}_MLB.csv"
    temp_path = baseline_path.with_suffix(baseline_path.suffix + ".tmp")
    write_csv(temp_path, OUTPUT_HEADER, rows)
    os.replace(temp_path, baseline_path)












def merge_with_same_day_baseline(
    date_str: str,
    refreshed_rows: list[dict],
    games_rows: list[dict],
    refreshed_eligible_count: int,
) -> tuple[list[dict], int]:
    if parse_date_value(date_str) != current_cutoff_date():
        return refreshed_rows, 0
    games_by_id = _same_day_games_by_id(date_str, games_rows)
    baseline_rows = load_same_day_baseline(date_str)
    if baseline_rows:
        validate_output_rows(date_str, 0, baseline_rows, allow_preserved=True)
    merged = _merge_preserved_baseline_rows(date_str, baseline_rows, games_by_id)
    baseline_ids = set(merged)
    refreshed_ids = _merge_refreshed_baseline_rows(
        date_str, refreshed_rows, games_by_id, merged
    )
    preserved_ids = sorted(baseline_ids - refreshed_ids)
    ordered_rows = _ordered_same_day_rows(games_rows, merged)
    validate_output_rows(
        date_str, refreshed_eligible_count, ordered_rows, allow_preserved=True
    )
    _write_same_day_baseline(date_str, ordered_rows)
    log(
        f"{date_str} | durable same-day baseline updated: rows={len(ordered_rows)} "
        f"refreshed={len(refreshed_ids)} preserved={len(preserved_ids)}"
    )
    if preserved_ids:
        log(
            f"{date_str} | preserved baseline predictions absent from latest refresh: "
            f"game_ids={preserved_ids}"
        )
    return ordered_rows, len(preserved_ids)

def load_same_day_fallback_output(date_str: str) -> list[dict]:
    """Return durable same-day state when games data is temporarily unavailable."""
    if parse_date_value(date_str) != current_cutoff_date():
        return []

    # load_same_day_baseline() already performs the one-time migration fallback
    # when no durable baseline exists. Do not consult the mutable output again
    # here: an existing-but-empty baseline must remain authoritative.
    rows = load_same_day_baseline(date_str)

    if rows:
        validate_output_rows(
            date_str,
            0,
            rows,
            allow_preserved=True,
        )

    return rows


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def load_preserved_same_day_state(
    date_str: str,
    games_path: Path,
) -> tuple[list[dict], int]:
    if games_path.exists():
        games_for_preserve = load_csv(
            games_path,
            REQUIRED_GAMES_COLS,
            "games input",
            required_file=False,
        )
        if games_for_preserve:
            return merge_with_same_day_baseline(
                date_str,
                [],
                games_for_preserve,
                0,
            )

    preserved_rows = load_same_day_fallback_output(date_str)
    return preserved_rows, len(preserved_rows)


def handle_current_future_games_unavailable(
    *,
    date_str: str,
    games_path: Path,
    unavailable_kind: str,
    pred_rows: list[dict],
    book_rows: list[dict],
    eligible_pred_indexes: set,
    sportsbook_presence: dict,
    eligible_count: int,
    rejection_path: Path,
    rejection_rows: list[dict],
    nonfatal_rejection_count: int,
    summary: dict,
) -> None:
    rejected_source_rows = {
        str(row.get("source_csv_row", ""))
        for row in rejection_rows
    }

    for idx in sorted(eligible_pred_indexes):
        source_csv_row = str(idx + 2)
        if source_csv_row in rejected_source_rows:
            continue

        pred_entry = {
            "row": pred_rows[idx],
            "index": idx,
            "csv_row": idx + 2,
        }
        presence = sportsbook_presence.get(
            idx,
            {"present": False, "detail": ""},
        )

        rejection_rows.append(
            make_rejection_row(
                pred_entry=pred_entry,
                reason=f"games_file_{unavailable_kind}_current_or_future",
                candidate_games=str(games_path),
                fatal=False,
                sportsbook_present=presence.get("present", False),
                sportsbook_match_detail=presence.get("detail", ""),
            )
        )
        rejected_source_rows.add(source_csv_row)
        nonfatal_rejection_count += 1

        log(
            f"{date_str} | non-fatal rejected prediction "
            f"because current/future games file is {unavailable_kind}: "
            f"csv_row={idx + 2} games_path={games_path}",
            "WARN",
        )

    if rejection_rows:
        write_csv(
            rejection_path,
            REJECTION_HEADER,
            rejection_rows,
        )
        print_rejection_rows(
            date_str,
            rejection_path,
            rejection_rows,
        )

    if parse_date_value(date_str) == current_cutoff_date():
        existing_rows = load_same_day_fallback_output(date_str)
        write_output_csv(
            date_str,
            OUTPUT_HEADER,
            existing_rows,
            summary,
        )
        summary["total_rows"] += len(existing_rows)
        output_action = (
            f"wrote_current_day_state rows={len(existing_rows)}"
        )
    else:
        output_action = "left_future_output_unchanged"

    log(
        f"{date_str} | current/future games file {unavailable_kind}. "
        f"games_path={games_path} "
        f"input_predictions={len(pred_rows)} "
        f"sportsbook_rows={len(book_rows)} "
        f"eligible_predictions={eligible_count} "
        f"nonfatal_rejections={nonfatal_rejection_count} "
        f"output_action={output_action}"
    )

    summary["rejected"] += len(rejection_rows)
    summary["nonfatal_rejections"] += nonfatal_rejection_count


def reject_blank_matched_game_id(
    *,
    date_str: str,
    label: str,
    pred_entry: dict,
    game_entry: dict,
    sportsbook_presence: dict,
    rejection_rows: list[dict],
) -> None:
    presence = sportsbook_presence.get(
        pred_entry["index"],
        {"present": False, "detail": ""},
    )

    rejection_rows.append(
        make_rejection_row(
            pred_entry=pred_entry,
            reason="matched_games_row_has_blank_game_id",
            candidate_games=describe_game_entry(game_entry),
            fatal=True,
            sportsbook_present=presence["present"],
            sportsbook_match_detail=presence.get("detail", ""),
        )
    )

    log(
        f"{date_str} | matched games row has blank game_id: "
        f"{label} games_csv_row={game_entry['csv_row']}",
        "ERROR",
    )


def record_output_match(
    output_by_pred_index: dict,
    pred_entry: dict,
    game_entry: dict,
    game_id: str,
) -> tuple[float | None, str, str]:
    output_by_pred_index[pred_entry["index"]] = make_output_row(
        pred_entry,
        game_id,
    )
    diff = minutes_between(
        pred_entry.get("dt"),
        game_entry.get("dt"),
    )
    diff_text = (
        ""
        if diff is None
        else f" diff_minutes={round(diff, 1)}"
    )
    level = (
        "WARN"
        if diff is not None and diff > MAX_TIME_DIFF_MINUTES
        else "INFO"
    )
    return diff, diff_text, level


def _prediction_date_paths(date_str):
    return (
        GAMES_DIR / f"{date_str}_games.csv",
        BOOK_DIR / f"{date_str}_MLB.csv",
        REJECTION_DIR / f"{date_str}_unmatched_predictions.csv",
    )


def _handle_no_prediction_rows(date_str, games_path, rejection_path, summary):
    durable_path = BASELINE_DIR / f"{date_str}_MLB.csv"
    existing_path = OUT_DIR / f"{date_str}_MLB.csv"
    current = parse_date_value(date_str) == current_cutoff_date()
    if current and (durable_path.exists() or existing_path.exists()):
        preserved_rows, preserved_count = load_preserved_same_day_state(
            date_str, games_path
        )
        validate_output_rows(date_str, 0, preserved_rows, allow_preserved=True)
        write_output_csv(date_str, OUTPUT_HEADER, preserved_rows, summary)
        clear_stale_rejection_file(rejection_path)
        log(
            f"{date_str} | no prediction rows; synchronized durable same-day state: "
            f"output_rows={len(preserved_rows)} preserved_rows={preserved_count}",
            "WARN",
        )
        summary["total_rows"] += len(preserved_rows)
        summary["skipped"] += 1
        return
    if current:
        clear_stale_rejection_file(rejection_path)
    log(f"{date_str} | no prediction rows ??? skipping", "WARN")
    summary["skipped"] += 1


def _prediction_sportsbook_state(date_str, pred_rows, book_rows, games_rows=None):
    pred_groups, pred_key_order = build_prediction_groups(pred_rows, date_str)
    book_groups = build_book_groups(book_rows, date_str) if book_rows else {}
    time_audit = _verify_prediction_times(
        date_str, pred_groups, book_groups, games_rows,
    )
    presence = build_sportsbook_presence(
        date_str=date_str,
        pred_groups=pred_groups,
        pred_key_order=pred_key_order,
        book_groups=book_groups,
    )
    rejection_rows = []
    nonfatal = 0
    for key in pred_key_order:
        for pred_entry in pred_groups.get(key, []):
            match = presence.get(pred_entry["index"], {"present": False, "detail": ""})
            if match["present"]:
                continue
            rejection_rows.append(
                make_rejection_row(
                    pred_entry=pred_entry,
                    reason="sportsbook_game_missing_for_prediction",
                    candidate_games="",
                    fatal=False,
                    sportsbook_present=False,
                    sportsbook_match_detail=match.get("detail", ""),
                )
            )
            nonfatal += 1
            log(
                f"{date_str} | non-fatal rejected prediction because sportsbook row is missing: "
                f"csv_row={pred_entry['csv_row']} {matchup_label(pred_entry['row'])} "
                f"pred_time={pred_entry['row'].get('game_time', '')}",
                "WARN",
            )
    eligible = {
        pred_entry["index"]
        for preds in pred_groups.values()
        for pred_entry in preds
        if presence.get(pred_entry["index"], {"present": False})["present"]
    }
    return pred_groups, pred_key_order, presence, rejection_rows, nonfatal, eligible, time_audit


def _handle_no_eligible_predictions(
    date_str, pred_rows, games_path, rejection_path, rejection_rows,
    nonfatal_rejection_count, summary,
):
    if rejection_rows:
        write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
        print_rejection_rows(date_str, rejection_path, rejection_rows)
    preserved_count = 0
    if parse_date_value(date_str) == current_cutoff_date():
        preserved_rows, preserved_count = load_preserved_same_day_state(date_str, games_path)
        write_output_csv(date_str, OUTPUT_HEADER, preserved_rows, summary)
        summary["total_rows"] += len(preserved_rows)
        output_action = f"wrote_current_day_state rows={len(preserved_rows)}"
    else:
        output_action = "left_noncurrent_output_unchanged"
    log(
        f"{date_str} | no sportsbook-eligible prediction rows. "
        f"input_predictions={len(pred_rows)} nonfatal_rejections={nonfatal_rejection_count} "
        f"preserved_rows={preserved_count} output_action={output_action}"
    )
    summary["rejected"] += len(rejection_rows)
    summary["nonfatal_rejections"] += nonfatal_rejection_count


def _load_prediction_games_rows(
    date_str, games_path, pred_rows, book_rows, eligible_pred_indexes,
    sportsbook_presence, eligible_count, rejection_path, rejection_rows,
    nonfatal_rejection_count, summary,
):
    unavailable = "missing" if not games_path.exists() else None
    if unavailable is None:
        games_rows = load_csv(
            games_path, REQUIRED_GAMES_COLS, "games input", required_file=True
        )
        unavailable = "empty" if not games_rows else None
    else:
        games_rows = []
    if unavailable is None:
        return games_rows
    if is_current_or_future_date(date_str):
        handle_current_future_games_unavailable(
            date_str=date_str,
            games_path=games_path,
            unavailable_kind=unavailable,
            pred_rows=pred_rows,
            book_rows=book_rows,
            eligible_pred_indexes=eligible_pred_indexes,
            sportsbook_presence=sportsbook_presence,
            eligible_count=eligible_count,
            rejection_path=rejection_path,
            rejection_rows=rejection_rows,
            nonfatal_rejection_count=nonfatal_rejection_count,
            summary=summary,
        )
        return None
    fail(
        f"{date_str} | past-date games file "
        f"{'missing' if unavailable == 'missing' else 'has zero rows'}: {games_path}"
    )


def _reject_prediction_group_without_games(
    date_str, label, preds, sportsbook_presence, rejection_rows,
):
    for pred_entry in preds:
        presence = sportsbook_presence.get(
            pred_entry["index"], {"present": False, "detail": ""}
        )
        rejection_rows.append(
            make_rejection_row(
                pred_entry=pred_entry,
                reason="no_games_row_for_same_home_away",
                candidate_games="",
                fatal=True,
                sportsbook_present=presence["present"],
                sportsbook_match_detail=presence.get("detail", ""),
            )
        )
        log(
            f"{date_str} | fatal unmatched sportsbook-eligible prediction no games row: "
            f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
            "ERROR",
        )
    return len(preds)


def _claim_matched_game_id(
    date_str,
    label,
    pred_entry,
    game_entry,
    sportsbook_presence,
    rejection_rows,
):
    game_entry["used"] = True
    game_id = (
        game_entry["row"].get("game_id")
        or ""
    ).strip()

    if game_id:
        return game_id

    reject_blank_matched_game_id(
        date_str=date_str,
        label=label,
        pred_entry=pred_entry,
        game_entry=game_entry,
        sportsbook_presence=sportsbook_presence,
        rejection_rows=rejection_rows,
    )
    return None


def _match_single_prediction_game(
    date_str, label, pred_entry, game_entry, sportsbook_presence,
    rejection_rows, output_by_pred_index,
):
    game_id = _claim_matched_game_id(
        date_str,
        label,
        pred_entry,
        game_entry,
        sportsbook_presence,
        rejection_rows,
    )
    if game_id is None:
        return 0, 1
    diff, diff_text, level = record_output_match(
        output_by_pred_index, pred_entry, game_entry, game_id
    )
    label_text = "MATCHED one-to-one"
    if diff is not None and diff > MAX_TIME_DIFF_MINUTES:
        level = "WARN"
        label_text = "MATCHED one-to-one with time mismatch"
    log(
        f"{date_str} | {label_text}: {label} "
        f"pred_time={pred_entry['row'].get('game_time', '')} "
        f"games_time={game_entry['row'].get('game_time', '')} "
        f"game_id={game_id}{diff_text}",
        level,
    )
    return 1, 0


def _match_ordered_prediction_games(
    date_str, label, preds, games, sportsbook_presence,
    rejection_rows, output_by_pred_index,
):
    sorted_preds = sorted(
        preds, key=lambda item: (dt_sort_value(item.get("dt")), item["index"])
    )
    sorted_games = sorted(
        games,
        key=lambda item: (
            dt_sort_value(item.get("dt")), item["game_number"], item["index"]
        ),
    )
    log(
        f"{date_str} | ORDER MATCH duplicate matchup: {label} "
        f"eligible_pred_count={len(sorted_preds)} games_count={len(sorted_games)}"
    )
    matched = 0
    fatal = 0
    for pred_entry, game_entry in zip(sorted_preds, sorted_games):
        game_id = _claim_matched_game_id(
            date_str,
            label,
            pred_entry,
            game_entry,
            sportsbook_presence,
            rejection_rows,
        )
        if game_id is None:
            fatal += 1
            continue
        diff, diff_text, level = record_output_match(
            output_by_pred_index, pred_entry, game_entry, game_id
        )
        if diff is not None and diff > MAX_TIME_DIFF_MINUTES:
            level = "WARN"
        matched += 1
        log(
            f"{date_str} | MATCHED order: {label} "
            f"pred_time={pred_entry['row'].get('game_time', '')} "
            f"games_time={game_entry['row'].get('game_time', '')} "
            f"gameNumber={game_entry['row'].get('gameNumber', '')} "
            f"game_id={game_id}{diff_text}", level,
        )
    return matched, fatal


def _match_closest_prediction_games(
    date_str, label, preds, games, sportsbook_presence,
    rejection_rows, output_by_pred_index,
):
    matched = 0
    fatal = 0
    sorted_preds = sorted(
        preds, key=lambda item: (dt_sort_value(item.get("dt")), item["index"])
    )
    for pred_entry in sorted_preds:
        available = [game for game in games if not game["used"]]
        if not available:
            presence = sportsbook_presence.get(
                pred_entry["index"], {"present": False, "detail": ""}
            )
            rejection_rows.append(make_rejection_row(
                pred_entry=pred_entry,
                reason="no_unused_games_row_for_same_home_away",
                candidate_games="", fatal=True,
                sportsbook_present=presence["present"],
                sportsbook_match_detail=presence.get("detail", ""),
            ))
            fatal += 1
            log(
                f"{date_str} | fatal unmatched sportsbook-eligible prediction no unused games row: "
                f"{label} pred_time={pred_entry['row'].get('game_time', '')}", "ERROR"
            )
            continue
        scored = [
            (minutes_between(pred_entry.get("dt"), game.get("dt")), game)
            for game in available
        ]
        valid = [item for item in scored if item[0] is not None]
        selected = None
        selected_diff = None
        if valid:
            selected_diff, selected = min(valid, key=lambda item: item[0])
            if selected_diff > MAX_TIME_DIFF_MINUTES:
                selected = None
        if selected is None:
            candidates = describe_candidates(scored)
            presence = sportsbook_presence.get(
                pred_entry["index"], {"present": False, "detail": ""}
            )
            rejection_rows.append(make_rejection_row(
                pred_entry=pred_entry,
                reason="no_games_row_within_time_threshold",
                candidate_games=candidates, fatal=True,
                sportsbook_present=presence["present"],
                sportsbook_match_detail=presence.get("detail", ""),
            ))
            fatal += 1
            log(
                f"{date_str} | fatal unmatched sportsbook-eligible prediction time threshold: "
                f"{label} pred_time={pred_entry['row'].get('game_time', '')} "
                f"candidate_games={candidates}", "ERROR"
            )
            continue
        game_id = _claim_matched_game_id(
            date_str,
            label,
            pred_entry,
            selected,
            sportsbook_presence,
            rejection_rows,
        )
        if game_id is None:
            fatal += 1
            continue
        output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
        matched += 1
        diff_text = "" if selected_diff is None else f" diff_minutes={round(selected_diff, 1)}"
        log(
            f"{date_str} | MATCHED closest: {label} "
            f"pred_time={pred_entry['row'].get('game_time', '')} "
            f"games_time={selected['row'].get('game_time', '')} "
            f"game_id={game_id}{diff_text}"
        )
    return matched, fatal


def _match_prediction_group(
    date_str, preds, games, sportsbook_presence, rejection_rows, output_by_pred_index,
):
    label = matchup_label(preds[0]["row"])
    if not games:
        return 0, _reject_prediction_group_without_games(
            date_str, label, preds, sportsbook_presence, rejection_rows
        )
    explicitly_matched = 0
    remaining = []
    for pred in preds:
        supplied_id = (pred["row"].get("game_id") or "").strip()
        if not supplied_id:
            remaining.append(pred)
            continue
        matched_games = [
            game for game in games
            if not game["used"] and game["row"].get("game_id", "").strip() == supplied_id
        ]
        if len(matched_games) != 1:
            fail(
                f"{date_str} | sportsbook-eligible supplied game_id cannot be "
                f"verified in authoritative games: {label} game_id={supplied_id} "
                f"matches={len(matched_games)}"
            )
        game = matched_games[0]
        actual_id = _claim_matched_game_id(
            date_str, label, pred, game, sportsbook_presence, rejection_rows,
        )
        if actual_id is None:
            return explicitly_matched, 1
        _, diff_text, level = record_output_match(
            output_by_pred_index, pred, game, actual_id,
        )
        log(
            f"{date_str} | MATCHED verified game ID: {label} "
            f"game_id={actual_id} pred_time={pred['row'].get('game_time', '')} "
            f"games_time={game['row'].get('game_time', '')}{diff_text}", level,
        )
        explicitly_matched += 1
    preds = remaining
    if not preds:
        return explicitly_matched, 0
    unused = [game for game in games if not game["used"]]
    if len(preds) == 1 and len(unused) == 1:
        m, f = _match_single_prediction_game(
            date_str, label, preds[0], unused[0], sportsbook_presence,
            rejection_rows, output_by_pred_index,
        )
        return explicitly_matched + m, f
    if 1 < len(preds) == len(unused):
        m, f = _match_ordered_prediction_games(
            date_str, label, preds, unused, sportsbook_presence,
            rejection_rows, output_by_pred_index,
        )
        return explicitly_matched + m, f
    m, f = _match_closest_prediction_games(
        date_str, label, preds, games, sportsbook_presence,
        rejection_rows, output_by_pred_index,
    )
    return explicitly_matched + m, f


def _match_predictions_to_games(
    date_str, pred_groups, pred_key_order, eligible_pred_indexes,
    games_groups, sportsbook_presence, rejection_rows,
):
    output = {}
    matched = 0
    fatal = 0
    for key in pred_key_order:
        preds = [
            entry for entry in pred_groups.get(key, [])
            if entry["index"] in eligible_pred_indexes
        ]
        if not preds:
            continue
        matched_delta, fatal_delta = _match_prediction_group(
            date_str, preds, games_groups.get(key, []), sportsbook_presence,
            rejection_rows, output,
        )
        matched += matched_delta
        fatal += fatal_delta
    return output, matched, fatal


def _log_unused_prediction_games(date_str, games_groups):
    for games in games_groups.values():
        for game_entry in [game for game in games if not game["used"]]:
            row = game_entry["row"]
            log(
                f"{date_str} | UNUSED games row: {row.get('away_team', '')} @ "
                f"{row.get('home_team', '')} game_id={row.get('game_id', '')} "
                f"game_time={row.get('game_time', '')}", "WARN"
            )


def _append_unprocessed_eligible_predictions(
    date_str, pred_rows, eligible_pred_indexes, output_by_pred_index,
    sportsbook_presence, rejection_rows,
):
    fatal = 0
    rejected_source_rows = {str(row.get("source_csv_row", "")) for row in rejection_rows}
    for idx in sorted(eligible_pred_indexes):
        if idx in output_by_pred_index or str(idx + 2) in rejected_source_rows:
            continue
        pred_entry = {"row": pred_rows[idx], "index": idx, "csv_row": idx + 2}
        presence = sportsbook_presence.get(idx, {"present": False, "detail": ""})
        rejection_rows.append(make_rejection_row(
            pred_entry=pred_entry,
            reason="eligible_prediction_row_not_processed",
            candidate_games="", fatal=True,
            sportsbook_present=presence["present"],
            sportsbook_match_detail=presence.get("detail", ""),
        ))
        fatal += 1
        log(
            f"{date_str} | eligible prediction row not processed: csv_row={idx + 2} "
            f"away={pred_rows[idx].get('away_team', '')} home={pred_rows[idx].get('home_team', '')}",
            "ERROR",
        )
    return fatal


def _finalize_prediction_date(
    date_str, pred_rows, book_rows, games_rows, eligible_count,
    output_by_pred_index, matched, rejection_rows, fatal_rejection_count,
    nonfatal_rejection_count, rejection_path, summary,
):
    if rejection_rows:
        write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
        print_rejection_rows(date_str, rejection_path, rejection_rows)
    if fatal_rejection_count:
        summary["rejected"] += len(rejection_rows)
        summary["fatal_rejections"] += fatal_rejection_count
        summary["nonfatal_rejections"] += nonfatal_rejection_count
        summary["errors"] += fatal_rejection_count
        fail(
            f"{date_str} | fatal sportsbook-eligible prediction rows could not be assigned game_id: "
            f"fatal_count={fatal_rejection_count} nonfatal_count={nonfatal_rejection_count} "
            f"rejection_csv={rejection_path}"
        )
    output_rows = [
        output_by_pred_index[idx]
        for idx in range(len(pred_rows))
        if idx in output_by_pred_index
    ]
    validate_output_rows(date_str, eligible_count, output_rows)
    output_rows, preserved_count = merge_with_same_day_baseline(
        date_str, output_rows, games_rows, eligible_count
    )
    validate_output_rows(date_str, eligible_count, output_rows, allow_preserved=True)
    written_path = write_output_csv(date_str, OUTPUT_HEADER, output_rows, summary)
    if not rejection_rows:
        clear_stale_rejection_file(rejection_path)
    log(
        f"{date_str} | WROTE: {written_path} | rows={len(output_rows)} matched={matched} "
        f"preserved_rows={preserved_count} input_predictions={len(pred_rows)} "
        f"sportsbook_rows={len(book_rows)} eligible_predictions={eligible_count} "
        f"nonfatal_rejections={nonfatal_rejection_count} fatal_rejections=0"
    )
    summary["total_rows"] += len(output_rows)
    summary["matched"] += matched
    summary["rejected"] += len(rejection_rows)
    summary["nonfatal_rejections"] += nonfatal_rejection_count






























def process_date(date_str: str, pred_path: Path, summary: dict) -> None:
    if parse_date_value(date_str) is None:
        fail(f"Could not parse prediction filename date: {date_str}")
    games_path, book_path, rejection_path = _prediction_date_paths(date_str)
    pred_rows = load_csv(pred_path, REQUIRED_PRED_COLS, "prediction input")
    book_rows = load_csv(
        book_path, REQUIRED_BOOK_COLS, "sportsbook input", required_file=False
    )
    if not pred_rows:
        _handle_no_prediction_rows(date_str, games_path, rejection_path, summary)
        return
    # Read schedule for verification; existing missing-games handling remains unchanged.
    schedule_rows = (
        load_csv(games_path, REQUIRED_GAMES_COLS, "games time verification")
        if games_path.exists() else []
    )
    (
        pred_groups, pred_key_order, sportsbook_presence,
        rejection_rows, nonfatal_rejection_count, eligible_pred_indexes,
        time_audit,
    ) = _prediction_sportsbook_state(date_str, pred_rows, book_rows, schedule_rows)
    eligible_count = len(eligible_pred_indexes)
    if eligible_count == 0:
        _handle_no_eligible_predictions(
            date_str, pred_rows, games_path, rejection_path,
            rejection_rows, nonfatal_rejection_count, summary,
        )
        return
    games_rows = _load_prediction_games_rows(
        date_str, games_path, pred_rows, book_rows, eligible_pred_indexes,
        sportsbook_presence, eligible_count, rejection_path, rejection_rows,
        nonfatal_rejection_count, summary,
    )
    if games_rows is None:
        return
    games_groups = build_games_groups(games_rows, date_str)
    output_by_pred_index, matched, fatal_rejection_count = _match_predictions_to_games(
        date_str, pred_groups, pred_key_order, eligible_pred_indexes,
        games_groups, sportsbook_presence, rejection_rows,
    )
    _log_unused_prediction_games(date_str, games_groups)
    fatal_rejection_count += _append_unprocessed_eligible_predictions(
        date_str, pred_rows, eligible_pred_indexes, output_by_pred_index,
        sportsbook_presence, rejection_rows,
    )
    _finalize_prediction_date(
        date_str, pred_rows, book_rows, games_rows, eligible_count,
        output_by_pred_index, matched, rejection_rows, fatal_rejection_count,
        nonfatal_rejection_count, rejection_path, summary,
    )
    _write_time_audit(date_str, time_audit)

def main():
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== game_id_pred RUN "
            f"{_now()} ===\n"
        )

    summary = {
        "files_written": 0,
        "total_rows": 0,
        "matched": 0,
        "rejected": 0,
        "nonfatal_rejections": 0,
        "fatal_rejections": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        pred_files = sorted(
            PRED_DIR.glob(
                "*_MLB.csv"
            )
        )

        pred_files = [
            p
            for p in pred_files
            if "pred_with_game_id"
            not in str(p)
        ]

        log(
            f"prediction files found: "
            f"{len(pred_files)}"
        )

        for pred_path in pred_files:
            date_str = (
                pred_path.stem.replace(
                    "_MLB",
                    "",
                )
            )

            process_date(
                date_str,
                pred_path,
                summary,
            )

    except Exception as e:
        tb = traceback.format_exc()

        log(
            f"FATAL: {e}\n{tb}",
            "ERROR",
        )

        print("")
        print("=" * 80)
        print(
            "FATAL ERROR IN game_id_pred.py"
        )
        print("=" * 80)
        print(f"error={e}")
        print("")
        print(tb)
        print("=" * 80)
        print("")

        if summary["errors"] == 0:
            summary["errors"] += 1

    status = (
        "SUCCESS"
        if (
            summary["errors"] == 0
            and summary[
                "fatal_rejections"
            ] == 0
        )
        else "FAILED"
    )

    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        (
            f"  files_written        : "
            f"{summary['files_written']}"
        ),
        (
            f"  total_rows           : "
            f"{summary['total_rows']}"
        ),
        (
            f"  matched              : "
            f"{summary['matched']}"
        ),
        (
            f"  rejected             : "
            f"{summary['rejected']}"
        ),
        (
            f"  nonfatal_rejections  : "
            f"{summary['nonfatal_rejections']}"
        ),
        (
            f"  fatal_rejections     : "
            f"{summary['fatal_rejections']}"
        ),
        (
            f"  skipped              : "
            f"{summary['skipped']}"
        ),
        (
            f"  errors               : "
            f"{summary['errors']}"
        ),
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
            + "\n"
        )

    print(
        f"game_id_pred complete. "
        f"{summary['files_written']} "
        f"files written. "
        f"matched="
        f"{summary['matched']} "
        f"rejected="
        f"{summary['rejected']} "
        f"nonfatal_rejections="
        f"{summary['nonfatal_rejections']} "
        f"fatal_rejections="
        f"{summary['fatal_rejections']} "
        f"errors="
        f"{summary['errors']} "
        f"Status: {status}"
    )

    if (
        summary["errors"] != 0
        or summary[
            "fatal_rejections"
        ] != 0
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
