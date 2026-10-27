#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/build_mlb_final_scores.py

import csv
import http.client
import json
import re
import traceback
from urllib.parse import urlsplit
from datetime import datetime, UTC
from pathlib import Path
from typing import Never
from zoneinfo import ZoneInfo

ERROR_DIR = Path("docs/win/baseball/mlb/errors/05_final_scores")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_mlb_final_scores.txt"

RAW_DIR = Path("docs/win/baseball/mlb/00_intake/drat_raw")
GAMES_DIR = Path("docs/win/baseball/mlb/00_intake/games")
PRED_DIR = Path("docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id")
SPORTSBOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")
FINAL_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/final_scores")
AUDIT_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/audit")

FINAL_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_AUDIT_FILE = AUDIT_DIR / "final_score_status_audit.csv"
KEY_AUDIT_FILE = AUDIT_DIR / "final_score_key_audit.csv"
UNRESOLVED_AUDIT_FILE = AUDIT_DIR / "unresolved_completed_games.csv"

RUN_TS = datetime.now(UTC).isoformat()

DOUBLEHEADER_TIME_TOLERANCE_MINUTES = 90
MLB_API_TIMEOUT_SECONDS = 20
MLB_API_USER_AGENT = "baseball_for_mat-final-score-builder/1.0"
MLB_API_HOST = "statsapi.mlb.com"

ET = ZoneInfo("America/New_York")

TEAM_KEY_ALIASES = {
    "oakland athletics": "athletics",
    "athletics": "athletics",
    "st louis cardinals": "st louis cardinals",
    "st. louis cardinals": "st louis cardinals",
}

with open(LOG_FILE, "w", encoding="utf-8") as startup_log:
    startup_log.write(f"=== build_mlb_final_scores RUN {RUN_TS} ===\n")


class FinalScoreConflictError(RuntimeError):
    """Fatal contradiction between records that identify the same game."""


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {msg}\n")


def fail(msg: str) -> None:
    log(f"FATAL: {msg}")
    raise RuntimeError(msg)


def fail_conflict(msg: str) -> Never:
    log(f"FATAL: {msg}")
    raise FinalScoreConflictError(msg)


def failure_context(
    *,
    source_file,
    game_date,
    game_time,
    away_team,
    home_team,
    game_id,
    game_pk,
):
    return (
        f"source_file={source_file} | "
        f"game_date={game_date} | "
        f"game_time={game_time} | "
        f"away_team={away_team} | "
        f"home_team={home_team} | "
        f"game_id={game_id} | "
        f"gamePk={game_pk}"
    )


def parse_datetime(dt_str):
    dt = datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M %p")
    return dt, dt.strftime("%Y_%m_%d"), dt.strftime("%I:%M %p")


def parse_time_minutes(value):
    value = str(value).strip()

    if not value:
        return None

    for fmt in ["%I:%M %p", "%H:%M:%S", "%H:%M"]:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.hour * 60 + parsed.minute
        except ValueError:
            continue

    return None


def clean_team(team_str):
    return str(team_str).split("(")[0].strip()


def normalize_team_key(team_str):
    cleaned = clean_team(team_str)
    lowered = cleaned.lower().strip()
    lowered = TEAM_KEY_ALIASES.get(lowered, lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def matchup_key(home_team, away_team):
    return (
        normalize_team_key(home_team),
        normalize_team_key(away_team),
    )


def normalize_status(raw_status):
    raw = str(raw_status or "").strip().lower()

    if raw in {
        "final",
        "game over",
        "completed",
        "complete",
        "completed early",
    }:
        return "final"

    if raw in {"postponed", "ppd"}:
        return "postponed"

    if raw in {"canceled", "cancelled"}:
        return "canceled"

    if raw in {"suspended"}:
        return "suspended"

    if raw in {"delayed", "delay"}:
        return "delayed"

    if raw in {"in progress", "live", "active"}:
        return "in_progress"

    if raw in {"scheduled", "pre-game", "pregame", "preview"}:
        return "scheduled"

    return "unknown"


def infer_game_status(row):
    explicit_status_fields = [
        "game_status",
        "status",
        "abstractGameState",
        "detailedState",
        "codedGameState",
        "statusCode",
    ]

    if isinstance(row, dict):
        for field in explicit_status_fields:
            val = row.get(field)

            if val not in (None, ""):
                return normalize_status(val), str(val).strip(), field, True

    if isinstance(row, list) and len(row) == 8:
        score_value = row[5]

        if isinstance(score_value, str):
            scores = score_value.split("\n")

            if len(scores) >= 2:
                try:
                    away_score = int(scores[0].strip())
                    home_score = int(scores[1].strip())
                except ValueError:
                    pass
                else:
                    if away_score == 0 and home_score == 0:
                        return (
                            "unknown",
                            "0-0",
                            "row_len_8_zero_zero_placeholder",
                            False,
                        )

        return "final", "final", "row_len_8_completed_score", False

    return "unknown", "unknown", "not_available_in_current_raw_shape", False


def is_completed_game(row):
    status_norm, _raw_status, _status_source, _status_available = infer_game_status(row)
    return status_norm == "final" and isinstance(row, list) and len(row) == 8


def raw_snapshot_date_from_path(path):
    suffix = "_mlb_raw.json"
    name = Path(path).name

    if not name.endswith(suffix):
        return ""

    return name[:-len(suffix)]


def et_utc_offset_minutes_for_date(game_date):
    try:
        local_dt = datetime.strptime(game_date, "%Y_%m_%d").replace(
            hour=12,
            tzinfo=ET,
        )
    except ValueError:
        return 0

    offset = local_dt.utcoffset()

    if offset is None:
        return 0

    return int(abs(offset.total_seconds()) // 60)


def time_match_targets(
    target_game_time,
    *,
    correction_minutes=0,
    prefer_correction=False,
):
    target_minutes = parse_time_minutes(target_game_time)

    if target_minutes is None:
        return []

    correction_minutes = int(correction_minutes or 0)

    if correction_minutes <= 0:
        return [(target_minutes, 0)]

    corrected_minutes = (target_minutes + correction_minutes) % (24 * 60)

    if corrected_minutes == target_minutes:
        return [(target_minutes, 0)]

    if prefer_correction:
        return [
            (corrected_minutes, 0),
            (target_minutes, 1),
        ]

    return [
        (target_minutes, 0),
        (corrected_minutes, 1),
    ]


def time_difference_minutes(first_time, second_time):
    first = parse_time_minutes(first_time)
    second = parse_time_minutes(second_time)

    if first is None or second is None:
        return None

    diff = abs(first - second)
    return min(diff, (24 * 60) - diff)


def closest_time_record_match(
    candidates,
    target_game_time,
    *,
    correction_minutes=0,
    prefer_correction=False,
):
    if not candidates:
        return {}

    targets = time_match_targets(
        target_game_time,
        correction_minutes=correction_minutes,
        prefer_correction=prefer_correction,
    )

    if not targets:
        return {}

    scored = []

    for candidate_index, candidate in enumerate(candidates):
        candidate_minutes = parse_time_minutes(candidate.get("game_time", ""))

        if candidate_minutes is None:
            continue

        candidate_best = None

        for target_minutes, target_priority in targets:
            diff = abs(candidate_minutes - target_minutes)
            diff = min(diff, (24 * 60) - diff)
            score = (diff, target_priority)

            if candidate_best is None or score < candidate_best:
                candidate_best = score

        if candidate_best is not None:
            scored.append((candidate_best, candidate_index, candidate))

    if not scored:
        return {}

    best_score = min(item[0] for item in scored)

    if best_score[0] > DOUBLEHEADER_TIME_TOLERANCE_MINUTES:
        return {}

    tied = [
        item
        for item in scored
        if item[0] == best_score
    ]

    if len(tied) != 1:
        return {}

    return tied[0][2]


def closest_time_match(
    candidates,
    target_game_time,
    value_field,
    *,
    correction_minutes=0,
    prefer_correction=False,
):
    match = closest_time_record_match(
        candidates,
        target_game_time,
        correction_minutes=correction_minutes,
        prefer_correction=prefer_correction,
    )

    if not match:
        return ""

    return match.get(value_field, "")


def closest_time_book_match(
    candidates,
    target_game_time,
    *,
    correction_minutes=0,
    prefer_correction=False,
):
    return closest_time_record_match(
        candidates,
        target_game_time,
        correction_minutes=correction_minutes,
        prefer_correction=prefer_correction,
    )


def _select_candidate_by_game_pk(
    candidates,
    target_game_time,
    current_game_pk,
    current_game_number,
):
    matches = [
        candidate
        for candidate in candidates
        if str(
            candidate.get("gamePk", "") or ""
        ).strip()
        == current_game_pk
    ]

    if len(matches) != 1:
        return {}, (
            "existing gamePk did not identify exactly one "
            "date/team candidate"
        )

    candidate = matches[0]
    candidate_game_number = str(
        candidate.get("gameNumber", "") or ""
    ).strip()

    if (
        current_game_number
        and candidate_game_number
        and candidate_game_number
        != current_game_number
    ):
        return {}, (
            "existing gamePk matched but gameNumber conflicted "
            f"(existing={current_game_number}, "
            f"candidate={candidate_game_number})"
        )

    if parse_time_minutes(target_game_time) is None:
        return candidate, (
            "gamePk+gameNumber+scheduled_time"
        )

    candidate_time = str(
        candidate.get("game_time", "") or ""
    ).strip()

    if parse_time_minutes(candidate_time) is None:
        return {}, (
            "existing gamePk matched but candidate scheduled "
            "time was unavailable"
        )

    diff = time_difference_minutes(
        target_game_time,
        candidate_time,
    )

    if (
        diff is None
        or diff
        > DOUBLEHEADER_TIME_TOLERANCE_MINUTES
    ):
        return {}, (
            "existing gamePk matched but scheduled time "
            f"was outside tolerance ({diff} minutes)"
        )

    return candidate, (
        "gamePk+gameNumber+scheduled_time"
    )


def _filter_candidates_by_game_number(
    candidates,
    current_game_number,
):
    if (
        len(candidates) <= 1
        or not current_game_number
    ):
        return candidates, ""

    matches = [
        candidate
        for candidate in candidates
        if str(
            candidate.get("gameNumber", "") or ""
        ).strip()
        == current_game_number
    ]

    if not matches:
        return [], (
            "doubleheader candidates existed but none matched "
            f"gameNumber={current_game_number}"
        )

    return matches, ""


def _resolve_candidate_by_time(
    candidates,
    target_game_time,
    current_game_number,
):
    if len(candidates) == 1:
        candidate = candidates[0]

        if parse_time_minutes(target_game_time) is None:
            if current_game_number:
                return candidate, (
                    "gameNumber_unique_no_time"
                )

            return {}, (
                "single date/team candidate existed but "
                "scheduled target time was unavailable"
            )

    matched = closest_time_record_match(
        candidates,
        target_game_time,
        correction_minutes=0,
        prefer_correction=False,
    )

    if matched:
        if current_game_number:
            return matched, (
                "gameNumber+scheduled_time"
            )

        return matched, "scheduled_time"

    if len(candidates) == 1:
        return {}, (
            "candidate failed scheduled-time tolerance"
        )

    if current_game_number:
        return {}, (
            "doubleheader gameNumber candidates remained "
            "ambiguous after scheduled-time matching"
        )

    return {}, (
        "same-team candidates could not be resolved "
        "uniquely by scheduled time"
    )


def select_game_candidate(
    candidates,
    target_game_time,
    *,
    current_game_pk="",
    current_game_number="",
):
    candidates = list(candidates or [])

    current_game_pk = str(
        current_game_pk or ""
    ).strip()

    current_game_number = str(
        current_game_number or ""
    ).strip()

    if not candidates:
        return {}, "no candidates"

    if current_game_pk:
        return _select_candidate_by_game_pk(
            candidates,
            target_game_time,
            current_game_pk,
            current_game_number,
        )

    pool, failure = (
        _filter_candidates_by_game_number(
            candidates,
            current_game_number,
        )
    )

    if failure:
        return {}, failure

    return _resolve_candidate_by_time(
        pool,
        target_game_time,
        current_game_number,
    )

def _load_identity_matchup_lookup(path, missing_message):
    lookup = {}

    if not path.exists():
        log(f"{missing_message}: {path}")
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            home_team = str(row.get("home_team", "") or "").strip()
            away_team = str(row.get("away_team", "") or "").strip()
            key = matchup_key(home_team, away_team)

            lookup.setdefault(key, []).append({
                "game_id": str(row.get("game_id", "") or "").strip(),
                "gamePk": str(row.get("gamePk", "") or "").strip(),
                "gameNumber": str(row.get("gameNumber", "") or "").strip(),
                "game_time": str(row.get("game_time", "") or "").strip(),
                "home_team": home_team,
                "away_team": away_team,
            })

    return lookup


def identity_fields(row):
    return (
        str(row.get("game_id", "") or "").strip(),
        str(row.get("gamePk", "") or "").strip(),
        str(row.get("gameNumber", "") or "").strip(),
        str(row.get("game_time", "") or "").strip(),
    )


def load_games_lookup(date):
    return _load_identity_matchup_lookup(
        GAMES_DIR / f"{date}_games.csv",
        "GAMES FILE MISSING FOR FINAL-SCORE GAME_ID/GAMEPK LOOKUP",
    )


def load_games_by_game_id(date):
    path = GAMES_DIR / f"{date}_games.csv"
    lookup = {}

    if not path.exists():
        log(f"GAMES FILE MISSING FOR FINAL-SCORE GAME_ID/GAMEPK LOOKUP: {path}")
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            game_id = str(row.get("game_id", "") or "").strip()

            if not game_id:
                continue

            if game_id in lookup:
                fail(
                    "Duplicate game_id in games file during final-score backfill: "
                    f"date={date} game_id={game_id}"
                )

            lookup[game_id] = {
                "game_id": game_id,
                "gamePk": str(row.get("gamePk", "") or "").strip(),
                "gameNumber": str(row.get("gameNumber", "") or "").strip(),
                "game_time": str(row.get("game_time", "") or "").strip(),
                "home_team": str(row.get("home_team", "") or "").strip(),
                "away_team": str(row.get("away_team", "") or "").strip(),
            }

    return lookup


def load_games_by_gamepk(date):
    path = GAMES_DIR / f"{date}_games.csv"
    lookup = {}

    if not path.exists():
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            game_pk = str(row.get("gamePk", "") or "").strip()

            if not game_pk:
                continue

            if game_pk in lookup:
                fail(
                    "Duplicate gamePk in games file during final-score backfill: "
                    f"date={date} gamePk={game_pk}"
                )

            lookup[game_pk] = {
                "game_id": str(row.get("game_id", "") or "").strip(),
                "gamePk": game_pk,
                "gameNumber": str(row.get("gameNumber", "") or "").strip(),
                "game_time": str(row.get("game_time", "") or "").strip(),
                "home_team": str(row.get("home_team", "") or "").strip(),
                "away_team": str(row.get("away_team", "") or "").strip(),
            }

    return lookup


def load_predictions_lookup(date):
    return _load_identity_matchup_lookup(
        PRED_DIR / f"{date}_MLB.csv",
        "PREDICTION FILE MISSING FOR FINAL-SCORE GAME_ID LOOKUP",
    )


def load_sportsbook_lookup(date):
    path = SPORTSBOOK_DIR / f"{date}_MLB.csv"
    lookup = {}

    if not path.exists():
        log(f"SPORTSBOOK FILE MISSING FOR FINAL-SCORE MARKET-LINE LOOKUP: {path}")
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for r in reader:
            home_team = str(r.get("home_team", "") or "").strip()
            away_team = str(r.get("away_team", "") or "").strip()
            key = matchup_key(home_team, away_team)

            lookup.setdefault(key, []).append({
                "game_time": str(r.get("game_time", "") or "").strip(),
                "away_run_line": r.get("away_run_line"),
                "home_run_line": r.get("home_run_line"),
                "total": r.get("total"),
            })

    return lookup


def candidate_matches_teams(candidate, home_team, away_team):
    if not candidate:
        return False

    return matchup_key(
        candidate.get("home_team", ""),
        candidate.get("away_team", ""),
    ) == matchup_key(home_team, away_team)


def _resolved_game_result(
    candidate,
    source,
    game_time,
    games_candidate_count,
    prediction_candidate_count,
):
    game_id = _identity_value(
        candidate,
        "game_id",
    )
    game_pk = _identity_value(
        candidate,
        "gamePk",
    )

    return {
        "resolved": bool(game_id and game_pk),
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": _identity_value(
            candidate,
            "gameNumber",
        ),
        "scheduled_game_time": str(
            candidate.get("game_time", "")
            or game_time
            or ""
        ).strip(),
        "resolution_source": source,
        "games_candidate_count": games_candidate_count,
        "prediction_candidate_count": prediction_candidate_count,
        "reason": "",
    }


def _resolve_existing_game_pk(
    *,
    candidates,
    game_time,
    current_game_id,
    current_game_pk,
    current_game_number,
    games_candidate_count,
    prediction_candidate_count,
):
    if not current_game_pk:
        return None

    match, reason = select_game_candidate(
        candidates,
        game_time,
        current_game_pk=current_game_pk,
        current_game_number=current_game_number,
    )

    if not match:
        return None

    resolved = _resolved_game_result(
        match,
        f"games_existing_gamePk_{reason}",
        game_time,
        games_candidate_count,
        prediction_candidate_count,
    )

    if (
        not resolved["game_id"]
        and current_game_id
    ):
        resolved["game_id"] = current_game_id
        resolved["resolved"] = bool(
            resolved["gamePk"]
            and resolved["game_id"]
        )

    return resolved


def _resolve_existing_game_id(
    *,
    games_by_game_id,
    game_time,
    home_team,
    away_team,
    current_game_id,
    current_game_number,
    games_candidate_count,
    prediction_candidate_count,
):
    if not current_game_id:
        return None

    candidate = games_by_game_id.get(
        current_game_id,
        {},
    )

    if not candidate:
        return None

    if not candidate_matches_teams(
        candidate,
        home_team,
        away_team,
    ):
        return None

    candidate_game_pk = _identity_value(
        candidate,
        "gamePk",
    )
    candidate_game_number = _identity_value(
        candidate,
        "gameNumber",
    )

    match, reason = select_game_candidate(
        [candidate],
        game_time,
        current_game_pk=candidate_game_pk,
        current_game_number=(
            current_game_number
            or candidate_game_number
        ),
    )

    if not match:
        return None

    return _resolved_game_result(
        match,
        f"games_existing_game_id_{reason}",
        game_time,
        games_candidate_count,
        prediction_candidate_count,
    )


def _official_from_prediction_game_pk(
    *,
    games_by_gamepk,
    pred_game_pk,
    pred_game_number,
    pred_game_time,
    home_team,
    away_team,
    source_prefix,
    games_candidate_count,
    prediction_candidate_count,
):
    if not pred_game_pk:
        return None

    official = games_by_gamepk.get(
        pred_game_pk,
        {},
    )

    if not official:
        return None

    if not candidate_matches_teams(
        official,
        home_team,
        away_team,
    ):
        return None

    match, reason = select_game_candidate(
        [official],
        pred_game_time,
        current_game_pk=pred_game_pk,
        current_game_number=pred_game_number,
    )

    if not match:
        return None

    return _resolved_game_result(
        match,
        (
            f"{source_prefix}"
            f"_then_games_by_gamePk_{reason}"
        ),
        pred_game_time,
        games_candidate_count,
        prediction_candidate_count,
    )


def _official_from_prediction_game_id(
    *,
    games_by_game_id,
    pred_game_id,
    pred_game_number,
    pred_game_time,
    home_team,
    away_team,
    source_prefix,
    games_candidate_count,
    prediction_candidate_count,
):
    if not pred_game_id:
        return None

    official = games_by_game_id.get(
        pred_game_id,
        {},
    )

    if not official:
        return None

    if not candidate_matches_teams(
        official,
        home_team,
        away_team,
    ):
        return None

    official_game_pk = _identity_value(
        official,
        "gamePk",
    )

    match, reason = select_game_candidate(
        [official],
        pred_game_time,
        current_game_pk=official_game_pk,
        current_game_number=pred_game_number,
    )

    if not match:
        return None

    return _resolved_game_result(
        match,
        (
            f"{source_prefix}"
            f"_then_games_by_game_id_{reason}"
        ),
        pred_game_time,
        games_candidate_count,
        prediction_candidate_count,
    )


def _resolve_prediction_candidate(
    *,
    pred_match,
    pred_match_reason,
    games_candidates,
    games_by_game_id,
    games_by_gamepk,
    game_time,
    home_team,
    away_team,
    games_candidate_count,
    prediction_candidate_count,
):
    pred_game_id = _identity_value(
        pred_match,
        "game_id",
    )
    pred_game_pk = _identity_value(
        pred_match,
        "gamePk",
    )
    pred_game_number = _identity_value(
        pred_match,
        "gameNumber",
    )
    pred_game_time = str(
        pred_match.get("game_time", "")
        or game_time
        or ""
    ).strip()

    source_prefix = (
        f"predictions_{pred_match_reason}"
    )

    resolved = _official_from_prediction_game_pk(
        games_by_gamepk=games_by_gamepk,
        pred_game_pk=pred_game_pk,
        pred_game_number=pred_game_number,
        pred_game_time=pred_game_time,
        home_team=home_team,
        away_team=away_team,
        source_prefix=source_prefix,
        games_candidate_count=games_candidate_count,
        prediction_candidate_count=prediction_candidate_count,
    )

    if resolved is not None:
        return resolved

    resolved = _official_from_prediction_game_id(
        games_by_game_id=games_by_game_id,
        pred_game_id=pred_game_id,
        pred_game_number=pred_game_number,
        pred_game_time=pred_game_time,
        home_team=home_team,
        away_team=away_team,
        source_prefix=source_prefix,
        games_candidate_count=games_candidate_count,
        prediction_candidate_count=prediction_candidate_count,
    )

    if resolved is not None:
        return resolved

    official, reason = select_game_candidate(
        games_candidates,
        pred_game_time,
        current_game_pk=pred_game_pk,
        current_game_number=pred_game_number,
    )

    if official:
        return _resolved_game_result(
            official,
            (
                f"{source_prefix}"
                f"_then_games_matchup_{reason}"
            ),
            pred_game_time,
            games_candidate_count,
            prediction_candidate_count,
        )

    return {
        "resolved": bool(
            pred_game_id
            and pred_game_pk
        ),
        "game_id": pred_game_id,
        "gamePk": pred_game_pk,
        "gameNumber": pred_game_number,
        "scheduled_game_time": pred_game_time,
        "resolution_source": source_prefix,
        "games_candidate_count": games_candidate_count,
        "prediction_candidate_count": prediction_candidate_count,
        "reason": (
            "prediction candidate resolved, but the corresponding "
            "official games row could not be verified using "
            "gamePk, gameNumber, and scheduled time"
        ),
    }


def _unresolved_resolution_result(
    *,
    game_time,
    current_game_id,
    current_game_pk,
    current_game_number,
    games_candidates,
    pred_candidates,
):
    reason_parts = []

    if not games_candidates:
        reason_parts.append(
            "no normalized date/team candidate in games"
        )
    else:
        reason_parts.append(
            "games candidates existed but gamePk/gameNumber/"
            "scheduled-time resolution was not unique"
        )

    if not pred_candidates:
        reason_parts.append(
            "no normalized date/team candidate in predictions"
        )
    else:
        reason_parts.append(
            "prediction candidates existed but gamePk/gameNumber/"
            "scheduled-time resolution was not unique"
        )

    if current_game_id:
        reason_parts.append(
            "existing game_id could not be verified against "
            "gamePk/gameNumber/scheduled time"
        )

    if current_game_pk:
        reason_parts.append(
            "existing gamePk could not be verified against "
            "date/team/gameNumber/scheduled time"
        )

    return {
        "resolved": False,
        "game_id": current_game_id,
        "gamePk": current_game_pk,
        "gameNumber": current_game_number,
        "scheduled_game_time": str(
            game_time or ""
        ).strip(),
        "resolution_source": "unresolved",
        "games_candidate_count": len(
            games_candidates
        ),
        "prediction_candidate_count": len(
            pred_candidates
        ),
        "reason": "; ".join(reason_parts),
    }


def resolve_completed_game_ids(
    *,
    game_time,
    home_team,
    away_team,
    current_game_id="",
    current_game_pk="",
    current_game_number="",
    games_lookup,
    games_by_game_id,
    games_by_gamepk,
    predictions_lookup,
):
    key = matchup_key(
        home_team,
        away_team,
    )

    games_candidates = games_lookup.get(
        key,
        [],
    )
    pred_candidates = predictions_lookup.get(
        key,
        [],
    )

    current_game_id = str(
        current_game_id or ""
    ).strip()
    current_game_pk = str(
        current_game_pk or ""
    ).strip()
    current_game_number = str(
        current_game_number or ""
    ).strip()

    counts = {
        "games_candidate_count": len(
            games_candidates
        ),
        "prediction_candidate_count": len(
            pred_candidates
        ),
    }

    resolved = _resolve_existing_game_pk(
        candidates=games_candidates,
        game_time=game_time,
        current_game_id=current_game_id,
        current_game_pk=current_game_pk,
        current_game_number=current_game_number,
        **counts,
    )

    if resolved is not None:
        return resolved

    resolved = _resolve_existing_game_id(
        games_by_game_id=games_by_game_id,
        game_time=game_time,
        home_team=home_team,
        away_team=away_team,
        current_game_id=current_game_id,
        current_game_number=current_game_number,
        **counts,
    )

    if resolved is not None:
        return resolved

    match, reason = select_game_candidate(
        games_candidates,
        game_time,
        current_game_number=current_game_number,
    )

    if match:
        return _resolved_game_result(
            match,
            f"games_date_teams_{reason}",
            game_time,
            **counts,
        )

    pred_match, pred_reason = select_game_candidate(
        pred_candidates,
        game_time,
        current_game_pk=current_game_pk,
        current_game_number=current_game_number,
    )

    if pred_match:
        return _resolve_prediction_candidate(
            pred_match=pred_match,
            pred_match_reason=pred_reason,
            games_candidates=games_candidates,
            games_by_game_id=games_by_game_id,
            games_by_gamepk=games_by_gamepk,
            game_time=game_time,
            home_team=home_team,
            away_team=away_team,
            **counts,
        )

    return _unresolved_resolution_result(
        game_time=game_time,
        current_game_id=current_game_id,
        current_game_pk=current_game_pk,
        current_game_number=current_game_number,
        games_candidates=games_candidates,
        pred_candidates=pred_candidates,
    )

def make_unresolved_completed_row(
    *,
    source_file,
    row_index,
    game_date,
    game_time,
    away_team,
    home_team,
    final_away_score,
    final_home_score,
    game_id,
    game_pk,
    game_number,
    games_candidate_count,
    prediction_candidate_count,
    resolution_reason,
    raw_row,
):
    return {
        "source_file": source_file,
        "row_index": row_index,
        "game_date": game_date,
        "game_time": game_time,
        "away_team": away_team,
        "home_team": home_team,
        "final_away_score": final_away_score,
        "final_home_score": final_home_score,
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "games_candidate_count": games_candidate_count,
        "prediction_candidate_count": prediction_candidate_count,
        "resolution_reason": resolution_reason,
        "raw_row": raw_row,
    }


SUMMARY_ROW_PREFIXES = {"Sportsbooks", "DRatings"}

FINAL_HEADER = [
    "sport",
    "league",
    "game_id",
    "gamePk",
    "gameNumber",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "final_away_score",
    "final_home_score",
    "final_total",
    "away_run_line",
    "home_run_line",
    "total",
    "game_status",
    "final_scores_generated_at",
]


def is_summary_row(row):
    return (
        row
        and isinstance(row, list)
        and str(row[0]).strip() in SUMMARY_ROW_PREFIXES
    )


def write_csv(path, header, rows, files_written, label):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, *rows])

    row_count = len(rows)
    files_written.append((str(path), row_count))
    log(f"WROTE {label} -> {path} ({row_count} rows)")


def write_audit_csv(path, header, rows, label):
    safe_path = Path(path).resolve()
    allowed_root = AUDIT_DIR.resolve()

    if not safe_path.is_relative_to(
        allowed_root
    ):
        fail(
            "Refusing final-score audit output "
            f"outside trusted directory: {path}"
        )

    if safe_path.suffix.lower() != ".csv":
        fail(
            f"Refusing non-CSV audit output: {path}"
        )

    path = safe_path

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
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                col: row.get(col, "")
                for col in header
            })

    log(
        f"WROTE {label} -> {path} "
        f"({len(rows)} rows)"
    )


def raw_row_text(row):
    try:
        return json.dumps(
            row,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        return repr(row)


def make_parse_error_row(
    *,
    source_file,
    row_index,
    stage,
    error,
    row,
):
    return {
        "source_file": source_file,
        "row_index": row_index,
        "stage": stage,
        "error": str(error),
        "raw_row": raw_row_text(row),
    }


def log_review_rows(
    parse_error_rows,
    unresolved_completed_rows,
):
    log("--- PARSE ERROR ROWS FOR REVIEW ---")

    if not parse_error_rows:
        log("None")
    else:
        for item in parse_error_rows:
            log(
                "PARSE_ERROR | "
                f"source_file={item.get('source_file', '')} | "
                f"row_index={item.get('row_index', '')} | "
                f"stage={item.get('stage', '')} | "
                f"error={item.get('error', '')} | "
                f"raw_row={item.get('raw_row', '')}"
            )

    log("--- UNRESOLVED COMPLETED GAMES FOR REVIEW ---")

    if not unresolved_completed_rows:
        log("None")
    else:
        for item in unresolved_completed_rows:
            log(
                "UNRESOLVED_COMPLETED_GAME | "
                f"source_file={item.get('source_file', '')} | "
                f"row_index={item.get('row_index', '')} | "
                f"game_date={item.get('game_date', '')} | "
                f"game_time={item.get('game_time', '')} | "
                f"away_team={item.get('away_team', '')} | "
                f"home_team={item.get('home_team', '')} | "
                f"final_away_score={item.get('final_away_score', '')} | "
                f"final_home_score={item.get('final_home_score', '')} | "
                f"game_id={item.get('game_id', '')} | "
                f"gamePk={item.get('gamePk', '')} | "
                f"gameNumber={item.get('gameNumber', '')} | "
                f"games_candidate_count={item.get('games_candidate_count', '')} | "
                f"prediction_candidate_count="
                f"{item.get('prediction_candidate_count', '')} | "
                f"resolution_reason={item.get('resolution_reason', '')} | "
                f"raw_row={item.get('raw_row', '')}"
            )


def final_row_signature(record):
    return (
        str(record.get("sport", "") or "").strip(),
        str(record.get("league", "") or "").strip(),
        str(record.get("game_date", "") or "").strip(),
        normalize_team_key(record.get("home_team", "")),
        normalize_team_key(record.get("away_team", "")),
        str(record.get("final_away_score", "") or "").strip(),
        str(record.get("final_home_score", "") or "").strip(),
        str(record.get("final_total", "") or "").strip(),
        str(record.get("game_status", "") or "").strip(),
    )


def _identity_value(record, key):
    return str(
        record.get(key, "") or ""
    ).strip()


def _different_nonblank(left, right):
    return bool(
        left
        and right
        and left != right
    )


def _same_nonblank(left, right):
    return bool(
        left
        and right
        and left == right
    )


def _scheduled_time_conflict(existing, incoming):
    existing_time = _identity_value(
        existing,
        "game_time",
    )
    incoming_time = _identity_value(
        incoming,
        "game_time",
    )

    diff = time_difference_minutes(
        existing_time,
        incoming_time,
    )

    conflict = (
        diff is not None
        and diff
        > DOUBLEHEADER_TIME_TOLERANCE_MINUTES
    )

    return (
        conflict,
        existing_time,
        incoming_time,
    )


def game_identity_conflict_reason(existing, incoming):
    existing_game_pk = _identity_value(
        existing,
        "gamePk",
    )
    incoming_game_pk = _identity_value(
        incoming,
        "gamePk",
    )

    if _different_nonblank(
        existing_game_pk,
        incoming_game_pk,
    ):
        return (
            "same game_id mapped to different gamePk values "
            f"({existing_game_pk} vs {incoming_game_pk})"
        )

    existing_game_number = _identity_value(
        existing,
        "gameNumber",
    )
    incoming_game_number = _identity_value(
        incoming,
        "gameNumber",
    )

    if _different_nonblank(
        existing_game_number,
        incoming_game_number,
    ):
        return (
            "same game_id mapped to different gameNumber values "
            f"({existing_game_number} vs {incoming_game_number})"
        )

    same_identity = (
        _same_nonblank(
            existing_game_pk,
            incoming_game_pk,
        )
        and _same_nonblank(
            existing_game_number,
            incoming_game_number,
        )
    )

    if not same_identity:
        return ""

    (
        time_conflict,
        existing_time,
        incoming_time,
    ) = _scheduled_time_conflict(
        existing,
        incoming,
    )

    if time_conflict:
        return (
            "same game_id/gamePk/gameNumber had incompatible "
            f"scheduled times ({existing_time} vs {incoming_time})"
        )

    return ""

def merge_duplicate_metadata(existing, record):
    existing_gamepk = str(existing.get("gamePk", "") or "").strip()
    incoming_gamepk = str(record.get("gamePk", "") or "").strip()

    for field in (
        "gamePk",
        "gameNumber",
        "away_run_line",
        "home_run_line",
        "total",
    ):
        if not str(existing.get(field, "") or "").strip():
            incoming = record.get(field, "")

            if str(incoming or "").strip():
                existing[field] = incoming

    if not existing_gamepk and incoming_gamepk:
        incoming_time = str(record.get("game_time", "") or "").strip()

        if incoming_time:
            existing["game_time"] = incoming_time

    return existing


def make_key_audit_row(
    *,
    game_date,
    game_id,
    game_pk,
    game_number,
    away_team,
    home_team,
    duplicate_count,
    status,
    notes,
):
    return {
        "game_date": game_date,
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "away_team": away_team,
        "home_team": home_team,
        "duplicate_count": duplicate_count,
        "status": status,
        "notes": notes,
    }


def _add_record_with_game_id(
    *,
    record,
    source_file,
    final_records_by_date,
    seen_by_game_id,
    key_audit_rows,
    game_id,
    game_pk,
    game_number,
    game_date,
    game_time,
    home_team,
    away_team,
):
    existing = seen_by_game_id.get(game_id)

    if existing is None:
        seen_by_game_id[game_id] = record
        final_records_by_date.setdefault(
            game_date,
            [],
        ).append(record)

        key_audit_rows.append(
            make_key_audit_row(
                game_date=game_date,
                game_id=game_id,
                game_pk=game_pk,
                game_number=game_number,
                away_team=away_team,
                home_team=home_team,
                duplicate_count=1,
                status="unique_game_id",
                notes="accepted; primary key game_id",
            )
        )

        return "accepted"

    identity_conflict = game_identity_conflict_reason(
        existing,
        record,
    )

    if identity_conflict:
        key_audit_rows.append(
            make_key_audit_row(
                game_date=game_date,
                game_id=game_id,
                game_pk=game_pk,
                game_number=game_number,
                away_team=away_team,
                home_team=home_team,
                duplicate_count=2,
                status="conflicting_duplicate_game_identity",
                notes=identity_conflict,
            )
        )

        context = failure_context(
            source_file=source_file,
            game_date=game_date,
            game_time=game_time,
            away_team=away_team,
            home_team=home_team,
            game_id=game_id,
            game_pk=game_pk,
        )

        existing_source_file = str(
            existing.get("_source_file", "") or ""
        ).strip()

        fail_conflict(
            "Conflicting final-score game identity found | "
            f"{context} | "
            f"gameNumber={game_number} | "
            f"reason={identity_conflict} | "
            f"existing_source_file={existing_source_file}"
        )

    if (
        final_row_signature(existing)
        == final_row_signature(record)
    ):
        merge_duplicate_metadata(
            existing,
            record,
        )

        key_audit_rows.append(
            make_key_audit_row(
                game_date=game_date,
                game_id=game_id,
                game_pk=game_pk,
                game_number=game_number,
                away_team=away_team,
                home_team=home_team,
                duplicate_count=2,
                status="identical_duplicate_collapsed",
                notes=(
                    "duplicate game_id row was identical "
                    "and had compatible gamePk/gameNumber/time"
                ),
            )
        )

        return "duplicate_collapsed"

    key_audit_rows.append(
        make_key_audit_row(
            game_date=game_date,
            game_id=game_id,
            game_pk=game_pk,
            game_number=game_number,
            away_team=away_team,
            home_team=home_team,
            duplicate_count=2,
            status="conflicting_duplicate_game_id",
            notes=(
                "same game_id had conflicting "
                "final-score fields"
            ),
        )
    )

    context = failure_context(
        source_file=source_file,
        game_date=game_date,
        game_time=game_time,
        away_team=away_team,
        home_team=home_team,
        game_id=game_id,
        game_pk=game_pk,
    )

    existing_source_file = str(
        existing.get("_source_file", "") or ""
    ).strip()

    fail_conflict(
        "Conflicting final-score duplicate game_id found | "
        f"{context} | "
        f"gameNumber={game_number} | "
        f"existing_source_file={existing_source_file}"
    )


def _add_record_without_game_id(
    *,
    record,
    source_file,
    final_records_by_date,
    seen_by_fallback_key,
    key_audit_rows,
    game_pk,
    game_number,
    game_date,
    game_time,
    home_team,
    away_team,
):
    fallback_key = (
        game_date,
        normalize_team_key(home_team),
        normalize_team_key(away_team),
        game_pk,
        game_number,
        game_time,
    )

    fallback_notes = (
        "game_id missing; fallback date/team/gamePk/gameNumber/time "
        "key used so same-team doubleheaders cannot collapse"
    )

    existing = seen_by_fallback_key.get(
        fallback_key
    )

    if existing is None:
        seen_by_fallback_key[fallback_key] = record
        final_records_by_date.setdefault(
            game_date,
            [],
        ).append(record)

        key_audit_rows.append(
            make_key_audit_row(
                game_date=game_date,
                game_id="",
                game_pk=game_pk,
                game_number=game_number,
                away_team=away_team,
                home_team=home_team,
                duplicate_count=1,
                status=(
                    "blank_game_id_written_for_downstream_audit"
                ),
                notes=fallback_notes,
            )
        )

        return "accepted_blank_game_id"

    if (
        final_row_signature(existing)
        == final_row_signature(record)
    ):
        merge_duplicate_metadata(
            existing,
            record,
        )

        key_audit_rows.append(
            make_key_audit_row(
                game_date=game_date,
                game_id="",
                game_pk=game_pk,
                game_number=game_number,
                away_team=away_team,
                home_team=home_team,
                duplicate_count=2,
                status=(
                    "blank_game_id_identical_duplicate_collapsed"
                ),
                notes=(
                    "blank-game_id duplicate had matching "
                    "gamePk/gameNumber/time and was not written twice"
                ),
            )
        )

        return "blank_game_id_duplicate_collapsed"

    key_audit_rows.append(
        make_key_audit_row(
            game_date=game_date,
            game_id="",
            game_pk=game_pk,
            game_number=game_number,
            away_team=away_team,
            home_team=home_team,
            duplicate_count=2,
            status="blank_game_id_conflicting_duplicate",
            notes=(
                "blank-game_id duplicate fallback identity had "
                "conflicting final-score fields"
            ),
        )
    )

    context = failure_context(
        source_file=source_file,
        game_date=game_date,
        game_time=game_time,
        away_team=away_team,
        home_team=home_team,
        game_id="",
        game_pk=game_pk,
    )

    existing_source_file = str(
        existing.get("_source_file", "") or ""
    ).strip()

    fail_conflict(
        "Conflicting blank-game_id final-score duplicate found | "
        f"{context} | "
        f"gameNumber={game_number} | "
        f"existing_source_file={existing_source_file}"
    )

    return "failed"


def add_final_record(
    *,
    record,
    source_file,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    key_audit_rows,
):
    game_id = str(
        record.get("game_id", "") or ""
    ).strip()
    game_pk = str(
        record.get("gamePk", "") or ""
    ).strip()
    game_number = str(
        record.get("gameNumber", "") or ""
    ).strip()
    game_date = str(
        record.get("game_date", "") or ""
    ).strip()
    game_time = str(
        record.get("game_time", "") or ""
    ).strip()
    home_team = str(
        record.get("home_team", "") or ""
    ).strip()
    away_team = str(
        record.get("away_team", "") or ""
    ).strip()

    record["_source_file"] = source_file

    common = {
        "record": record,
        "source_file": source_file,
        "final_records_by_date": final_records_by_date,
        "key_audit_rows": key_audit_rows,
        "game_pk": game_pk,
        "game_number": game_number,
        "game_date": game_date,
        "game_time": game_time,
        "home_team": home_team,
        "away_team": away_team,
    }

    if game_id:
        return _add_record_with_game_id(
            **common,
            seen_by_game_id=seen_by_game_id,
            game_id=game_id,
        )

    return _add_record_without_game_id(
        **common,
        seen_by_fallback_key=seen_by_fallback_key,
    )

def legacy_final_date_from_path(path):
    suffix = "_final_scores_MLB.csv"
    name = path.name

    if not name.endswith(suffix):
        return ""

    return name[:-len(suffix)]


def games_date_from_path(path):
    suffix = "_games.csv"
    name = path.name

    if not name.endswith(suffix):
        return ""

    return name[:-len(suffix)]


def legacy_row_has_final_score(row):
    try:
        away_score = int(
            str(row.get("final_away_score", "")).strip()
        )
        home_score = int(
            str(row.get("final_home_score", "")).strip()
        )
    except (TypeError, ValueError):
        return False

    return away_score >= 0 and home_score >= 0


def _prepare_existing_final_record(row, date):
    record = {
        col: str(
            row.get(col, "") or ""
        ).strip()
        for col in FINAL_HEADER
    }

    record["sport"] = (
        record["sport"]
        or "baseball"
    )
    record["league"] = (
        record["league"]
        or "mlb"
    )
    record["game_date"] = (
        record["game_date"]
        or date
    )

    if (
        not record["game_status"]
        and legacy_row_has_final_score(record)
    ):
        record["game_status"] = "final"

    completed = (
        record["game_status"].strip().lower()
        == "final"
        and legacy_row_has_final_score(record)
    )

    if not completed:
        return record, "not_final"

    if not record["final_total"]:
        record["final_total"] = str(
            int(record["final_away_score"])
            + int(record["final_home_score"])
        )

    if not record["final_scores_generated_at"]:
        record[
            "final_scores_generated_at"
        ] = RUN_TS

    if (
        not record["game_id"]
        or not record["gamePk"]
    ):
        return record, "missing_ids"

    return record, "ready"


def _append_existing_final_status(
    status_audit_rows,
    record,
):
    status_audit_rows.append({
        "game_date": record["game_date"],
        "game_id": record["game_id"],
        "gamePk": record["gamePk"],
        "gameNumber": record["gameNumber"],
        "away_team": record["away_team"],
        "home_team": record["home_team"],
        "final_away_score": (
            record["final_away_score"]
        ),
        "final_home_score": (
            record["final_home_score"]
        ),
        "game_status": "final",
        "status_source": (
            "existing_final_score_file"
        ),
        "status_available": "True",
        "status_notes": (
            "valid existing final preserved "
            "before DRatings rebuild"
        ),
    })


def _preserve_existing_final_row(
    *,
    row,
    date,
    path,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
):
    record, state = (
        _prepare_existing_final_record(
            row,
            date,
        )
    )

    if state != "ready":
        return state

    action = add_final_record(
        record=record,
        source_file=f"existing:{path.name}",
        final_records_by_date=final_records_by_date,
        seen_by_game_id=seen_by_game_id,
        seen_by_fallback_key=seen_by_fallback_key,
        key_audit_rows=key_audit_rows,
    )

    _append_existing_final_status(
        status_audit_rows,
        record,
    )

    if action in {
        "duplicate_collapsed",
        "blank_game_id_duplicate_collapsed",
    }:
        return "duplicate"

    return "preserved"


def preserve_existing_final_score_records(
    *,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
):
    counts = {
        "files_seen": 0,
        "rows_seen": 0,
        "rows_preserved": 0,
        "skipped_missing_ids": 0,
        "skipped_not_final": 0,
        "duplicates": 0,
    }

    counter_by_state = {
        "preserved": "rows_preserved",
        "duplicate": "duplicates",
        "missing_ids": "skipped_missing_ids",
        "not_final": "skipped_not_final",
    }

    for path in sorted(
        FINAL_DIR.glob(
            "*_final_scores_MLB.csv"
        )
    ):
        counts["files_seen"] += 1

        date = legacy_final_date_from_path(
            path
        )

        if not date:
            continue

        with open(
            path,
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                counts["rows_seen"] += 1

                state = _preserve_existing_final_row(
                    row=row,
                    date=date,
                    path=path,
                    final_records_by_date=final_records_by_date,
                    seen_by_game_id=seen_by_game_id,
                    seen_by_fallback_key=seen_by_fallback_key,
                    status_audit_rows=status_audit_rows,
                    key_audit_rows=key_audit_rows,
                )

                counts[
                    counter_by_state[state]
                ] += 1

    log(
        "EXISTING FINAL PRESERVATION | "
        f"files_seen={counts['files_seen']} | "
        f"rows_seen={counts['rows_seen']} | "
        f"rows_preserved={counts['rows_preserved']} | "
        f"duplicates={counts['duplicates']} | "
        f"skipped_missing_ids="
        f"{counts['skipped_missing_ids']} | "
        f"skipped_not_final="
        f"{counts['skipped_not_final']}"
    )

    return counts

def _validated_mlb_api_target(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            f"Invalid MLB API URL: {url}"
        ) from exc

    trusted = (
        parsed.scheme.lower() == "https"
        and parsed.hostname == MLB_API_HOST
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
    )

    if not trusted:
        raise ValueError(
            f"Refusing untrusted MLB API URL: {url}"
        )

    target = parsed.path or "/"

    if parsed.query:
        target += f"?{parsed.query}"

    return target


def _fetch_mlb_api_json(
    url: str,
    *,
    timeout: int,
    headers: dict | None = None,
) -> dict:
    target = _validated_mlb_api_target(url)

    connection = http.client.HTTPSConnection(
        MLB_API_HOST,
        443,
        timeout=timeout,
    )

    try:
        connection.request(
            "GET",
            target,
            headers=headers or {},
        )

        response = connection.getresponse()
        body = response.read()

    finally:
        connection.close()

    if response.status >= 400:
        raise RuntimeError(
            "MLB API HTTP error "
            f"{response.status} {response.reason}"
        )

    try:
        payload = json.loads(
            body.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "MLB API returned invalid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "MLB API returned non-object JSON"
        )

    return payload


def fetch_mlb_game_feed(game_pk, cache):
    game_pk = str(game_pk or "").strip()

    if not game_pk:
        return None

    if game_pk in cache:
        return cache[game_pk]

    url = (
        "https://statsapi.mlb.com/api/v1.1/game/"
        f"{game_pk}/feed/live"
    )

    try:
        payload = _fetch_mlb_api_json(
            url,
            timeout=MLB_API_TIMEOUT_SECONDS,
            headers={
                "User-Agent": MLB_API_USER_AGENT,
                "Accept": "application/json",
            },
        )

    except (
        ValueError,
        RuntimeError,
        http.client.HTTPException,
        OSError,
    ) as exc:
        log(
            "MLB API ERROR | "
            f"gamePk={game_pk} | "
            f"error={exc}"
        )
        cache[game_pk] = None
        return None

    cache[game_pk] = payload
    return payload

def extract_mlb_feed_status(feed):
    if not isinstance(feed, dict):
        return "unknown", "", ""

    game_data = feed.get("gameData") or {}
    status = game_data.get("status") or {}

    for field in (
        "abstractGameState",
        "detailedState",
        "codedGameState",
        "statusCode",
    ):
        raw = str(status.get(field, "") or "").strip()

        if not raw:
            continue

        normalized = normalize_status(raw)

        if normalized != "unknown":
            return normalized, raw, field

    return "unknown", "", ""


def extract_mlb_final_score(feed):
    status_norm, status_raw, status_field = extract_mlb_feed_status(feed)

    if status_norm != "final":
        return {
            "is_final": False,
            "game_status": status_norm,
            "raw_status": status_raw,
            "status_field": status_field,
        }

    try:
        live_data = feed.get("liveData") or {}
        linescore = live_data.get("linescore") or {}
        teams = linescore.get("teams") or {}

        away_score = int(
            (teams.get("away") or {}).get("runs")
        )
        home_score = int(
            (teams.get("home") or {}).get("runs")
        )
    except (TypeError, ValueError):
        return {
            "is_final": False,
            "game_status": "final",
            "raw_status": status_raw,
            "status_field": status_field,
            "score_missing": True,
        }

    game_data = feed.get("gameData") or {}
    team_data = game_data.get("teams") or {}

    api_away_team = str(
        ((team_data.get("away") or {}).get("name")) or ""
    ).strip()

    api_home_team = str(
        ((team_data.get("home") or {}).get("name")) or ""
    ).strip()

    return {
        "is_final": True,
        "game_status": "final",
        "raw_status": status_raw,
        "status_field": status_field,
        "away_score": away_score,
        "home_score": home_score,
        "api_away_team": api_away_team,
        "api_home_team": api_home_team,
    }


def _new_mlb_backfill_counts():
    return {
        "games_files_seen": 0,
        "games_rows_seen": 0,
        "skipped_already_present": 0,
        "skipped_missing_ids": 0,
        "api_checked": 0,
        "api_errors": 0,
        "api_not_final": 0,
        "api_score_missing": 0,
        "api_team_mismatch": 0,
        "added": 0,
        "duplicate_collapsed": 0,
    }


def _log_mlb_backfill_nonfinal(
    *,
    games_path,
    row_index,
    game_id,
    game_pk,
    away_team,
    home_team,
    result,
    counts,
):
    if result.get("score_missing"):
        counts["api_score_missing"] += 1

        log(
            "MLB FALLBACK FINAL SCORE MISSING | "
            f"games_file={games_path.name} | "
            f"row={row_index} | "
            f"game_id={game_id} | "
            f"gamePk={game_pk} | "
            f"away_team={away_team} | "
            f"home_team={home_team}"
        )
        return

    counts["api_not_final"] += 1

    log(
        "MLB FALLBACK NOT FINAL | "
        f"games_file={games_path.name} | "
        f"row={row_index} | "
        f"game_id={game_id} | "
        f"gamePk={game_pk} | "
        f"status={result.get('raw_status', '')}"
    )


def _mlb_backfill_team_mismatch(
    *,
    result,
    home_team,
    away_team,
):
    api_away_team = str(
        result.get("api_away_team", "") or ""
    ).strip()

    api_home_team = str(
        result.get("api_home_team", "") or ""
    ).strip()

    mismatch = (
        api_away_team
        and api_home_team
        and matchup_key(
            api_home_team,
            api_away_team,
        )
        != matchup_key(
            home_team,
            away_team,
        )
    )

    return (
        bool(mismatch),
        api_away_team,
        api_home_team,
    )


def _process_mlb_backfill_row(
    *,
    row,
    row_index,
    games_path,
    date,
    sportsbook_lookup,
    feed_cache,
    counts,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
):
    (
        game_id,
        game_pk,
        game_number,
        game_time,
    ) = identity_fields(row)

    home_team = str(
        row.get("home_team", "") or ""
    ).strip()

    away_team = str(
        row.get("away_team", "") or ""
    ).strip()

    if not game_id or not game_pk:
        counts["skipped_missing_ids"] += 1
        return

    if game_id in seen_by_game_id:
        counts["skipped_already_present"] += 1
        return

    counts["api_checked"] += 1

    feed = fetch_mlb_game_feed(
        game_pk,
        feed_cache,
    )

    if feed is None:
        counts["api_errors"] += 1
        return

    result = extract_mlb_final_score(feed)

    if not result.get("is_final"):
        _log_mlb_backfill_nonfinal(
            games_path=games_path,
            row_index=row_index,
            game_id=game_id,
            game_pk=game_pk,
            away_team=away_team,
            home_team=home_team,
            result=result,
            counts=counts,
        )
        return

    (
        team_mismatch,
        api_away_team,
        api_home_team,
    ) = _mlb_backfill_team_mismatch(
        result=result,
        home_team=home_team,
        away_team=away_team,
    )

    if team_mismatch:
        counts["api_team_mismatch"] += 1

        log(
            "MLB FALLBACK TEAM MISMATCH; SKIPPED | "
            f"games_file={games_path.name} | "
            f"row={row_index} | "
            f"game_id={game_id} | "
            f"gamePk={game_pk} | "
            f"local={away_team} @ {home_team} | "
            f"mlb={api_away_team} @ {api_home_team}"
        )
        return

    away_score = int(result["away_score"])
    home_score = int(result["home_score"])
    final_total = str(
        away_score + home_score
    )

    book = closest_time_book_match(
        sportsbook_lookup.get(
            matchup_key(
                home_team,
                away_team,
            ),
            [],
        ),
        game_time,
        correction_minutes=0,
        prefer_correction=False,
    )

    record = {
        "sport": "baseball",
        "league": "mlb",
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "game_date": date,
        "game_time": game_time,
        "home_team": home_team,
        "away_team": away_team,
        "final_away_score": str(away_score),
        "final_home_score": str(home_score),
        "final_total": final_total,
        "away_run_line": book.get("away_run_line"),
        "home_run_line": book.get("home_run_line"),
        "total": book.get("total"),
        "game_status": "final",
        "final_scores_generated_at": RUN_TS,
    }

    action = add_final_record(
        record=record,
        source_file=(
            "MLB_STATSAPI_"
            f"gamePk_{game_pk}"
        ),
        final_records_by_date=final_records_by_date,
        seen_by_game_id=seen_by_game_id,
        seen_by_fallback_key=seen_by_fallback_key,
        key_audit_rows=key_audit_rows,
    )

    if action in {
        "duplicate_collapsed",
        "blank_game_id_duplicate_collapsed",
    }:
        counts["duplicate_collapsed"] += 1
    else:
        counts["added"] += 1

    status_audit_rows.append({
        "game_date": date,
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "away_team": away_team,
        "home_team": home_team,
        "final_away_score": str(away_score),
        "final_home_score": str(home_score),
        "game_status": "final",
        "status_source": (
            "MLB StatsAPI gamePk fallback"
        ),
        "status_available": "True",
        "status_notes": (
            "DRatings/existing finals did not contain "
            "this game; official MLB final score added "
            f"using gamePk={game_pk}"
        ),
    })

    log(
        "MLB FALLBACK ADDED | "
        f"game_date={date} | "
        f"game_id={game_id} | "
        f"gamePk={game_pk} | "
        f"gameNumber={game_number} | "
        f"game_time={game_time} | "
        f"away_team={away_team} | "
        f"home_team={home_team} | "
        f"final={away_score}-{home_score}"
    )


def backfill_missing_finals_from_mlb(
    *,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
):
    feed_cache = {}
    sportsbook_cache = {}
    counts = _new_mlb_backfill_counts()

    for games_path in sorted(
        GAMES_DIR.glob("*_games.csv")
    ):
        date = games_date_from_path(
            games_path
        )

        if not date:
            continue

        counts["games_files_seen"] += 1

        if date not in sportsbook_cache:
            sportsbook_cache[date] = (
                load_sportsbook_lookup(date)
            )

        with open(
            games_path,
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)

            for row_index, row in enumerate(
                reader,
                start=2,
            ):
                counts["games_rows_seen"] += 1

                _process_mlb_backfill_row(
                    row=row,
                    row_index=row_index,
                    games_path=games_path,
                    date=date,
                    sportsbook_lookup=(
                        sportsbook_cache[date]
                    ),
                    feed_cache=feed_cache,
                    counts=counts,
                    final_records_by_date=final_records_by_date,
                    seen_by_game_id=seen_by_game_id,
                    seen_by_fallback_key=seen_by_fallback_key,
                    status_audit_rows=status_audit_rows,
                    key_audit_rows=key_audit_rows,
                )

    log(
        "MLB FALLBACK SUMMARY | "
        f"games_files_seen={counts['games_files_seen']} | "
        f"games_rows_seen={counts['games_rows_seen']} | "
        f"skipped_already_present="
        f"{counts['skipped_already_present']} | "
        f"skipped_missing_ids="
        f"{counts['skipped_missing_ids']} | "
        f"api_checked={counts['api_checked']} | "
        f"api_errors={counts['api_errors']} | "
        f"api_not_final={counts['api_not_final']} | "
        f"api_score_missing="
        f"{counts['api_score_missing']} | "
        f"api_team_mismatch="
        f"{counts['api_team_mismatch']} | "
        f"added={counts['added']} | "
        f"duplicate_collapsed="
        f"{counts['duplicate_collapsed']}"
    )

    return counts

def _new_process_file_counts():
    return {
        "parse_errors": 0,
        "skipped_summary": 0,
        "skipped_duplicate": 0,
        "skipped_not_completed": 0,
        "completed_rows_seen": 0,
        "accepted_rows": 0,
        "unresolved_rows": 0,
    }


def _validate_final_source_row(
    row,
    row_index,
    file_path,
    parse_error_rows,
    status_audit_rows,
):
    if not isinstance(row, list):
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="validate_row_structure",
                error=(
                    "expected row list, found "
                    f"{type(row).__name__}"
                ),
                row=row,
            )
        )
        return "parse_error", None

    if not row:
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="validate_row_structure",
                error="empty row",
                row=row,
            )
        )
        return "parse_error", None

    if is_summary_row(row):
        return "summary", None

    if len(row) < 2:
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="validate_row_structure",
                error=(
                    "expected at least 2 fields, "
                    f"found {len(row)}"
                ),
                row=row,
            )
        )
        return "parse_error", None

    status = infer_game_status(row)

    if is_completed_game(row):
        return "ready", status

    (
        status_norm,
        _raw_status,
        status_source,
        status_available,
    ) = status

    status_audit_rows.append({
        "game_date": "",
        "game_id": "",
        "gamePk": "",
        "gameNumber": "",
        "away_team": "",
        "home_team": "",
        "final_away_score": "",
        "final_home_score": "",
        "game_status": status_norm,
        "status_source": status_source,
        "status_available": str(status_available),
        "status_notes": (
            "non-final row not written "
            "to final-score output"
        ),
    })

    return "not_completed", status


def _parse_final_source_fields(
    row,
    row_index,
    file_path,
    parse_error_rows,
):
    try:
        _dt, game_date, game_time = (
            parse_datetime(row[0])
        )
    except Exception as exc:
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="parse_datetime",
                error=exc,
                row=row,
            )
        )
        return None

    try:
        team_value = row[1]

        if not isinstance(team_value, str):
            raise TypeError(
                "expected team field to be str, found "
                f"{type(team_value).__name__}"
            )

        teams = team_value.split("\n")

        if len(teams) < 2:
            raise ValueError(
                "expected at least two team names"
            )

        away_team = clean_team(teams[0])
        home_team = clean_team(teams[1])

        if not away_team or not home_team:
            raise ValueError(
                "away or home team is blank"
            )

    except Exception as exc:
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="parse_teams",
                error=exc,
                row=row,
            )
        )
        return None

    try:
        score_value = row[5]

        if not isinstance(score_value, str):
            raise TypeError(
                "expected score field to be str, found "
                f"{type(score_value).__name__}"
            )

        scores = score_value.split("\n")

        if len(scores) < 2:
            raise ValueError(
                "expected away/home final scores, found "
                f"{len(scores)} score field(s)"
            )

        away_score = int(scores[0].strip())
        home_score = int(scores[1].strip())

    except Exception as exc:
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="parse_scores",
                error=exc,
                row=row,
            )
        )
        return None

    return {
        "game_date": game_date,
        "raw_game_time": game_time,
        "away_team": away_team,
        "home_team": home_team,
        "away_score": away_score,
        "home_score": home_score,
        "final_total": str(
            away_score + home_score
        ),
    }


def _load_final_source_lookups(
    game_date,
    caches,
):
    if game_date not in caches["games"]:
        caches["games"][game_date] = (
            load_games_lookup(game_date)
        )
        caches["game_ids"][game_date] = (
            load_games_by_game_id(game_date)
        )
        caches["game_pks"][game_date] = (
            load_games_by_gamepk(game_date)
        )

    if game_date not in caches["predictions"]:
        caches["predictions"][game_date] = (
            load_predictions_lookup(game_date)
        )

    if game_date not in caches["sportsbook"]:
        caches["sportsbook"][game_date] = (
            load_sportsbook_lookup(game_date)
        )

    return {
        "games": caches["games"][game_date],
        "game_ids": caches["game_ids"][game_date],
        "game_pks": caches["game_pks"][game_date],
        "predictions": caches["predictions"][game_date],
        "sportsbook": caches["sportsbook"][game_date],
    }


def _append_unresolved_source_row(
    *,
    file_path,
    row_index,
    row,
    parsed,
    resolution,
    status,
    unresolved_completed_rows,
    status_audit_rows,
):
    game_id = _identity_value(
        resolution,
        "game_id",
    )
    game_pk = _identity_value(
        resolution,
        "gamePk",
    )
    game_number = _identity_value(
        resolution,
        "gameNumber",
    )

    unresolved_completed_rows.append(
        make_unresolved_completed_row(
            source_file=file_path.name,
            row_index=row_index,
            game_date=parsed["game_date"],
            game_time=parsed["raw_game_time"],
            away_team=parsed["away_team"],
            home_team=parsed["home_team"],
            final_away_score=str(
                parsed["away_score"]
            ),
            final_home_score=str(
                parsed["home_score"]
            ),
            game_id=game_id,
            game_pk=game_pk,
            game_number=game_number,
            games_candidate_count=resolution.get(
                "games_candidate_count",
                0,
            ),
            prediction_candidate_count=resolution.get(
                "prediction_candidate_count",
                0,
            ),
            resolution_reason=resolution.get(
                "reason",
                "unresolved",
            ),
            raw_row=raw_row_text(row),
        )
    )

    (
        status_norm,
        _raw_status,
        status_source,
        status_available,
    ) = status

    status_audit_rows.append({
        "game_date": parsed["game_date"],
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "away_team": parsed["away_team"],
        "home_team": parsed["home_team"],
        "final_away_score": str(
            parsed["away_score"]
        ),
        "final_home_score": str(
            parsed["home_score"]
        ),
        "game_status": status_norm,
        "status_source": status_source,
        "status_available": str(
            status_available
        ),
        "status_notes": (
            "completed game unresolved; excluded "
            "from final-score output and written to "
            "unresolved_completed_games.csv"
        ),
    })


def _accept_final_source_row(
    *,
    file_path,
    parsed,
    resolution,
    lookups,
    status,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
):
    game_id = _identity_value(
        resolution,
        "game_id",
    )
    game_pk = _identity_value(
        resolution,
        "gamePk",
    )
    game_number = _identity_value(
        resolution,
        "gameNumber",
    )

    scheduled_game_time = str(
        resolution.get(
            "scheduled_game_time",
            "",
        )
        or parsed["raw_game_time"]
    ).strip()

    book = closest_time_book_match(
        lookups["sportsbook"].get(
            matchup_key(
                parsed["home_team"],
                parsed["away_team"],
            ),
            [],
        ),
        parsed["raw_game_time"],
        correction_minutes=0,
        prefer_correction=False,
    )

    (
        status_norm,
        _raw_status,
        status_source,
        status_available,
    ) = status

    record = {
        "sport": "baseball",
        "league": "mlb",
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "game_date": parsed["game_date"],
        "game_time": scheduled_game_time,
        "home_team": parsed["home_team"],
        "away_team": parsed["away_team"],
        "final_away_score": str(
            parsed["away_score"]
        ),
        "final_home_score": str(
            parsed["home_score"]
        ),
        "final_total": parsed["final_total"],
        "away_run_line": book.get(
            "away_run_line"
        ),
        "home_run_line": book.get(
            "home_run_line"
        ),
        "total": book.get("total"),
        "game_status": status_norm,
        "final_scores_generated_at": RUN_TS,
    }

    action = add_final_record(
        record=record,
        source_file=file_path.name,
        final_records_by_date=final_records_by_date,
        seen_by_game_id=seen_by_game_id,
        seen_by_fallback_key=seen_by_fallback_key,
        key_audit_rows=key_audit_rows,
    )

    status_note = (
        "explicit source status available"
        if status_available
        else (
            "status inferred as final from "
            "completed DRatings row shape"
        )
    )

    status_audit_rows.append({
        "game_date": parsed["game_date"],
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "away_team": parsed["away_team"],
        "home_team": parsed["home_team"],
        "final_away_score": str(
            parsed["away_score"]
        ),
        "final_home_score": str(
            parsed["home_score"]
        ),
        "game_status": status_norm,
        "status_source": status_source,
        "status_available": str(
            status_available
        ),
        "status_notes": (
            "resolved_ids="
            f"{resolution.get('resolution_source', '')}; "
            f"{status_note}"
        ),
    })

    return action in {
        "duplicate_collapsed",
        "blank_game_id_duplicate_collapsed",
    }


def _process_final_source_row(
    *,
    row,
    row_index,
    file_path,
    caches,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
    parse_error_rows,
    unresolved_completed_rows,
):
    state, status = _validate_final_source_row(
        row,
        row_index,
        file_path,
        parse_error_rows,
        status_audit_rows,
    )

    if state != "ready":
        return state

    parsed = _parse_final_source_fields(
        row,
        row_index,
        file_path,
        parse_error_rows,
    )

    if parsed is None:
        return "parse_error"

    try:
        lookups = _load_final_source_lookups(
            parsed["game_date"],
            caches,
        )

        resolution = resolve_completed_game_ids(
            game_time=parsed["raw_game_time"],
            home_team=parsed["home_team"],
            away_team=parsed["away_team"],
            games_lookup=lookups["games"],
            games_by_game_id=lookups["game_ids"],
            games_by_gamepk=lookups["game_pks"],
            predictions_lookup=lookups[
                "predictions"
            ],
        )

        game_id = _identity_value(
            resolution,
            "game_id",
        )
        game_pk = _identity_value(
            resolution,
            "gamePk",
        )

        if (
            not resolution.get("resolved")
            or not game_id
            or not game_pk
        ):
            _append_unresolved_source_row(
                file_path=file_path,
                row_index=row_index,
                row=row,
                parsed=parsed,
                resolution=resolution,
                status=status,
                unresolved_completed_rows=(
                    unresolved_completed_rows
                ),
                status_audit_rows=(
                    status_audit_rows
                ),
            )
            return "unresolved"

        duplicate = _accept_final_source_row(
            file_path=file_path,
            parsed=parsed,
            resolution=resolution,
            lookups=lookups,
            status=status,
            final_records_by_date=(
                final_records_by_date
            ),
            seen_by_game_id=seen_by_game_id,
            seen_by_fallback_key=(
                seen_by_fallback_key
            ),
            status_audit_rows=status_audit_rows,
            key_audit_rows=key_audit_rows,
        )

        return (
            "duplicate"
            if duplicate
            else "accepted"
        )

    except FinalScoreConflictError:
        raise

    except Exception as exc:
        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="build_final_record",
                error=exc,
                row=row,
            )
        )
        return "parse_error"


def _log_process_file_counts(
    counts,
    final_records_by_date,
):
    log(
        f"  completed_rows_seen="
        f"{counts['completed_rows_seen']}, "
        f"accepted_rows="
        f"{counts['accepted_rows']}, "
        f"unresolved_completed_rows="
        f"{counts['unresolved_rows']}, "
        f"parse_errors="
        f"{counts['parse_errors']}, "
        f"skipped_summary="
        f"{counts['skipped_summary']}, "
        f"skipped_duplicate="
        f"{counts['skipped_duplicate']}, "
        f"skipped_not_completed="
        f"{counts['skipped_not_completed']}, "
        f"final_score_dates_accumulated="
        f"{len(final_records_by_date)}"
    )


def process_file(
    file_path,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
    parse_error_rows,
    unresolved_completed_rows,
):
    safe_file_path = Path(
        file_path
    ).resolve()

    allowed_root = RAW_DIR.resolve()

    if not safe_file_path.is_relative_to(
        allowed_root
    ):
        fail(
            "Refusing DRatings input outside "
            f"trusted directory: {file_path}"
        )

    if not safe_file_path.name.endswith(
        "_mlb_raw.json"
    ):
        fail(
            "Refusing unexpected DRatings "
            f"input file: {file_path}"
        )

    file_path = safe_file_path

    log(f"Processing {file_path.name}")

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    counts = _new_process_file_counts()

    if not isinstance(data, list):
        counts["parse_errors"] += 1

        parse_error_rows.append(
            make_parse_error_row(
                source_file=file_path.name,
                row_index="",
                stage="validate_json_structure",
                error=(
                    "expected top-level JSON list, found "
                    f"{type(data).__name__}"
                ),
                row=data,
            )
        )

        _log_process_file_counts(
            counts,
            final_records_by_date,
        )
        return

    caches = {
        "games": {},
        "game_ids": {},
        "game_pks": {},
        "predictions": {},
        "sportsbook": {},
    }

    counter_by_state = {
        "parse_error": "parse_errors",
        "summary": "skipped_summary",
        "duplicate": "skipped_duplicate",
        "not_completed": "skipped_not_completed",
        "accepted": "accepted_rows",
        "unresolved": "unresolved_rows",
    }

    for row_index, row in enumerate(
        data,
        start=1,
    ):
        state = _process_final_source_row(
            row=row,
            row_index=row_index,
            file_path=file_path,
            caches=caches,
            final_records_by_date=(
                final_records_by_date
            ),
            seen_by_game_id=seen_by_game_id,
            seen_by_fallback_key=(
                seen_by_fallback_key
            ),
            status_audit_rows=status_audit_rows,
            key_audit_rows=key_audit_rows,
            parse_error_rows=parse_error_rows,
            unresolved_completed_rows=(
                unresolved_completed_rows
            ),
        )

        if state in {
            "accepted",
            "duplicate",
            "unresolved",
        }:
            counts["completed_rows_seen"] += 1

        counts[counter_by_state[state]] += 1

    _log_process_file_counts(
        counts,
        final_records_by_date,
    )

def _prepare_legacy_final_record(row, date):
    record = {
        col: str(
            row.get(col, "") or ""
        ).strip()
        for col in FINAL_HEADER
    }

    changed = False

    record["sport"] = (
        record["sport"]
        or "baseball"
    )
    record["league"] = (
        record["league"]
        or "mlb"
    )
    record["game_date"] = (
        record["game_date"]
        or date
    )

    has_final_score = (
        legacy_row_has_final_score(record)
    )

    if (
        not record["game_status"]
        and has_final_score
    ):
        record["game_status"] = "final"
        changed = True

    if (
        not record["final_total"]
        and has_final_score
    ):
        record["final_total"] = str(
            int(record["final_away_score"])
            + int(record["final_home_score"])
        )
        changed = True

    if not record["final_scores_generated_at"]:
        record[
            "final_scores_generated_at"
        ] = RUN_TS
        changed = True

    return record, changed


def _resolve_legacy_completed_record(
    *,
    record,
    row,
    row_index,
    path,
    games_lookup,
    games_by_game_id,
    games_by_gamepk,
    predictions_lookup,
    unresolved_completed_rows,
):
    completed = (
        record["game_status"].strip().lower()
        == "final"
        and legacy_row_has_final_score(record)
    )

    if not completed:
        return True, False, 0, 0

    before_ids = (
        record["game_id"],
        record["gamePk"],
        record["gameNumber"],
        record["game_time"],
    )

    resolution = resolve_completed_game_ids(
        game_time=record["game_time"],
        home_team=record["home_team"],
        away_team=record["away_team"],
        current_game_id=record["game_id"],
        current_game_pk=record["gamePk"],
        current_game_number=record["gameNumber"],
        games_lookup=games_lookup,
        games_by_game_id=games_by_game_id,
        games_by_gamepk=games_by_gamepk,
        predictions_lookup=predictions_lookup,
    )

    record["game_id"] = str(
        resolution.get("game_id", "")
        or ""
    ).strip()

    record["gamePk"] = str(
        resolution.get("gamePk", "")
        or ""
    ).strip()

    record["gameNumber"] = str(
        resolution.get("gameNumber", "")
        or ""
    ).strip()

    scheduled_time = str(
        resolution.get(
            "scheduled_game_time",
            "",
        )
        or ""
    ).strip()

    if scheduled_time:
        record["game_time"] = scheduled_time

    after_ids = (
        record["game_id"],
        record["gamePk"],
        record["gameNumber"],
        record["game_time"],
    )

    identity_changed = (
        after_ids != before_ids
    )

    unresolved = (
        not resolution.get("resolved")
        or not record["game_id"]
        or not record["gamePk"]
    )

    if not unresolved:
        return (
            True,
            identity_changed,
            int(identity_changed),
            0,
        )

    unresolved_completed_rows.append(
        make_unresolved_completed_row(
            source_file=path.name,
            row_index=row_index,
            game_date=record["game_date"],
            game_time=record["game_time"],
            away_team=record["away_team"],
            home_team=record["home_team"],
            final_away_score=record[
                "final_away_score"
            ],
            final_home_score=record[
                "final_home_score"
            ],
            game_id=record["game_id"],
            game_pk=record["gamePk"],
            game_number=record[
                "gameNumber"
            ],
            games_candidate_count=(
                resolution.get(
                    "games_candidate_count",
                    0,
                )
            ),
            prediction_candidate_count=(
                resolution.get(
                    "prediction_candidate_count",
                    0,
                )
            ),
            resolution_reason=(
                resolution.get(
                    "reason",
                    (
                        "legacy completed game "
                        "could not resolve both IDs"
                    ),
                )
            ),
            raw_row=raw_row_text(row),
        )
    )

    return (
        False,
        True,
        int(identity_changed),
        1,
    )


def _migrate_legacy_final_score_file(
    path,
    files_written,
    unresolved_completed_rows,
):
    safe_path = Path(path).resolve()
    allowed_root = FINAL_DIR.resolve()

    if not safe_path.is_relative_to(
        allowed_root
    ):
        fail(
            "Refusing legacy final-score file "
            f"outside trusted directory: {path}"
        )

    if not safe_path.name.endswith(
        "_final_scores_MLB.csv"
    ):
        fail(
            "Refusing unexpected legacy "
            f"final-score file: {path}"
        )

    path = safe_path

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(
            reader.fieldnames or []
        )
        rows = list(reader)

    if not fieldnames:
        fail(
            "Legacy final-score file has "
            f"no header: {path}"
        )

    date = legacy_final_date_from_path(path)

    if not date:
        fail(
            "Could not derive date from legacy "
            f"final-score path: {path}"
        )

    games_lookup = load_games_lookup(date)
    games_by_game_id = (
        load_games_by_game_id(date)
    )
    games_by_gamepk = (
        load_games_by_gamepk(date)
    )
    predictions_lookup = (
        load_predictions_lookup(date)
    )

    missing_header_columns = [
        col
        for col in FINAL_HEADER
        if col not in fieldnames
    ]

    changed = bool(
        missing_header_columns
    )
    resolved_rows = 0
    unresolved_rows = 0
    output_rows = []

    for row_index, row in enumerate(
        rows,
        start=2,
    ):
        record, record_changed = (
            _prepare_legacy_final_record(
                row,
                date,
            )
        )

        changed = (
            changed or record_changed
        )

        (
            retain,
            resolution_changed,
            resolved_delta,
            unresolved_delta,
        ) = _resolve_legacy_completed_record(
            record=record,
            row=row,
            row_index=row_index,
            path=path,
            games_lookup=games_lookup,
            games_by_game_id=games_by_game_id,
            games_by_gamepk=games_by_gamepk,
            predictions_lookup=predictions_lookup,
            unresolved_completed_rows=unresolved_completed_rows,
        )

        changed = (
            changed or resolution_changed
        )
        resolved_rows += resolved_delta
        unresolved_rows += unresolved_delta

        if retain:
            output_rows.append([
                record.get(col, "")
                for col in FINAL_HEADER
            ])

    if changed:
        write_csv(
            path,
            FINAL_HEADER,
            output_rows,
            files_written,
            "historical final-score ID/schema backfill",
        )

        log(
            "MIGRATED HISTORICAL FINAL-SCORE FILE | "
            f"file={path.name} | "
            f"rows={len(output_rows)} | "
            f"missing_header_columns="
            f"{missing_header_columns}"
        )

    return {
        "migrated_files": int(changed),
        "migrated_rows": (
            len(output_rows)
            if changed
            else 0
        ),
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
    }


def migrate_legacy_final_score_files(
    files_written,
    unresolved_completed_rows,
):
    totals = {
        "migrated_files": 0,
        "migrated_rows": 0,
        "resolved_rows": 0,
        "unresolved_rows": 0,
    }

    for path in sorted(
        FINAL_DIR.glob(
            "*_final_scores_MLB.csv"
        )
    ):
        result = (
            _migrate_legacy_final_score_file(
                path,
                files_written,
                unresolved_completed_rows,
            )
        )

        for key in totals:
            totals[key] += result[key]

    log(
        "Historical final-score files updated: "
        f"{totals['migrated_files']}"
    )
    log(
        "Historical final-score rows retained: "
        f"{totals['migrated_rows']}"
    )
    log(
        "Historical completed rows "
        "resolved/backfilled: "
        f"{totals['resolved_rows']}"
    )
    log(
        "Historical completed rows moved "
        "to unresolved audit: "
        f"{totals['unresolved_rows']}"
    )

    return totals

def verify_final_score_outputs_have_gamepk():
    bad_rows = []

    for path in sorted(FINAL_DIR.glob("*_final_scores_MLB.csv")):
        with open(
            path,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)

            for row_index, row in enumerate(reader, start=2):
                status = str(
                    row.get("game_status", "") or ""
                ).strip().lower()

                if (
                    status != "final"
                    or not legacy_row_has_final_score(row)
                ):
                    continue

                game_pk = str(
                    row.get("gamePk", "") or ""
                ).strip()

                if game_pk:
                    continue

                bad_rows.append({
                    "file": path.name,
                    "row": row_index,
                    "game_id": str(
                        row.get("game_id", "") or ""
                    ).strip(),
                    "game_date": str(
                        row.get("game_date", "") or ""
                    ).strip(),
                    "game_time": str(
                        row.get("game_time", "") or ""
                    ).strip(),
                    "away_team": str(
                        row.get("away_team", "") or ""
                    ).strip(),
                    "home_team": str(
                        row.get("home_team", "") or ""
                    ).strip(),
                })

    if bad_rows:
        sample = bad_rows[:10]

        fail(
            "Completed final-score rows with blank gamePk "
            "remain after backfill; "
            f"bad_rows={len(bad_rows)} "
            f"sample={sample}"
        )

    log(
        "VERIFY: completed final-score output rows "
        "with blank gamePk: 0"
    )


def _multi_game_matchup_groups(games_rows):
    groups = {}
    for row in games_rows:
        key = matchup_key(
            row.get("home_team", ""),
            row.get("away_team", ""),
        )
        groups.setdefault(key, []).append(row)

    return {
        key: rows
        for key, rows in groups.items()
        if len(rows) > 1
    }


def _candidate_gamepks(
    candidate_rows,
    date,
    key,
):
    gamepks = [
        str(row.get("gamePk", "") or "").strip()
        for row in candidate_rows
        if str(row.get("gamePk", "") or "").strip()
    ]

    if len(gamepks) != len(set(gamepks)):
        fail(
            "Duplicate gamePk values exist inside a "
            "same-date/same-team games group; "
            f"date={date} matchup={key} "
            f"gamePks={gamepks}"
        )

    return gamepks


def _relevant_multi_game_finals(final_rows, key):
    return [
        row
        for row in final_rows
        if matchup_key(
            row.get("home_team", ""),
            row.get("away_team", ""),
        ) == key
        and str(
            row.get("game_status", "") or ""
        ).strip().lower() == "final"
        and legacy_row_has_final_score(row)
    ]


def _doubleheader_bad_row(
    date,
    key,
    identity,
    reason,
):
    game_id, game_pk, game_number, game_time = identity
    return {
        "date": date,
        "matchup": key,
        "game_id": game_id,
        "gamePk": game_pk,
        "gameNumber": game_number,
        "game_time": game_time,
        "reason": reason,
    }


def _validate_final_identity_uniqueness(
    identity,
    seen_game_ids,
    seen_gamepks,
):
    game_id, game_pk, _game_number, _game_time = identity

    if not game_id or not game_pk:
        return "blank game_id/gamePk in multi-game matchup"

    if game_id in seen_game_ids:
        return (
            "same game_id used by multiple finals "
            "in same-team multi-game matchup"
        )

    if game_pk in seen_gamepks:
        return (
            "same gamePk used by multiple finals "
            "in same-team multi-game matchup"
        )

    seen_game_ids.add(game_id)
    seen_gamepks.add(game_pk)
    return None


def _official_candidate_for_gamepk(
    candidate_rows,
    game_pk,
):
    matches = [
        candidate
        for candidate in candidate_rows
        if str(
            candidate.get("gamePk", "") or ""
        ).strip() == game_pk
    ]

    if len(matches) != 1:
        return None

    return matches[0]


def _validate_final_against_games(
    identity,
    candidate_rows,
):
    game_id, game_pk, game_number, game_time = identity
    official = _official_candidate_for_gamepk(
        candidate_rows,
        game_pk,
    )

    if official is None:
        return (
            "final gamePk did not map to exactly one "
            "games candidate"
        )

    official_game_id = str(
        official.get("game_id", "") or ""
    ).strip()
    official_game_number = str(
        official.get("gameNumber", "") or ""
    ).strip()
    official_game_time = str(
        official.get("game_time", "") or ""
    ).strip()

    if official_game_id and official_game_id != game_id:
        return (
            "final game_id disagreed with the games "
            "row selected by gamePk"
        )

    if (
        game_number
        and official_game_number
        and game_number != official_game_number
    ):
        return (
            "final gameNumber disagreed with games "
            "row selected by gamePk"
        )

    diff = time_difference_minutes(
        game_time,
        official_game_time,
    )
    if (
        diff is None
        or diff > DOUBLEHEADER_TIME_TOLERANCE_MINUTES
    ):
        return (
            "final scheduled time did not agree with "
            "games row selected by gamePk/gameNumber"
        )

    return None


def _verify_multi_game_group(
    date,
    key,
    candidate_rows,
    final_rows,
    bad_rows,
):
    _candidate_gamepks(
        candidate_rows,
        date,
        key,
    )
    relevant_finals = _relevant_multi_game_finals(
        final_rows,
        key,
    )
    seen_game_ids = set()
    seen_gamepks = set()
    verified = 0

    for final_row in relevant_finals:
        identity = identity_fields(final_row)
        reason = _validate_final_identity_uniqueness(
            identity,
            seen_game_ids,
            seen_gamepks,
        )

        if reason is None:
            reason = _validate_final_against_games(
                identity,
                candidate_rows,
            )

        if reason is not None:
            bad_rows.append(
                _doubleheader_bad_row(
                    date,
                    key,
                    identity,
                    reason,
                )
            )
            continue

        verified += 1

    return verified


def verify_doubleheader_identity_integrity():
    doubleheader_matchups = 0
    verified_final_rows = 0
    bad_rows = []

    for games_path in sorted(
        GAMES_DIR.glob("*_games.csv")
    ):
        date = games_date_from_path(games_path)
        if not date:
            continue

        with open(
            games_path,
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            games_rows = list(csv.DictReader(handle))

        multi_groups = _multi_game_matchup_groups(
            games_rows
        )
        if not multi_groups:
            continue

        final_path = (
            FINAL_DIR
            / f"{date}_final_scores_MLB.csv"
        )
        if not final_path.exists():
            continue

        with open(
            final_path,
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            final_rows = list(csv.DictReader(handle))

        for key, candidate_rows in multi_groups.items():
            doubleheader_matchups += 1
            verified_final_rows += _verify_multi_game_group(
                date,
                key,
                candidate_rows,
                final_rows,
                bad_rows,
            )

    if bad_rows:
        fail(
            "Doubleheader/multi-game identity verification failed; "
            f"bad_rows={len(bad_rows)} "
            f"sample={bad_rows[:10]}"
        )

    log(
        "VERIFY: doubleheader/multi-game identity integrity passed | "
        f"matchups={doubleheader_matchups} | "
        f"final_rows_verified={verified_final_rows}"
    )



def _final_score_audit_headers():
    status_header = [
        "game_date",
        "game_id",
        "gamePk",
        "gameNumber",
        "away_team",
        "home_team",
        "final_away_score",
        "final_home_score",
        "game_status",
        "status_source",
        "status_available",
        "status_notes",
    ]
    key_header = [
        "game_date",
        "game_id",
        "gamePk",
        "gameNumber",
        "away_team",
        "home_team",
        "duplicate_count",
        "status",
        "notes",
    ]
    unresolved_header = [
        "source_file",
        "row_index",
        "game_date",
        "game_time",
        "away_team",
        "home_team",
        "final_away_score",
        "final_home_score",
        "game_id",
        "gamePk",
        "gameNumber",
        "games_candidate_count",
        "prediction_candidate_count",
        "resolution_reason",
        "raw_row",
    ]
    return status_header, key_header, unresolved_header


def _load_final_score_raw_records(
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    status_audit_rows,
    key_audit_rows,
    parse_error_rows,
    unresolved_completed_rows,
):
    raw_files = sorted(
        RAW_DIR.glob("*_mlb_raw.json")
    )
    if not raw_files:
        fail(
            f"No DRatings raw files found in {RAW_DIR}"
        )

    log(f"Raw files found: {len(raw_files)}")
    log(
        "Historical final-score build timestamp: "
        f"{RUN_TS}"
    )

    existing_summary = preserve_existing_final_score_records(
        final_records_by_date=final_records_by_date,
        seen_by_game_id=seen_by_game_id,
        seen_by_fallback_key=seen_by_fallback_key,
        status_audit_rows=status_audit_rows,
        key_audit_rows=key_audit_rows,
    )

    for file in raw_files:
        process_file(
            file_path=file,
            final_records_by_date=final_records_by_date,
            seen_by_game_id=seen_by_game_id,
            seen_by_fallback_key=seen_by_fallback_key,
            status_audit_rows=status_audit_rows,
            key_audit_rows=key_audit_rows,
            parse_error_rows=parse_error_rows,
            unresolved_completed_rows=unresolved_completed_rows,
        )

    return raw_files, existing_summary


def _abort_on_final_score_parse_errors(
    raw_files,
    parse_error_rows,
    unresolved_completed_rows,
):
    total_parse_errors = len(parse_error_rows)
    if not total_parse_errors:
        return

    log("--- SUMMARY ---")
    log(
        "Raw files processed before failure: "
        f"{len(raw_files)}"
    )
    log(
        "Parse errors encountered: "
        f"{total_parse_errors}"
    )
    log_review_rows(
        parse_error_rows,
        unresolved_completed_rows,
    )
    fail(
        "Final-score build aborted because "
        f"parse_errors={total_parse_errors}. "
        "Final-score outputs were not written."
    )


def _resolved_final_missing_ids(record):
    status = str(
        record.get("game_status", "") or ""
    ).strip().lower()
    if status != "final":
        return False

    game_id = str(
        record.get("game_id", "") or ""
    ).strip()
    game_pk = str(
        record.get("gamePk", "") or ""
    ).strip()
    return not game_id or not game_pk


def _write_final_score_records(
    final_records_by_date,
    files_written,
):
    for date in sorted(final_records_by_date):
        records = final_records_by_date[date]
        bad_resolved = [
            record
            for record in records
            if _resolved_final_missing_ids(record)
        ]
        if bad_resolved:
            fail(
                "Resolved completed rows cannot be written "
                "with blank game_id/gamePk; "
                f"date={date} "
                f"bad_rows={len(bad_resolved)}"
            )

        out = (
            FINAL_DIR
            / f"{date}_final_scores_MLB.csv"
        )
        rows = [
            [
                record.get(col, "")
                for col in FINAL_HEADER
            ]
            for record in records
        ]
        write_csv(
            out,
            FINAL_HEADER,
            rows,
            files_written,
            "final scores",
        )


def _write_final_score_audits(
    status_header,
    key_header,
    unresolved_header,
    status_audit_rows,
    key_audit_rows,
    unresolved_completed_rows,
):
    write_audit_csv(
        STATUS_AUDIT_FILE,
        status_header,
        status_audit_rows,
        "final-score status audit",
    )
    write_audit_csv(
        KEY_AUDIT_FILE,
        key_header,
        key_audit_rows,
        "final-score key audit",
    )
    write_audit_csv(
        UNRESOLVED_AUDIT_FILE,
        unresolved_header,
        unresolved_completed_rows,
        "unresolved completed-game audit",
    )


def _unknown_final_status_count(status_audit_rows):
    return sum(
        1
        for row in status_audit_rows
        if str(
            row.get("game_status", "")
        ).strip().lower() == "unknown"
    )


def _log_final_score_review_status(
    total_parse_errors,
    unresolved_count,
):
    if total_parse_errors:
        fail(
            "Final-score build cannot report success because "
            f"parse_errors={total_parse_errors}"
        )

    log(
        "Parse-error review: "
        "no parse errors encountered."
    )

    if unresolved_count:
        log(
            "WARNING: Genuinely unresolved completed games "
            "were excluded from final-score outputs and "
            f"written to {UNRESOLVED_AUDIT_FILE}."
        )
        return

    log(
        "Unresolved completed-game review: none."
    )


def _log_final_score_summary(
    raw_files,
    files_written,
    final_records_by_date,
    seen_by_game_id,
    existing_summary,
    mlb_summary,
    legacy_summary,
    unresolved_count,
    total_parse_errors,
    unknown_status_count,
):
    log("--- SUMMARY ---")
    summary_lines = [
        f"Raw files processed: {len(raw_files)}",
        f"Files written: {len(files_written)}",
        (
            "Final-score dates written once: "
            f"{len(final_records_by_date)}"
        ),
        (
            "Final-score game_id primary-key rows: "
            f"{len(seen_by_game_id)}"
        ),
        (
            "Existing valid final-score rows preserved: "
            f"{existing_summary['rows_preserved']}"
        ),
        (
            "Existing final rows skipped for missing IDs: "
            f"{existing_summary['skipped_missing_ids']}"
        ),
        (
            "MLB fallback games checked: "
            f"{mlb_summary['api_checked']}"
        ),
        (
            "MLB fallback final rows added: "
            f"{mlb_summary['added']}"
        ),
        (
            "MLB fallback games not final: "
            f"{mlb_summary['api_not_final']}"
        ),
        (
            "MLB fallback API errors: "
            f"{mlb_summary['api_errors']}"
        ),
        (
            "MLB fallback final-score missing: "
            f"{mlb_summary['api_score_missing']}"
        ),
        (
            "MLB fallback team mismatches: "
            f"{mlb_summary['api_team_mismatch']}"
        ),
        f"Unresolved completed rows: {unresolved_count}",
        f"Parse errors encountered: {total_parse_errors}",
        f"Unknown status audit rows: {unknown_status_count}",
        (
            "Historical final-score files updated: "
            f"{legacy_summary['migrated_files']}"
        ),
        (
            "Historical final-score rows retained: "
            f"{legacy_summary['migrated_rows']}"
        ),
        (
            "Historical completed rows resolved/backfilled: "
            f"{legacy_summary['resolved_rows']}"
        ),
        (
            "Historical completed rows moved to unresolved audit: "
            f"{legacy_summary['unresolved_rows']}"
        ),
        f"Status audit: {STATUS_AUDIT_FILE}",
        f"Key audit: {KEY_AUDIT_FILE}",
        (
            "Unresolved completed-game audit: "
            f"{UNRESOLVED_AUDIT_FILE}"
        ),
    ]
    for line in summary_lines:
        log(line)

    _log_final_score_review_status(
        total_parse_errors,
        unresolved_count,
    )

    for output_path, count in files_written:
        log(
            f"  FILE: {output_path} ({count} rows)"
        )


def _run_final_score_build():
    files_written = []
    final_records_by_date = {}
    seen_by_game_id = {}
    seen_by_fallback_key = {}
    status_audit_rows = []
    key_audit_rows = []
    parse_error_rows = []
    unresolved_completed_rows = []

    (
        status_header,
        key_header,
        unresolved_header,
    ) = _final_score_audit_headers()

    raw_files, existing_summary = _load_final_score_raw_records(
        final_records_by_date,
        seen_by_game_id,
        seen_by_fallback_key,
        status_audit_rows,
        key_audit_rows,
        parse_error_rows,
        unresolved_completed_rows,
    )
    _abort_on_final_score_parse_errors(
        raw_files,
        parse_error_rows,
        unresolved_completed_rows,
    )

    mlb_summary = backfill_missing_finals_from_mlb(
        final_records_by_date=final_records_by_date,
        seen_by_game_id=seen_by_game_id,
        seen_by_fallback_key=seen_by_fallback_key,
        status_audit_rows=status_audit_rows,
        key_audit_rows=key_audit_rows,
    )

    _write_final_score_records(
        final_records_by_date,
        files_written,
    )

    legacy_summary = migrate_legacy_final_score_files(
        files_written,
        unresolved_completed_rows,
    )

    verify_final_score_outputs_have_gamepk()
    verify_doubleheader_identity_integrity()

    _write_final_score_audits(
        status_header,
        key_header,
        unresolved_header,
        status_audit_rows,
        key_audit_rows,
        unresolved_completed_rows,
    )

    total_parse_errors = len(parse_error_rows)
    unresolved_count = len(
        unresolved_completed_rows
    )
    unknown_status_count = _unknown_final_status_count(
        status_audit_rows
    )

    _log_final_score_summary(
        raw_files,
        files_written,
        final_records_by_date,
        seen_by_game_id,
        existing_summary,
        mlb_summary,
        legacy_summary,
        unresolved_count,
        total_parse_errors,
        unknown_status_count,
    )
    log_review_rows(
        parse_error_rows,
        unresolved_completed_rows,
    )
    log("STATUS: SUCCESS")


def main():
    try:
        _run_final_score_build()
    except Exception as error:
        log(
            f"FATAL ERROR: {error}\n"
            f"{traceback.format_exc()}"
        )
        log("STATUS: FAILED")
        raise

    print("MLB final-score build complete.")



if __name__ == "__main__":
    main()
