#!/usr/bin/env python3
"""Grade morning MLB picks using current markets_AM.yaml rules.

The upstream morning selector rebuilds historical daily selections. This wrapper
also re-filters previously selected historical rows before grading, so stale
or old-configuration picks cannot enter the morning graded master or reports.
No selection CSV or locked-bet archive is modified by this grading step.
"""

from pathlib import Path

import pandas as pd

from results_am_common import configure_morning_grade, load_sibling

GRADE_DIR = Path(__file__).parent
AUDIT_FILENAME = "morning_historical_refilter_audit.csv"


def _filter_with_morning_rules(bets):
    """Return (eligible rows, exclusion audit), using selector's exact band logic."""
    from importlib.util import module_from_spec, spec_from_file_location
    import yaml

    selector_path = GRADE_DIR.parent / "04_select" / "baseball_select_bets.py"
    config_path = Path("docs/win/baseball/mlb/config/markets_AM.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["markets"]["mlb"]

    spec = spec_from_file_location("mlb_morning_grading_rule_engine", selector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load selection rules: {selector_path}")
    selector = module_from_spec(spec)
    spec.loader.exec_module(selector)
    selector.CONFIG = config
    selector.FILTERS = config
    selector.validate_config()

    required = {"game_id", "game_date", "market_type", "bet_side",
                "dk_odds_american", "model_prob", "ev", "kelly", "line"}
    missing = required - set(bets.columns)
    if missing:
        raise ValueError(f"Morning historical filter missing columns: {sorted(missing)}")

    if bets.empty:
        return bets.copy(), pd.DataFrame(columns=list(bets.columns)+["refilter_reason"])

    keep_indices = []
    rejected = []

    for index, row in bets.iterrows():
        market = str(row["market_type"]).strip().lower()
        side = str(row["bet_side"]).strip().lower()
        market_cfg = config.get(market)
        if not isinstance(market_cfg, dict):
            raise ValueError(f"Unknown morning market: {market}")
        rules = market_cfg.get(side)
        if not isinstance(rules, dict):
            raise ValueError(f"Unknown morning market/side: {market}/{side}")

        if not market_cfg.get("enabled", False) or not rules.get("enabled", False):
            reason = "market_or_side_disabled"
        else:
            prob = selector.fv(row.get("model_prob"))
            ev = selector.fv(row.get("ev"))
            kelly = selector.fv(row.get("kelly"))
            odds = selector.fv(row.get("dk_odds_american"))
            line = selector.fv(row.get("line"))
            passed, _, reason = selector.check_rules(
                ev, kelly, odds, line, prob, rules, selector.init_counter()
            )
            if passed:
                keep_indices.append(index)
                continue

        excluded = row.to_dict()
        excluded["refilter_reason"] = reason
        rejected.append(excluded)

    eligible = bets.loc[keep_indices].copy()

    # Ensure the same pick_preference applies to any historical CSVs containing
    # both sides for one game/market. This cannot invent a missing candidate:
    # the morning selector must be rerun to obtain newly qualifying picks.
    if not eligible.empty:
        selected_indices = []
        for (_, _, market), group in eligible.groupby(
            ["game_date", "game_id", "market_type"], sort=False, dropna=False
        ):
            preference = config[str(market).strip().lower()].get(
                "pick_preference", "best_ev"
            )
            if preference == "all" and str(market).strip().lower() != "run_line":
                selected_indices.extend(group.index)
                continue
            key = "model_prob" if preference == "best_prob" else "ev"
            values = pd.to_numeric(group[key], errors="coerce")
            if values.isna().all():
                raise ValueError(f"Cannot select best {key}: no numeric values")
            chosen = values.idxmax()
            selected_indices.append(chosen)
            for index, row in group.iterrows():
                if index != chosen:
                    excluded = row.to_dict()
                    excluded["refilter_reason"] = f"not_{preference}_for_game_market"
                    rejected.append(excluded)
        eligible = eligible.loc[selected_indices].copy()

    audit = pd.DataFrame(rejected, columns=list(bets.columns) + ["refilter_reason"])
    return eligible, audit


def refilter_historical_bets(bets, *, audit_path=None):
    """Apply active AM selection bands to rows; optionally save exclusions."""
    filtered, audit = _filter_with_morning_rules(bets)
    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)
    return filtered


grade = load_sibling("01_mlb_results_grade.py", "mlb_results_grade_shared")
configure_morning_grade(grade)
_original_load_selected_bets = grade.load_selected_bets


def _load_selected_bets_with_morning_refilter():
    bets, duplicate_audit = _original_load_selected_bets()
    # Refresh the audit on every grading run, even if no selections exist.
    audit_path = grade.AUDIT_DIR / AUDIT_FILENAME
    before = len(bets)
    filtered = refilter_historical_bets(bets, audit_path=audit_path)
    print(
        f"MORNING CONFIG REFILTER | before={before} after={len(filtered)} "
        f"excluded={before - len(filtered)} | audit={audit_path}"
    )
    return filtered, duplicate_audit


grade.load_selected_bets = _load_selected_bets_with_morning_refilter

if __name__ == "__main__":
    grade.main("01_mlb_results_grade_AM.py")
