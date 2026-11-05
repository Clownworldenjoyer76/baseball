#!/usr/bin/env python3
"""Deterministic MLB probability / EV / Kelly invariant tests."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_repo_root() -> Path:
    file_path = Path(__file__).resolve()

    for parent in file_path.parents:
        if (
            parent
            / "docs/win/baseball/mlb/requirements.txt"
        ).exists():
            return parent

    raise RuntimeError(
        f"Could not resolve repository root from {file_path}"
    )


REPO_ROOT = _find_repo_root()

PROBABILITY_SCRIPT = (
    REPO_ROOT
    / "docs/win/baseball/mlb/scripts/01_merge/build_juice_files.py"
)

EV_KELLY_SCRIPT = (
    REPO_ROOT
    / "docs/win/baseball/mlb/scripts/03_edges/compute_ev_kelly.py"
)

ABS_TOL = 1e-10
SIGN_TOL = 1e-10


def _load_module(name: str, path: Path):
    if not path.exists():
        raise RuntimeError(
            f"Required production module not found: {path}"
        )

    module_dir = str(path.parent)
    added_to_path = module_dir not in sys.path

    if added_to_path:
        sys.path.insert(0, module_dir)

    try:
        spec = importlib.util.spec_from_file_location(
            name,
            path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not load production module: {path}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        return module

    finally:
        if added_to_path:
            sys.path.remove(module_dir)


PROBS = _load_module(
    "mlb_build_juice_files",
    PROBABILITY_SCRIPT,
)

EVK = _load_module(
    "mlb_compute_ev_kelly",
    EV_KELLY_SCRIPT,
)

EVALUATOR = _load_module(
    "mlb_evaluate_run_model_regression",
    REPO_ROOT / "docs/win/baseball/mlb/scripts/modeling/evaluate_run_model.py",
)
GAME_ID = _load_module(
    "mlb_game_id_pred_regression",
    REPO_ROOT / "docs/win/baseball/mlb/scripts/00_intake/game_id_pred.py",
)
TRAINING_BUILDER = _load_module(
    "mlb_build_run_training_set_regression",
    REPO_ROOT / "docs/win/baseball/mlb/scripts/modeling/build_run_training_set.py",
)


def _scalar(value) -> float:
    if isinstance(value, pd.Series):
        if len(value) != 1:
            raise AssertionError(
                f"Expected one-value Series, got {len(value)}"
            )

        return float(value.iloc[0])

    array = np.asarray(value, dtype=float)

    if array.size != 1:
        raise AssertionError(
            f"Expected scalar-like value, got shape={array.shape}"
        )

    return float(array.reshape(-1)[0])


def _binary_ev(
    probability: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        EVK.compute_binary_ev(
            pd.Series([probability], dtype=float),
            pd.Series([decimal_odds], dtype=float),
        )
    )


def _binary_kelly_raw(
    probability: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        EVK.compute_binary_kelly_raw(
            pd.Series([probability], dtype=float),
            pd.Series([decimal_odds], dtype=float),
        )
    )


def _total_ev(
    p_win: float,
    p_loss: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        EVK.compute_total_ev(
            pd.Series([p_win], dtype=float),
            pd.Series([p_loss], dtype=float),
            pd.Series([decimal_odds], dtype=float),
        )
    )


def _total_kelly_raw(
    p_win: float,
    p_loss: float,
    decimal_odds: float,
) -> float:
    return _scalar(
        EVK.compute_total_kelly_raw(
            pd.Series([p_win], dtype=float),
            pd.Series([p_loss], dtype=float),
            pd.Series([decimal_odds], dtype=float),
            pd.Series(["deterministic_test"]),
            "deterministic total test",
        )
    )


def _assert_close(
    actual: float,
    expected: float,
    tol: float = ABS_TOL,
) -> None:
    assert math.isfinite(actual), (
        f"actual is non-finite: {actual}"
    )

    assert math.isfinite(expected), (
        f"expected is non-finite: {expected}"
    )

    assert abs(actual - expected) <= tol, (
        f"actual={actual} "
        f"expected={expected} "
        f"diff={actual - expected}"
    )


def _sign(
    value: float,
    tol: float = SIGN_TOL,
) -> int:
    if value > tol:
        return 1

    if value < -tol:
        return -1

    return 0


RUN_LINE_SWAP_CASES = [
    (4.9, 3.6),
    (3.2, 5.0),
    (4.1, 4.1),
    (6.0, 2.8),
]


def _assert_binary_ev_kelly_contract(
    probability: float,
    decimal_odds: float,
    label: str,
) -> None:
    ev = _binary_ev(
        probability,
        decimal_odds,
    )
    raw_kelly = _binary_kelly_raw(
        probability,
        decimal_odds,
    )

    _assert_close(
        ev,
        (probability * decimal_odds) - 1.0,
    )

    assert _sign(raw_kelly) == _sign(ev), (
        f"{label} Kelly/EV sign mismatch: "
        f"p={probability} "
        f"decimal={decimal_odds} "
        f"ev={ev} "
        f"kelly={raw_kelly}"
    )


def _assert_run_line_home_away_swap() -> None:
    for home_runs, away_runs in RUN_LINE_SWAP_CASES:
        home_cover, away_cover = PROBS.run_line_probabilities(
            home_runs,
            away_runs,
            -1.5,
            1.5,
        )

        _assert_close(
            home_cover + away_cover,
            1.0,
        )

        (
            swapped_home_cover,
            swapped_away_cover,
        ) = PROBS.run_line_probabilities(
            away_runs,
            home_runs,
            1.5,
            -1.5,
        )

        _assert_close(
            swapped_home_cover,
            away_cover,
        )
        _assert_close(
            swapped_away_cover,
            home_cover,
        )


def test_moneyline_probability_ev_kelly_contract() -> None:
    rows = [
        (4.8, 3.6, 1.80, 2.20),
        (3.9, 4.2, 2.05, 1.85),
        (5.1, 5.1, 1.95, 1.95),
        (2.7, 4.9, 2.75, 1.50),
    ]

    for (
        home_runs,
        away_runs,
        home_decimal,
        away_decimal,
    ) in rows:
        (
            home_prob,
            away_prob,
            _tie_raw,
        ) = PROBS.moneyline_probabilities(
            home_runs,
            away_runs,
        )

        _assert_close(
            home_prob + away_prob,
            1.0,
        )

        for probability, decimal_odds in [
            (home_prob, home_decimal),
            (away_prob, away_decimal),
        ]:
            _assert_binary_ev_kelly_contract(
                probability,
                decimal_odds,
                "Moneyline",
            )


def test_run_line_complement_and_home_away_swap() -> None:
    _assert_run_line_home_away_swap()


def test_run_line_probability_contract_and_ranking() -> None:
    cases = [
        (2.5, 6.0),
        (3.0, 5.5),
        (3.5, 5.0),
        (4.0, 4.5),
        (4.5, 4.0),
        (5.0, 3.5),
        (5.5, 3.0),
        (6.0, 2.5),
    ]

    home_cover_probs = []

    for home_runs, away_runs in cases:
        (
            home_cover,
            away_cover,
        ) = PROBS.run_line_probabilities(
            home_runs,
            away_runs,
            -1.5,
            1.5,
        )

        _assert_close(
            home_cover + away_cover,
            1.0,
        )

        assert 0.0 <= home_cover <= 1.0
        assert 0.0 <= away_cover <= 1.0

        home_cover_probs.append(
            home_cover
        )

    assert all(
        right > left
        for left, right in zip(
            home_cover_probs,
            home_cover_probs[1:],
        )
    )


def test_run_line_probability_home_away_swap() -> None:
    _assert_run_line_home_away_swap()


def test_run_line_ev_kelly_contract() -> None:
    cases = [
        (4.9, 3.6, -1.5, 1.5, 1.95, 1.87),
        (3.2, 5.0, 1.5, -1.5, 1.80, 2.05),
        (4.1, 4.1, -1.5, 1.5, 2.20, 1.70),
        (6.0, 2.8, -1.5, 1.5, 1.72, 2.25),
    ]

    for (
        home_runs,
        away_runs,
        home_line,
        away_line,
        home_decimal,
        away_decimal,
    ) in cases:
        home_prob, away_prob = PROBS.run_line_probabilities(
            home_runs,
            away_runs,
            home_line,
            away_line,
        )

        _assert_close(
            home_prob + away_prob,
            1.0,
        )

        for probability, decimal_odds in [
            (home_prob, home_decimal),
            (away_prob, away_decimal),
        ]:
            _assert_binary_ev_kelly_contract(
                probability,
                decimal_odds,
                "Run-line",
            )


def test_half_run_total_has_no_push_and_resolves_to_one() -> None:
    cases = [
        (4.6, 3.8, 7.5),
        (5.0, 4.0, 8.5),
        (3.2, 3.4, 6.5),
        (5.5, 4.7, 10.5),
    ]

    for (
        home_runs,
        away_runs,
        total_line,
    ) in cases:
        (
            p_over,
            p_under,
            p_push,
        ) = PROBS.totals_probabilities(
            home_runs,
            away_runs,
            total_line,
        )

        _assert_close(
            p_push,
            0.0,
        )

        _assert_close(
            p_over + p_under,
            1.0,
        )


def test_whole_run_total_probability_and_push_aware_ev() -> None:
    cases = [
        (4.6, 3.8, 8.0, 2.05),
        (5.1, 4.2, 9.0, 1.91),
        (3.4, 3.3, 7.0, 2.20),
    ]

    for (
        home_runs,
        away_runs,
        total_line,
        decimal_odds,
    ) in cases:
        (
            p_over,
            p_under,
            p_push,
        ) = PROBS.totals_probabilities(
            home_runs,
            away_runs,
            total_line,
        )

        _assert_close(
            p_over + p_under + p_push,
            1.0,
        )

        over_ev = _total_ev(
            p_over,
            p_under,
            decimal_odds,
        )

        expected_over_ev = (
            p_over * (decimal_odds - 1.0)
            - p_under
        )

        _assert_close(
            over_ev,
            expected_over_ev,
        )

        under_ev = _total_ev(
            p_under,
            p_over,
            decimal_odds,
        )

        expected_under_ev = (
            p_under * (decimal_odds - 1.0)
            - p_over
        )

        _assert_close(
            under_ev,
            expected_under_ev,
        )


def test_model_probabilities_are_price_independent() -> None:
    model_home_runs = 4.8
    model_away_runs = 3.7
    home_line = -1.5
    away_line = 1.5

    prices_a = {
        "home": 1.80,
        "away": 2.10,
    }

    prices_b = {
        "home": 2.25,
        "away": 1.70,
    }

    probs_a = PROBS.run_line_probabilities(
        model_home_runs,
        model_away_runs,
        home_line,
        away_line,
    )

    probs_b = PROBS.run_line_probabilities(
        model_home_runs,
        model_away_runs,
        home_line,
        away_line,
    )

    _assert_close(
        probs_a[0],
        probs_b[0],
    )

    _assert_close(
        probs_a[1],
        probs_b[1],
    )

    home_prob = probs_a[0]

    break_even_a = (
        1.0 / prices_a["home"]
    )

    break_even_b = (
        1.0 / prices_b["home"]
    )

    edge_a = (
        home_prob - break_even_a
    )

    edge_b = (
        home_prob - break_even_b
    )

    ev_a = _binary_ev(
        home_prob,
        prices_a["home"],
    )

    ev_b = _binary_ev(
        home_prob,
        prices_b["home"],
    )

    kelly_a = _binary_kelly_raw(
        home_prob,
        prices_a["home"],
    )

    kelly_b = _binary_kelly_raw(
        home_prob,
        prices_b["home"],
    )

    assert not math.isclose(
        break_even_a,
        break_even_b,
        abs_tol=ABS_TOL,
    )

    assert not math.isclose(
        edge_a,
        edge_b,
        abs_tol=ABS_TOL,
    )

    assert not math.isclose(
        ev_a,
        ev_b,
        abs_tol=ABS_TOL,
    )

    assert not math.isclose(
        kelly_a,
        kelly_b,
        abs_tol=ABS_TOL,
    )


def test_binary_ev_and_raw_kelly_sign_consistency_grid() -> None:
    probabilities = [
        0.05,
        0.15,
        0.25,
        0.40,
        0.50,
        0.60,
        0.75,
        0.90,
        0.95,
    ]

    decimal_prices = [
        1.20,
        1.40,
        1.60,
        1.80,
        2.00,
        2.25,
        2.75,
        3.50,
        5.00,
    ]

    for probability in probabilities:
        for decimal_odds in decimal_prices:
            ev = _binary_ev(
                probability,
                decimal_odds,
            )

            raw_kelly = _binary_kelly_raw(
                probability,
                decimal_odds,
            )

            assert _sign(ev) == _sign(raw_kelly), (
                f"Binary sign mismatch: "
                f"p={probability} "
                f"decimal={decimal_odds} "
                f"ev={ev} "
                f"kelly={raw_kelly}"
            )

            if ev > SIGN_TOL:
                assert raw_kelly > SIGN_TOL

            elif ev < -SIGN_TOL:
                assert raw_kelly < -SIGN_TOL

            else:
                assert abs(raw_kelly) <= SIGN_TOL


def test_push_aware_total_ev_and_raw_kelly_sign_consistency_grid() -> None:
    total_cases = [
        PROBS.totals_probabilities(
            4.0,
            3.5,
            7.0,
        ),
        PROBS.totals_probabilities(
            4.8,
            4.1,
            9.0,
        ),
        PROBS.totals_probabilities(
            5.3,
            3.9,
            8.0,
        ),
        PROBS.totals_probabilities(
            3.2,
            3.1,
            6.0,
        ),
    ]

    decimal_prices = [
        1.50,
        1.75,
        1.91,
        2.00,
        2.20,
        2.50,
        3.00,
    ]

    for (
        p_over,
        p_under,
        p_push,
    ) in total_cases:
        _assert_close(
            p_over + p_under + p_push,
            1.0,
        )

        for p_win, p_loss in [
            (p_over, p_under),
            (p_under, p_over),
        ]:
            resolved_mass = (
                p_win + p_loss
            )

            assert resolved_mass > 0

            for decimal_odds in decimal_prices:
                ev = _total_ev(
                    p_win,
                    p_loss,
                    decimal_odds,
                )

                raw_kelly = _total_kelly_raw(
                    p_win,
                    p_loss,
                    decimal_odds,
                )

                assert _sign(ev) == _sign(raw_kelly), (
                    f"Total sign mismatch: "
                    f"p_win={p_win} "
                    f"p_loss={p_loss} "
                    f"p_push={p_push} "
                    f"decimal={decimal_odds} "
                    f"ev={ev} "
                    f"kelly={raw_kelly}"
                )

                if ev > SIGN_TOL:
                    assert raw_kelly > SIGN_TOL

                elif ev < -SIGN_TOL:
                    assert raw_kelly < -SIGN_TOL

                else:
                    assert abs(raw_kelly) <= SIGN_TOL


# Regression coverage for the 2026-09-20 historical source/enriched mismatch.
def _regression_market_frame(home_prob=float("nan"), away_prob=float("nan")):
    return pd.DataFrame([{
        "game_id": "401817012",
        "dratings_home_prob": home_prob,
        "dratings_away_prob": away_prob,
        "dratings_home_projected_runs": 4.72,
        "dratings_away_projected_runs": 2.94,
        "model_home_runs": 4.4,
        "model_away_runs": 3.2,
        "home_run_line": -1.5,
        "away_run_line": 1.5,
        "total": 7.5,
    }])


def test_evaluator_ignores_missing_input_probabilities_when_validating_derived():
    # Missing DRatings input features must not be misclassified as bad derived
    # probabilities. Both independent run projection systems remain validated.
    result = EVALUATOR.derive_market_probabilities(
        _regression_market_frame(), PROBS,
    )
    assert pd.isna(result.loc[0, "dratings_home_prob"])
    assert pd.isna(result.loc[0, "dratings_away_prob"])
    for system in ("dratings", "new_model"):
        for suffix in (
            "home_ml_prob", "away_ml_prob", "home_rl_prob", "away_rl_prob",
            "over_total_win_prob", "under_total_win_prob", "total_push_prob",
            "over_total_conditional_prob", "under_total_conditional_prob",
        ):
            p = float(result.loc[0, f"{system}_{suffix}"])
            assert math.isfinite(p) and 0 <= p <= 1


def test_evaluator_keeps_generated_probability_validation():
    class BadProbabilities:
        moneyline_probabilities = staticmethod(lambda *args: (float("nan"), 0.5, 0.0))
        run_line_probabilities = staticmethod(PROBS.run_line_probabilities)
        totals_probabilities = staticmethod(PROBS.totals_probabilities)

    import pytest
    with pytest.raises(RuntimeError, match="Derived probability dratings_home_ml_prob"):
        EVALUATOR.derive_market_probabilities(
            _regression_market_frame(0.6, 0.4), BadProbabilities,
        )


def _regression_prediction_row(game_id, home_prob, away_prob):
    return {
        "game_id": game_id,
        "sport": "baseball",
        "league": "mlb",
        "game_date": "2026_09_20",
        "game_time": "01:40 PM",
        "home_team": "Cleveland Guardians",
        "away_team": "Athletics",
        "home_pitcher": "",
        "away_pitcher": "",
        "home_prob": home_prob,
        "away_prob": away_prob,
        "away_projected_runs": "2.94",
        "home_projected_runs": "4.72",
        "total_projected_runs": "7.66",
    }


def _write_regression_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAME_ID.OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_historical_reconciliation_restores_only_missing_probabilities(tmp_path, monkeypatch):
    assert "LOG_FILE" in vars(GAME_ID)
    monkeypatch.setitem(vars(GAME_ID), "LOG_FILE", tmp_path / "game_id_pred.log")
    source_path = tmp_path / "original.csv"
    output_path = tmp_path / "enriched.csv"
    original = _regression_prediction_row("401817012", "0.674", "0.326")
    stale = _regression_prediction_row("401817012", "", "")
    _write_regression_csv(source_path, [original])
    _write_regression_csv(output_path, [stale])

    assert GAME_ID.reconcile_historical_date(
        "2026_09_20", source_path, output_path,
    ) == 1
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [original]
    first_bytes = output_path.read_bytes()
    assert GAME_ID.reconcile_historical_date(
        "2026_09_20", source_path, output_path,
    ) == 0
    assert output_path.read_bytes() == first_bytes


def test_historical_reconciliation_preserves_conflicting_existing_probabilities(tmp_path, monkeypatch):
    assert "LOG_FILE" in vars(GAME_ID)
    monkeypatch.setitem(vars(GAME_ID), "LOG_FILE", tmp_path / "game_id_pred.log")
    source_path = tmp_path / "original.csv"
    output_path = tmp_path / "enriched.csv"
    _write_regression_csv(source_path, [
        _regression_prediction_row("401817012", "0.674", "0.326"),
    ])
    _write_regression_csv(output_path, [
        _regression_prediction_row("401817012", "0.600", "0.400"),
    ])
    original_bytes = output_path.read_bytes()
    assert GAME_ID.reconcile_historical_date("2026_09_20", source_path, output_path) == 0
    assert output_path.read_bytes() == original_bytes
    assert "source/enriched probability disagreement" in (
        tmp_path / "game_id_pred.log"
    ).read_text()


def test_training_probability_audit_allows_absent_pair_but_rejects_partial(tmp_path, monkeypatch):
    import pytest
    assert "LOG_FILE" in vars(TRAINING_BUILDER)
    monkeypatch.setitem(vars(TRAINING_BUILDER), "LOG_FILE", tmp_path / "build.log")
    summary = {"missing_dratings_prob_rows": 0, "invalid_dratings_prob_rows": 0}
    frame = pd.DataFrame([{
        "game_id": "401817012", "dratings_home_prob": "", "dratings_away_prob": "",
    }])
    TRAINING_BUILDER._audit_dratings_probabilities(frame, summary, "test")
    assert summary["missing_dratings_prob_rows"] == 1
    frame.loc[0, "dratings_home_prob"] = "0.6"
    with pytest.raises(RuntimeError, match="invalid or partial"):
        TRAINING_BUILDER._audit_dratings_probabilities(frame, summary, "test")
