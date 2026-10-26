"""Walk-forward backtest: refit each week on past data only, predict that week.

This is the honest test of the model. Nothing from the future leaks in, and
the bookmaker's closing odds are scored alongside as the benchmark to beat.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Optional

from .analysis import Probs, blend_h2h, h2h_probs, head_to_head, implied_probs
from .config import Settings
from .data import Match
from .model import fit, predict, promoted_teams

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
H2H_WEIGHTS = [i / 20 for i in range(11)]  # 0.00 .. 0.50


@dataclass
class MethodScore:
    name: str
    n: int
    accuracy: float  # share of matches where the favourite won
    brier: float     # lower is better; 0.667 = coin-flip-ish, ~0.57-0.60 is good
    log_loss: float  # lower is better


@dataclass
class BacktestResult:
    season: str
    scores: list[MethodScore]
    best_h2h_weight: float
    skipped: int


def _score(name: str, rows: list[tuple[Probs, str]]) -> MethodScore:
    if not rows:
        return MethodScore(name, 0, math.nan, math.nan, math.nan)
    acc = brier = ll = 0.0
    for p, res in rows:
        k = OUTCOME_INDEX[res]
        acc += max(range(3), key=lambda i: p[i]) == k
        brier += sum((p[i] - (i == k)) ** 2 for i in range(3))
        ll -= math.log(max(p[k], 1e-12))
    n = len(rows)
    return MethodScore(name, n, acc / n, brier / n, ll / n)


def run_backtest(matches: list[Match], season: str, settings: Settings,
                 progress: Optional[Callable[[str], None]] = None) -> BacktestResult:
    test = [m for m in matches if m.season == season]
    if not test:
        raise ValueError(f"No matches in season {season}")
    history = [m for m in matches if m.season <= season]
    promoted = promoted_teams(history)

    weeks: dict = defaultdict(list)
    for m in test:
        weeks[m.date - timedelta(days=m.date.weekday())].append(m)

    model_rows, book_rows, base_rows = [], [], []
    pairs: list[tuple[Probs, Probs, str]] = []  # (model, h2h, result) for weight sweep
    skipped = 0
    ratings = None
    for i, week in enumerate(sorted(weeks)):
        if progress:
            progress(f"Backtesting week {i + 1}/{len(weeks)}...")
        past = [m for m in history if m.date < week]
        if len(past) < 50:
            skipped += len(weeks[week])
            continue
        ratings = fit(past, half_life_days=settings.half_life_days,
                      prior_strength=settings.prior_strength,
                      promoted=promoted, init=ratings)
        recent = past[-760:]
        base = tuple(sum(m.result == r for m in recent) / len(recent) for r in "HDA")

        for m in weeks[week]:
            if m.home not in ratings.attack or m.away not in ratings.attack:
                skipped += 1
                continue
            p = predict(ratings, m.home, m.away).probs
            h = h2h_probs(head_to_head(past, m.home, m.away, settings.h2h_matches), p, m.date)
            model_rows.append((p, m.result))
            base_rows.append((base, m.result))
            pairs.append((p, h, m.result))
            if m.odds:
                book_rows.append((implied_probs(m.odds), m.result))

    sweep = {w: _score("", [(blend_h2h(p, h, w), r) for p, h, r in pairs]).log_loss
             for w in H2H_WEIGHTS} if pairs else {0.0: math.nan}
    best_w = min(sweep, key=lambda w: sweep[w])
    blended = [(blend_h2h(p, h, settings.h2h_weight), r) for p, h, r in pairs]

    scores = [
        _score("Model (past results only)", model_rows),
        _score(f"Model + H2H (weight {settings.h2h_weight:.2f})", blended),
        _score("Bookmaker closing odds", book_rows),
        _score("League base rates (no skill)", base_rows),
    ]
    return BacktestResult(season, scores, best_w, skipped)
