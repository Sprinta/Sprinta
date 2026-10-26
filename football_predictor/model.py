"""Time-weighted Dixon-Coles Poisson model.

Expected goals:
    home: base * home_adv * attack[home] * defence[away]
    away: base *            attack[away] * defence[home]

attack > 1 scores more than an average side; defence < 1 concedes less.
Recent matches weigh more (exponential decay), and teams with little data
are shrunk toward a prior so a few lucky results don't dominate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

from .data import Match

MAX_GOALS = 10
# Newly promoted sides start below league average until results say otherwise.
PROMOTED_PRIOR = (0.85, 1.15)


@dataclass
class Ratings:
    attack: dict[str, float]
    defence: dict[str, float]
    base: float
    home_adv: float
    rho: float
    as_of: date
    matches_used: int

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        for team in (home, away):
            if team not in self.attack:
                raise KeyError(f"No data for team '{team}'")
        lam = self.base * self.home_adv * self.attack[home] * self.defence[away]
        mu = self.base * self.attack[away] * self.defence[home]
        return lam, mu


@dataclass
class Prediction:
    home: str
    away: str
    home_xg: float
    away_xg: float
    matrix: list[list[float]] = field(repr=False)
    p_home: float
    p_draw: float
    p_away: float
    over_2_5: float
    btts: float
    top_scores: list[tuple[int, int, float]]

    @property
    def probs(self) -> tuple[float, float, float]:
        return self.p_home, self.p_draw, self.p_away


def _tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction for the 0-0, 1-0, 0-1, 1-1 scores."""
    if i == 0 and j == 0:
        return 1 - lam * mu * rho
    if i == 0 and j == 1:
        return 1 + lam * rho
    if i == 1 and j == 0:
        return 1 + mu * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def _poisson_row(lam: float, n: int) -> list[float]:
    row = [math.exp(-lam)]
    for k in range(1, n + 1):
        row.append(row[-1] * lam / k)
    return row


def promoted_teams(matches: list[Match]) -> set[str]:
    """Teams in the latest season that were absent from the season before."""
    seasons = sorted({m.season for m in matches})
    if len(seasons) < 2:
        return set()
    def teams_in(s: str) -> set[str]:
        return {t for m in matches if m.season == s for t in (m.home, m.away)}
    return teams_in(seasons[-1]) - teams_in(seasons[-2])


def fit(
    matches: Iterable[Match],
    as_of: Optional[date] = None,
    half_life_days: float = 300,
    prior_strength: float = 3.0,
    promoted: Optional[set[str]] = None,
    init: Optional[Ratings] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Ratings:
    """Fit ratings on matches played strictly before `as_of` (default: all)."""
    ms = [m for m in matches if as_of is None or m.date < as_of]
    if not ms:
        raise ValueError("No matches to fit on")
    ref = as_of or max(m.date for m in ms)
    xi = math.log(2) / half_life_days
    promoted = promoted or set()

    teams = sorted({t for m in ms for t in (m.home, m.away)})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = [idx[m.home] for m in ms]
    A = [idx[m.away] for m in ms]
    HG = [m.home_goals for m in ms]
    AG = [m.away_goals for m in ms]
    W = [math.exp(-xi * (ref - m.date).days) for m in ms]
    rows = list(zip(H, A, HG, AG, W))

    sum_w = sum(W)
    w_home_goals = sum(w * hg for hg, w in zip(HG, W))
    w_away_goals = sum(w * ag for ag, w in zip(AG, W))
    goals_per_team = (w_home_goals + w_away_goals) / (2 * sum_w)
    prior_a = [PROMOTED_PRIOR[0] if t in promoted else 1.0 for t in teams]
    prior_d = [PROMOTED_PRIOR[1] if t in promoted else 1.0 for t in teams]
    k = prior_strength * goals_per_team  # prior expressed in goals

    if init:
        att = [init.attack.get(t, prior_a[i]) for i, t in enumerate(teams)]
        dfn = [init.defence.get(t, prior_d[i]) for i, t in enumerate(teams)]
        base, home = init.base, init.home_adv
    else:
        att, dfn = prior_a[:], prior_d[:]
        base = w_away_goals / sum_w
        home = w_home_goals / max(w_away_goals, 1e-9)

    for _ in range(max_iter):
        old = att + dfn + [home]

        num = [k * p for p in prior_a]
        den = [k] * n
        for h, a, hg, ag, w in rows:
            num[h] += w * hg
            den[h] += w * base * home * dfn[a]
            num[a] += w * ag
            den[a] += w * base * dfn[h]
        att = [x / y for x, y in zip(num, den)]

        num = [k * p for p in prior_d]
        den = [k] * n
        for h, a, hg, ag, w in rows:
            num[a] += w * hg
            den[a] += w * base * home * att[h]
            num[h] += w * ag
            den[h] += w * base * att[a]
        dfn = [x / y for x, y in zip(num, den)]

        # Keep the average team at 1.0 and let `base` carry the scale.
        ma, md = sum(att) / n, sum(dfn) / n
        att = [x / ma for x in att]
        dfn = [x / md for x in dfn]

        home_exp = sum(w * att[h] * dfn[a] for h, a, _, _, w in rows)
        away_exp = sum(w * att[a] * dfn[h] for h, a, _, _, w in rows)
        home = (w_home_goals / home_exp) / (w_away_goals / away_exp)
        base = (w_home_goals + w_away_goals) / (home * home_exp + away_exp)

        if max(abs(x - y) for x, y in zip(att + dfn + [home], old)) < tol:
            break

    rho = _fit_rho(rows, att, dfn, base, home)
    return Ratings(
        attack=dict(zip(teams, att)),
        defence=dict(zip(teams, dfn)),
        base=base,
        home_adv=home,
        rho=rho,
        as_of=ref,
        matches_used=len(ms),
    )


def _fit_rho(rows, att, dfn, base, home) -> float:
    """Grid-search the low-score correction; only 0/1-goal games depend on it."""
    low = []
    for h, a, hg, ag, w in rows:
        if hg <= 1 and ag <= 1:
            low.append((hg, ag, base * home * att[h] * dfn[a], base * att[a] * dfn[h], w))
    best, best_ll = 0.0, -math.inf
    for step in range(-25, 16):
        rho = step / 100
        ll = 0.0
        for hg, ag, lam, mu, w in low:
            t = _tau(hg, ag, lam, mu, rho)
            if t <= 0:
                ll = -math.inf
                break
            ll += w * math.log(t)
        if ll > best_ll:
            best, best_ll = rho, ll
    return best


def predict(ratings: Ratings, home: str, away: str) -> Prediction:
    lam, mu = ratings.expected_goals(home, away)
    ph, pa = _poisson_row(lam, MAX_GOALS), _poisson_row(mu, MAX_GOALS)
    matrix = [[_tau(i, j, lam, mu, ratings.rho) * ph[i] * pa[j]
               for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    total = sum(map(sum, matrix))
    matrix = [[p / total for p in row] for row in matrix]

    p_home = p_draw = p_away = over = btts = 0.0
    cells = []
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i + j > 2:
                over += p
            if i > 0 and j > 0:
                btts += p
            cells.append((i, j, p))
    cells.sort(key=lambda c: c[2], reverse=True)
    return Prediction(home, away, lam, mu, matrix, p_home, p_draw, p_away,
                      over, btts, cells[:5])
