"""Recent form, head-to-head records, and blending H2H into the model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .data import Match

Probs = tuple[float, float, float]


@dataclass
class Form:
    team: str
    results: str  # e.g. "WWDLW", most recent last
    points: int
    goals_for: int
    goals_against: int
    matches: list[Match]


def form(matches: list[Match], team: str, n: int = 6,
         venue: Optional[str] = None, before: Optional[date] = None) -> Form:
    """Last n results for team. venue: None, 'home' or 'away'."""
    picked = []
    for m in reversed(matches):
        if before and m.date >= before:
            continue
        if (m.home == team and venue != "away") or (m.away == team and venue != "home"):
            picked.append(m)
            if len(picked) == n:
                break
    picked.reverse()
    letters, pts, gf, ga = [], 0, 0, 0
    for m in picked:
        scored, conceded = (m.home_goals, m.away_goals) if m.home == team else (m.away_goals, m.home_goals)
        gf += scored
        ga += conceded
        if scored > conceded:
            letters.append("W"); pts += 3
        elif scored == conceded:
            letters.append("D"); pts += 1
        else:
            letters.append("L")
    return Form(team, "".join(letters), pts, gf, ga, picked)


@dataclass
class HeadToHead:
    home: str
    away: str
    matches: list[Match]  # most recent first
    home_wins: int        # wins for `home` regardless of venue
    draws: int
    away_wins: int
    home_goals: int
    away_goals: int


def head_to_head(matches: list[Match], home: str, away: str, n: int = 10,
                 before: Optional[date] = None) -> HeadToHead:
    picked = []
    for m in reversed(matches):
        if before and m.date >= before:
            continue
        if {m.home, m.away} == {home, away}:
            picked.append(m)
            if len(picked) == n:
                break
    hw = d = aw = hg = ag = 0
    for m in picked:
        g_home, g_away = (m.home_goals, m.away_goals) if m.home == home else (m.away_goals, m.home_goals)
        hg += g_home
        ag += g_away
        if g_home > g_away:
            hw += 1
        elif g_home == g_away:
            d += 1
        else:
            aw += 1
    return HeadToHead(home, away, picked, hw, d, aw, hg, ag)


def h2h_probs(h2h: HeadToHead, model: Probs, ref: date,
              half_life_days: float = 3 * 365, prior_strength: float = 3.0) -> Probs:
    """Outcome probabilities implied by past meetings.

    Recent meetings count more, and results are smoothed toward the model so
    that two or three meetings can't produce extreme numbers on their own.
    Meetings at the other venue count half, since home advantage flips.
    """
    xi = math.log(2) / half_life_days
    counts = [prior_strength * p for p in model]
    for m in h2h.matches:
        w = math.exp(-xi * (ref - m.date).days)
        if m.home != h2h.home:
            w *= 0.5
        g_home, g_away = (m.home_goals, m.away_goals) if m.home == h2h.home else (m.away_goals, m.home_goals)
        counts[0 if g_home > g_away else 1 if g_home == g_away else 2] += w
    total = sum(counts)
    return counts[0] / total, counts[1] / total, counts[2] / total


def blend_h2h(model: Probs, h2h: Probs, weight: float) -> Probs:
    weight = min(max(weight, 0.0), 1.0)
    return tuple((1 - weight) * m + weight * h for m, h in zip(model, h2h))  # type: ignore[return-value]


def implied_probs(odds: tuple[float, float, float]) -> Probs:
    """Bookmaker odds -> probabilities with the margin removed."""
    inv = [1 / o for o in odds]
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s
