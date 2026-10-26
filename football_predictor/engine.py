"""Ties data, model and analysis together. Used by both the GUI and the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from .analysis import (Form, HeadToHead, Probs, blend_h2h, form, h2h_probs,
                       head_to_head, implied_probs)
from .backtest import BacktestResult, run_backtest
from .config import LEAGUES, Settings
from .data import DataError, Fixture, Match, load_fixtures, load_league
from .model import Prediction, Ratings, fit, predict, promoted_teams

Progress = Optional[Callable[[str], None]]


@dataclass
class MatchReport:
    league: str
    prediction: Prediction
    model_probs: Probs
    h2h_probs: Probs
    final_probs: Probs
    h2h_weight: float
    h2h: HeadToHead
    home_form: Form
    away_form: Form
    home_form_at_home: Form
    away_form_away: Form
    home_rating: tuple[float, float]  # (attack, defence)
    away_rating: tuple[float, float]
    odds_probs: Optional[Probs] = None

    @property
    def home(self) -> str:
        return self.prediction.home

    @property
    def away(self) -> str:
        return self.prediction.away

    @property
    def verdict(self) -> str:
        labels = (f"{self.home} win", "Draw", f"{self.away} win")
        i = max(range(3), key=lambda k: self.final_probs[k])
        return labels[i]


class Engine:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        self._cache: dict[str, tuple[list[Match], Ratings]] = {}

    def invalidate(self) -> None:
        self._cache.clear()

    def load(self, league: str, refresh: bool = False, progress: Progress = None) -> list[Match]:
        if league in self._cache and not refresh:
            return self._cache[league][0]
        matches = load_league(league, self.settings.seasons, refresh, progress)
        if progress:
            progress(f"Fitting {LEAGUES.get(league, league)} model...")
        ratings = fit(matches, as_of=date.today(),
                      half_life_days=self.settings.half_life_days,
                      prior_strength=self.settings.prior_strength,
                      promoted=promoted_teams(matches))
        self._cache[league] = (matches, ratings)
        return matches

    def ratings(self, league: str) -> Ratings:
        self.load(league)
        return self._cache[league][1]

    def teams(self, league: str) -> list[str]:
        matches = self.load(league)
        latest = matches[-1].season
        return sorted({t for m in matches if m.season == latest for t in (m.home, m.away)})

    def table(self, league: str) -> list[tuple[str, float, float, float]]:
        """(team, attack, defence, overall strength) for the current season's teams."""
        r = self.ratings(league)
        rows = [(t, r.attack[t], r.defence[t], r.attack[t] / r.defence[t]) for t in self.teams(league)]
        return sorted(rows, key=lambda x: x[3], reverse=True)

    def predict(self, league: str, home: str, away: str,
                odds: Optional[tuple[float, float, float]] = None,
                progress: Progress = None) -> MatchReport:
        if home == away:
            raise ValueError("Pick two different teams.")
        matches = self.load(league, progress=progress)
        ratings = self._cache[league][1]
        pred = predict(ratings, home, away)
        s = self.settings
        h2h = head_to_head(matches, home, away, s.h2h_matches)
        hp = h2h_probs(h2h, pred.probs, date.today())
        return MatchReport(
            league=league,
            prediction=pred,
            model_probs=pred.probs,
            h2h_probs=hp,
            final_probs=blend_h2h(pred.probs, hp, s.h2h_weight),
            h2h_weight=s.h2h_weight,
            h2h=h2h,
            home_form=form(matches, home),
            away_form=form(matches, away),
            home_form_at_home=form(matches, home, venue="home"),
            away_form_away=form(matches, away, venue="away"),
            home_rating=(ratings.attack[home], ratings.defence[home]),
            away_rating=(ratings.attack[away], ratings.defence[away]),
            odds_probs=implied_probs(odds) if odds else None,
        )

    def fixtures(self, refresh: bool = False, progress: Progress = None,
                 leagues: Optional[list[str]] = None
                 ) -> list[tuple[Fixture, Optional[MatchReport], str]]:
        """Upcoming fixtures with a prediction, or an error message when not possible."""
        if progress:
            progress("Downloading upcoming fixtures...")
        wanted = set(leagues or LEAGUES)
        failed: dict[str, str] = {}  # league -> reason, so we don't retry downloads per fixture
        out = []
        for fx in load_fixtures(refresh):
            if fx.league not in wanted or fx.date < date.today():
                continue
            if fx.league in failed:
                out.append((fx, None, failed[fx.league]))
                continue
            try:
                out.append((fx, self.predict(fx.league, fx.home, fx.away, fx.odds, progress), ""))
            except (KeyError, ValueError) as exc:
                out.append((fx, None, str(exc).strip("'\"")))
            except DataError:
                failed[fx.league] = "league data unavailable (offline?)"
                out.append((fx, None, failed[fx.league]))
            except Exception as exc:  # one broken league shouldn't hide the rest
                failed[fx.league] = f"{type(exc).__name__}: {exc}"
                out.append((fx, None, failed[fx.league]))
        return out

    def seasons(self, league: str) -> list[str]:
        return sorted({m.season for m in self.load(league)})

    def backtest(self, league: str, season: str, progress: Progress = None) -> BacktestResult:
        matches = self.load(league, progress=progress)
        return run_backtest(matches, season, self.settings, progress)
