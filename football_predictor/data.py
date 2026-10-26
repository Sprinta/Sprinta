"""Download, cache and parse historical results from football-data.co.uk."""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

from .config import CACHE_DIR

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
USER_AGENT = "Mozilla/5.0 (FootballPredictor/1.0)"

# Closing-odds columns in order of preference (market average first).
ODDS_COLUMNS = [
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
]

Odds = Optional[tuple[float, float, float]]


class DataError(Exception):
    pass


@dataclass(frozen=True)
class Match:
    date: date
    league: str
    season: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    odds: Odds = None

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


@dataclass(frozen=True)
class Fixture:
    date: date
    league: str
    home: str
    away: str
    odds: Odds = None


def season_code(start_year: int) -> str:
    """2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: Optional[date] = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def recent_seasons(n: int, today: Optional[date] = None) -> list[str]:
    """Season codes, oldest first, ending with the current season."""
    start = current_season_start(today)
    return [season_code(y) for y in range(start - n + 1, start + 1)]


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _fetch(url: str, cache_path: Path, max_age: Optional[float]) -> str:
    """Return the file at url, served from cache when fresh enough.

    max_age=None means the cached copy never expires (finished seasons).
    If the download fails, a stale cached copy is used rather than failing.
    """
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if max_age is None or age < max_age:
            return _decode(cache_path.read_bytes())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if cache_path.exists():
            return _decode(cache_path.read_bytes())
        raise DataError(f"Could not download {url}: {exc}") from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return _decode(raw)


def _parse_date(text: str) -> Optional[date]:
    text = text.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_odds(row: dict) -> Odds:
    for cols in ODDS_COLUMNS:
        try:
            odds = tuple(float(row[c]) for c in cols)
        except (KeyError, TypeError, ValueError):
            continue
        if all(o > 1.0 for o in odds):
            return odds  # type: ignore[return-value]
    return None


def _first(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return ""


def parse_matches(text: str, league: str, season: str) -> list[Match]:
    matches = []
    for row in csv.DictReader(io.StringIO(text)):
        d = _parse_date(row.get("Date") or "")
        home = (row.get("HomeTeam") or row.get("Home") or "").strip()
        away = (row.get("AwayTeam") or row.get("Away") or "").strip()
        try:
            hg = int(float(_first(row, "FTHG", "HG")))
            ag = int(float(_first(row, "FTAG", "AG")))
        except ValueError:
            continue  # unplayed or malformed row
        if d is None or not home or not away:
            continue
        matches.append(Match(d, league, season, home, away, hg, ag, _parse_odds(row)))
    return matches


def parse_fixtures(text: str) -> list[Fixture]:
    fixtures = []
    for row in csv.DictReader(io.StringIO(text)):
        d = _parse_date(row.get("Date") or "")
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        league = (row.get("Div") or "").strip()
        if d and home and away and league:
            fixtures.append(Fixture(d, league, home, away, _parse_odds(row)))
    return fixtures


def load_league(
    league: str,
    seasons: int,
    refresh: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Match]:
    """All results for a league over the last `seasons` seasons, oldest first."""
    codes = recent_seasons(seasons)
    current = codes[-1]
    matches: list[Match] = []
    errors = []
    for code in codes:
        if progress:
            progress(f"Loading {league} {code[:2]}/{code[2:]}...")
        max_age = (0 if refresh else 6 * 3600) if code == current else None
        try:
            text = _fetch(SEASON_URL.format(season=code, league=league),
                          CACHE_DIR / f"{league}_{code}.csv", max_age)
        except DataError as exc:
            errors.append(str(exc))
            continue
        matches.extend(parse_matches(text, league, code))
    if not matches:
        raise DataError("No data available.\n" + "\n".join(errors))
    matches.sort(key=lambda m: m.date)
    return matches


def load_fixtures(refresh: bool = False) -> list[Fixture]:
    text = _fetch(FIXTURES_URL, CACHE_DIR / "fixtures.csv", 0 if refresh else 3 * 3600)
    return sorted(parse_fixtures(text), key=lambda f: f.date)
