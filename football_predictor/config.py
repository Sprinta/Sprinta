"""Leagues, paths and user settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Division codes used by football-data.co.uk.
LEAGUES: dict[str, str] = {
    "E0": "England - Premier League",
    "E1": "England - Championship",
    "E2": "England - League One",
    "E3": "England - League Two",
    "SC0": "Scotland - Premiership",
    "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga",
    "SP1": "Spain - La Liga",
    "SP2": "Spain - Segunda Division",
    "I1": "Italy - Serie A",
    "I2": "Italy - Serie B",
    "F1": "France - Ligue 1",
    "F2": "France - Ligue 2",
    "N1": "Netherlands - Eredivisie",
    "B1": "Belgium - Pro League",
    "P1": "Portugal - Primeira Liga",
    "T1": "Turkey - Super Lig",
    "G1": "Greece - Super League",
}

DATA_DIR = Path.home() / ".football_predictor"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_FILE = DATA_DIR / "settings.json"


@dataclass
class Settings:
    seasons: int = 5               # seasons of history to download per league
    half_life_days: int = 300      # older matches count half as much after this many days
    prior_strength: float = 3.0    # pseudo-matches pulling thin-data teams to average
    h2h_weight: float = 0.15       # 0 = ignore head-to-head, 1 = trust it fully
    h2h_matches: int = 10          # most recent meetings considered

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in raw.items() if k in known})
        except TypeError:
            return cls()

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
