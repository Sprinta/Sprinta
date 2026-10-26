"""Command-line interface. Run `python -m football_predictor --help`."""

from __future__ import annotations

import argparse
import difflib
import sys

from .config import LEAGUES
from .data import DataError
from .engine import Engine, MatchReport


def _progress(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr)


def resolve_team(engine: Engine, league: str, name: str) -> str:
    teams = engine.teams(league)
    lower = {t.lower(): t for t in teams}
    if name.lower() in lower:
        return lower[name.lower()]
    partial = [t for t in teams if name.lower() in t.lower()]
    if len(partial) == 1:
        return partial[0]
    hint = partial or difflib.get_close_matches(name, teams, n=5, cutoff=0.4)
    raise SystemExit(f"Unknown team '{name}'. Did you mean: {', '.join(hint) or 'see `teams` command'}?")


def print_report(r: MatchReport) -> None:
    p = r.prediction
    pct = lambda x: f"{x * 100:5.1f}%"
    print(f"\n{r.home}  vs  {r.away}   ({LEAGUES.get(r.league, r.league)})")
    print("-" * 60)
    print(f"{'':12}{'Home':>9}{'Draw':>9}{'Away':>9}")
    print(f"{'Model':12}{pct(r.model_probs[0]):>9}{pct(r.model_probs[1]):>9}{pct(r.model_probs[2]):>9}")
    print(f"{'H2H':12}{pct(r.h2h_probs[0]):>9}{pct(r.h2h_probs[1]):>9}{pct(r.h2h_probs[2]):>9}")
    print(f"{'FINAL':12}{pct(r.final_probs[0]):>9}{pct(r.final_probs[1]):>9}{pct(r.final_probs[2]):>9}")
    if r.odds_probs:
        print(f"{'Bookmaker':12}{pct(r.odds_probs[0]):>9}{pct(r.odds_probs[1]):>9}{pct(r.odds_probs[2]):>9}")
    print(f"\nPrediction: {r.verdict}")
    print(f"Expected goals: {p.home_xg:.2f} - {p.away_xg:.2f}   "
          f"Over 2.5: {pct(p.over_2_5)}   BTTS: {pct(p.btts)}")
    print("Likely scores: " + ", ".join(f"{i}-{j} ({q * 100:.0f}%)" for i, j, q in p.top_scores))
    print(f"\nForm (oldest -> newest)  {r.home}: {r.home_form.results or '-'}   "
          f"{r.away}: {r.away_form.results or '-'}")
    h = r.h2h
    print(f"Head-to-head (last {len(h.matches)}): {r.home} {h.home_wins}W  {h.draws}D  {h.away_wins}W {r.away}")
    for m in h.matches:
        print(f"  {m.date:%d %b %Y}  {m.home} {m.home_goals}-{m.away_goals} {m.away}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="football_predictor",
                                 description="Football match predictor (past results + head-to-head).")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the desktop app (default)")
    sub.add_parser("leagues", help="list supported leagues")
    p = sub.add_parser("teams", help="list teams in a league"); p.add_argument("league")
    p = sub.add_parser("table", help="model strength table"); p.add_argument("league")
    p = sub.add_parser("predict", help="predict one match")
    p.add_argument("league"); p.add_argument("home"); p.add_argument("away")
    p.add_argument("--ai", action="store_true", help="add a Claude-written preview (uses your Claude subscription)")
    p = sub.add_parser("fixtures", help="predict upcoming fixtures")
    p.add_argument("--league", action="append", help="limit to league code(s)")
    p.add_argument("--refresh", action="store_true")
    p = sub.add_parser("backtest", help="measure accuracy on a past season")
    p.add_argument("league"); p.add_argument("--season", help="e.g. 2425 (default: last completed)")
    p = sub.add_parser("refresh", help="re-download the current season"); p.add_argument("league")
    args = ap.parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui import main as gui_main
        return gui_main()

    if args.cmd == "leagues":
        for code, name in LEAGUES.items():
            print(f"{code:5} {name}")
        return 0

    league = getattr(args, "league", None)
    if isinstance(league, str):
        league = league.upper()
        if league not in LEAGUES:
            raise SystemExit(f"Unknown league '{league}'. Run `leagues` to see codes.")

    engine = Engine()
    try:
        if args.cmd == "refresh":
            engine.load(league, refresh=True, progress=_progress)
            print("Up to date.")
        elif args.cmd == "teams":
            engine.load(league, progress=_progress)
            print("\n".join(engine.teams(league)))
        elif args.cmd == "table":
            engine.load(league, progress=_progress)
            print(f"{'#':>3} {'Team':25}{'Attack':>8}{'Defence':>9}{'Strength':>10}")
            for i, (t, a, d, s) in enumerate(engine.table(league), 1):
                print(f"{i:3} {t:25}{a:8.2f}{d:9.2f}{s:10.2f}")
        elif args.cmd == "predict":
            engine.load(league, progress=_progress)
            home, away = resolve_team(engine, league, args.home), resolve_team(engine, league, args.away)
            report = engine.predict(league, home, away)
            print_report(report)
            if args.ai:
                from .ai import ask_claude
                print("\nClaude's preview:\n" + ask_claude(report))
        elif args.cmd == "fixtures":
            leagues = [l.upper() for l in args.league] if args.league else None
            for fx, r, err in engine.fixtures(args.refresh, _progress, leagues):
                head = f"{fx.date:%a %d %b}  {fx.league:4} {fx.home} vs {fx.away}"
                if r:
                    h, d, a = (f"{x * 100:.0f}%" for x in r.final_probs)
                    print(f"{head:60} H {h:>4}  D {d:>4}  A {a:>4}  -> {r.verdict}")
                else:
                    print(f"{head:60} (no prediction: {err})")
        elif args.cmd == "backtest":
            seasons = engine.seasons(league)
            season = args.season or (seasons[-2] if len(seasons) > 1 else seasons[-1])
            res = engine.backtest(league, season, _progress)
            print(f"\nBacktest {league} season {season[:2]}/{season[2:]}  (skipped {res.skipped})")
            print(f"{'Method':40}{'N':>6}{'Accuracy':>10}{'Brier':>8}{'LogLoss':>9}")
            for s in res.scores:
                print(f"{s.name:40}{s.n:6}{s.accuracy * 100:9.1f}%{s.brier:8.3f}{s.log_loss:9.3f}")
            print(f"\nBest H2H weight on this season: {res.best_h2h_weight:.2f} "
                  f"(current setting {engine.settings.h2h_weight:.2f})")
    except DataError as exc:
        raise SystemExit(f"Data error: {exc}")
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0
