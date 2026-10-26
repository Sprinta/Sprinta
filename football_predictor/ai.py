"""Optional AI match preview through the Claude Code CLI.

This runs `claude -p` using your logged-in Claude subscription. There's no
API key and no per-call billing. ANTHROPIC_API_KEY is removed from the child
environment so the CLI can't silently fall back to paid API usage.

The numbers always come from the statistical model. Claude only writes
the explanation; it is never asked to override the probabilities.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from .config import LEAGUES
from .engine import MatchReport

INSTRUCTION = (
    "You are a football analyst. Using ONLY the statistics provided on stdin, "
    "write a concise match preview (max 200 words): key factors, how recent form "
    "and head-to-head compare with the model, and the main risk to the prediction. "
    "Do not invent injuries, news or facts not in the data. Do not change the "
    "probabilities; explain them."
)


def claude_path() -> Optional[str]:
    return shutil.which("claude")


def claude_available() -> bool:
    return claude_path() is not None


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def build_summary(r: MatchReport) -> str:
    p = r.prediction
    lines = [
        f"Match: {r.home} (home) vs {r.away} (away), {LEAGUES.get(r.league, r.league)}",
        f"Final probabilities: home {_pct(r.final_probs[0])}, draw {_pct(r.final_probs[1])}, "
        f"away {_pct(r.final_probs[2])}",
        f"Model-only probabilities: {', '.join(_pct(x) for x in r.model_probs)}",
        f"H2H-implied probabilities (weight {r.h2h_weight:.2f}): {', '.join(_pct(x) for x in r.h2h_probs)}",
        f"Expected goals: {r.home} {p.home_xg:.2f}, {r.away} {p.away_xg:.2f}",
        f"Over 2.5 goals: {_pct(p.over_2_5)}; both teams score: {_pct(p.btts)}",
        "Most likely scores: " + ", ".join(f"{i}-{j} ({_pct(q)})" for i, j, q in p.top_scores),
        f"Ratings (attack, defence; 1.00 = league average, lower defence is better): "
        f"{r.home} {r.home_rating[0]:.2f}/{r.home_rating[1]:.2f}, "
        f"{r.away} {r.away_rating[0]:.2f}/{r.away_rating[1]:.2f}",
        f"{r.home} last {len(r.home_form.results)}: {r.home_form.results} "
        f"(GF {r.home_form.goals_for}, GA {r.home_form.goals_against}); at home: {r.home_form_at_home.results}",
        f"{r.away} last {len(r.away_form.results)}: {r.away_form.results} "
        f"(GF {r.away_form.goals_for}, GA {r.away_form.goals_against}); away: {r.away_form_away.results}",
        f"Head-to-head last {len(r.h2h.matches)}: {r.home} {r.h2h.home_wins}W, "
        f"{r.h2h.draws}D, {r.away} {r.h2h.away_wins}W",
    ]
    for m in r.h2h.matches:
        lines.append(f"  {m.date:%Y-%m-%d} {m.home} {m.home_goals}-{m.away_goals} {m.away}")
    if r.odds_probs:
        lines.append(f"Bookmaker implied: {', '.join(_pct(x) for x in r.odds_probs)}")
    return "\n".join(lines)


def ask_claude(report: MatchReport, timeout: int = 180) -> str:
    path = claude_path()
    if not path:
        raise RuntimeError(
            "Claude Code CLI not found. Install it (https://claude.ai/code), run "
            "`claude` once to log in with your Claude subscription, then restart this app."
        )
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(
        [path, "-p", INSTRUCTION],
        input=build_summary(report),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "claude exited with an error").strip())
    return proc.stdout.strip()
