- 👋 Hi, I’m @Sprinta
- 🌱 I’m currently learning data analytics

# Football Predictor

A desktop app (Windows, macOS, Linux) that predicts football matches from past results and head-to-head records.
It's pure Python with **no packages to install**.

## Quick start (Windows)

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/). In the installer, tick **"Add python.exe to PATH"**. Keep **"tcl/tk and IDLE"** ticked (it's on by default).
2. Download this repo (green **Code** button → *Download ZIP*) and unzip it.
3. Double-click **`run_windows.bat`**.

The first time you open a league, the app downloads about 5 seasons of results from [football-data.co.uk](https://www.football-data.co.uk) (free). After that it works from a local cache in `%USERPROFILE%\.football_predictor`.

## What it does

| Tab | What you get |
|---|---|
| **Predict** | Home/draw/away %, expected goals, over 2.5, both teams to score, most likely scores, last-6 form (overall + home/away), head-to-head list, team ratings |
| **Upcoming fixtures** | Predictions for every scheduled match in a league, next to the bookmaker odds. Double-click a match for the full analysis |
| **Team ratings** | Attack/defence strength of every team |
| **Backtest** | Replays a past season week by week and scores the model against the bookmaker and a no-skill baseline. This is how you find out whether it actually works |
| **Settings** | History length, how fast old results fade, and how much weight head-to-head gets |

18 leagues: England (top 4 tiers), Scotland, Germany (2), Spain (2), Italy (2), France (2), Netherlands, Belgium, Portugal, Turkey, Greece.

## How the prediction is made

1. **Team strength from past results (the core).** This is a Dixon-Coles Poisson model. Each team gets an attack rating and a defence rating, fitted to every result in the loaded seasons. Recent games count more (form half-life, 300 days by default). The model also includes home advantage and a correction for low-scoring draws. Promoted teams start below average, and teams with few games are pulled toward average until they've played more. The fitted ratings give each team's expected goals, which turn into a full scoreline probability grid.
2. **Head-to-head (an adjustment).** The last 10 meetings are converted into home/draw/away rates. Recent meetings count more, and meetings at the other venue count half. The H2H estimate is smoothed toward the model, so 2–3 meetings can't swing things on their own.
3. **Final = (1 − w) × model + w × H2H**, with *w* = 0.15 by default.

### Why H2H only gets 15%

Most of what a head-to-head record "knows" is already in the teams' current strength. A 4-1 from three seasons ago tells you little about this weekend once squads and managers have changed. Weighting it heavily usually makes predictions worse. Don't take that on faith: run the **Backtest** tab. It shows the best H2H weight for each season, and **Use the best H2H weight** applies it. Check a few seasons before trusting any one number.

### What to expect

- A good model picks the right result in roughly **50–54%** of matches. Draws are almost never the favourite, so the model rarely "predicts" one.
- Bookmaker closing odds are the benchmark. A model built only on public results rarely beats them over a full season. If a backtest says yours does, suspect luck first.
- The model can't see injuries, suspensions, transfers or a new manager.

## Optional: AI match preview using your Claude subscription

The **Ask Claude** button sends the match statistics to the [Claude Code](https://claude.ai/code) CLI (`claude -p`), which writes a short preview. It runs on your **Claude Pro/Max subscription login**. No API key is needed, and `ANTHROPIC_API_KEY` is removed from the environment so it can't bill the API by mistake.

Setup: install Claude Code and run `claude` once in a terminal to log in. Then restart the app.

Claude only explains the numbers. It never changes the probabilities, and it's told not to invent news.

## Command line

```
python -m football_predictor                         # opens the app
python -m football_predictor leagues
python -m football_predictor predict E0 Arsenal Chelsea
python -m football_predictor predict E0 Arsenal Chelsea --ai
python -m football_predictor fixtures --league E0
python -m football_predictor table E0
python -m football_predictor backtest E0 --season 2425
```

Team names can be partial (`"man u"` → Man United).

## Development

```
python -m unittest        # tests run on synthetic leagues with known true strengths
```

Code layout: `data.py` (download/cache/parse), `model.py` (Dixon-Coles), `analysis.py` (form, H2H, blending), `backtest.py`, `engine.py` (glue), `gui.py` (Tkinter), `cli.py`, `ai.py` (Claude CLI).
