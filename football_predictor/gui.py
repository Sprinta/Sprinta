"""Tkinter desktop app."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:  # pragma: no cover - depends on the Python install
    tk = None  # type: ignore[assignment]

from . import ai
from .config import CACHE_DIR, LEAGUES
from .data import current_season_start, season_code
from .engine import Engine, MatchReport

HOME_C, DRAW_C, AWAY_C = "#2563eb", "#9ca3af", "#dc2626"
FORM_C = {"W": "#16a34a", "D": "#9ca3af", "L": "#dc2626"}
FONT = "Segoe UI"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _season_label(code: str) -> str:
    return f"20{code[:2]}/{code[2:]}"


class App(tk.Tk if tk else object):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.title("Football Predictor")
        self.geometry("1180x800")
        self.minsize(1000, 700)
        self.engine = Engine()
        self.q: "queue.Queue[tuple[str, Any, Any]]" = queue.Queue()
        self.busy = False
        self.report: Optional[MatchReport] = None
        self.league_names = {f"{name} [{code}]": code for code, name in LEAGUES.items()}
        self.code_to_name = {c: n for n, c in self.league_names.items()}
        self._bar_probs = (0.0, 0.0, 0.0)

        style = ttk.Style(self)
        style.configure("Big.TLabel", font=(FONT, 16, "bold"))
        style.configure("Verdict.TLabel", font=(FONT, 13, "bold"))
        style.configure("Accent.TButton", font=(FONT, 10, "bold"))

        self._build_statusbar()
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.nb = nb
        self._build_predict_tab(nb)
        self._build_fixtures_tab(nb)
        self._build_ratings_tab(nb)
        self._build_backtest_tab(nb)
        self._build_settings_tab(nb)

        self.after(100, self._poll)
        self.after(200, self._on_league_change)

    # ---------- background work ----------

    def run_task(self, label: str, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        if self.busy:
            self.status.set("Still working on the previous task - please wait...")
            return
        self.busy = True
        self.status.set(label)
        self.progress.start(12)

        def worker() -> None:
            try:
                self.q.put(("done", done, fn()))
            except Exception as exc:  # surfaced to the user in _poll
                self.q.put(("error", None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def report_progress(self, msg: str) -> None:
        self.q.put(("status", None, msg))

    def _poll(self) -> None:
        try:
            while True:
                kind, cb, payload = self.q.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                    continue
                self.busy = False
                self.progress.stop()
                self.progress["value"] = 0
                if kind == "error":
                    self.status.set("Error")
                    messagebox.showerror("Something went wrong", str(payload) or type(payload).__name__)
                    continue
                self.status.set("Ready")
                try:
                    cb(payload)
                except Exception as exc:
                    messagebox.showerror("Display error", f"{type(exc).__name__}: {exc}")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ---------- layout ----------

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right")

    def _league_combo(self, parent: tk.Widget, default: str = "E0") -> ttk.Combobox:
        cb = ttk.Combobox(parent, values=list(self.league_names), state="readonly", width=36)
        cb.set(self.code_to_name[default])
        return cb

    def _build_predict_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Predict  ")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        right = ttk.Frame(tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        right.rowconfigure(5, weight=1)

        ctl = ttk.LabelFrame(left, text="Match", padding=10)
        ctl.pack(fill="x")
        ttk.Label(ctl, text="League").pack(anchor="w")
        self.league_cb = self._league_combo(ctl)
        self.league_cb.pack(fill="x", pady=(0, 8))
        self.league_cb.bind("<<ComboboxSelected>>", lambda e: self._on_league_change())
        ttk.Label(ctl, text="Home team").pack(anchor="w")
        self.home_cb = ttk.Combobox(ctl, state="readonly")
        self.home_cb.pack(fill="x", pady=(0, 8))
        ttk.Label(ctl, text="Away team").pack(anchor="w")
        self.away_cb = ttk.Combobox(ctl, state="readonly")
        self.away_cb.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(ctl)
        row.pack(fill="x")
        ttk.Button(row, text="Predict", style="Accent.TButton",
                   command=self._predict).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Swap", width=6, command=self._swap).pack(side="left", padx=4)
        ttk.Button(ctl, text="Refresh latest results", command=self._refresh).pack(fill="x", pady=(8, 0))

        aibox = ttk.LabelFrame(left, text="AI match preview", padding=10)
        aibox.pack(fill="x", pady=(10, 0))
        has = ai.claude_available()
        ttk.Label(aibox, wraplength=260, foreground="#555",
                  text=("Uses the Claude Code CLI with your Claude subscription. "
                        "It explains the numbers; it doesn't change them.")
                  if has else
                  ("Claude Code CLI not found. Install it from claude.ai/code and run "
                   "`claude` once to log in with your subscription.")).pack(anchor="w")
        self.ai_btn = ttk.Button(aibox, text="Ask Claude", command=self._ask_claude,
                                 state="normal" if has else "disabled")
        self.ai_btn.pack(fill="x", pady=(8, 0))

        fbox = ttk.LabelFrame(left, text="Recent form (oldest -> newest)", padding=10)
        fbox.pack(fill="x", pady=(10, 0))
        self.form_rows: list[tuple[ttk.Label, tk.Canvas, ttk.Label]] = []
        for _ in range(4):
            lbl = ttk.Label(fbox, text="", width=34)
            lbl.pack(anchor="w")
            line = ttk.Frame(fbox)
            line.pack(anchor="w", pady=(0, 6))
            cv = tk.Canvas(line, width=6 * 24, height=22, highlightthickness=0)
            cv.pack(side="left")
            extra = ttk.Label(line, text="", foreground="#555")
            extra.pack(side="left", padx=6)
            self.form_rows.append((lbl, cv, extra))
        self.ratings_lbl = ttk.Label(fbox, text="", foreground="#555", wraplength=270, justify="left")
        self.ratings_lbl.pack(anchor="w", pady=(4, 0))

        self.title_lbl = ttk.Label(right, text="Pick two teams and press Predict", style="Big.TLabel")
        self.title_lbl.grid(row=0, column=0, sticky="w")
        self.verdict_lbl = ttk.Label(right, text="", style="Verdict.TLabel")
        self.verdict_lbl.grid(row=1, column=0, sticky="w", pady=(2, 6))
        self.bar = tk.Canvas(right, height=44, highlightthickness=0, background="#f3f4f6")
        self.bar.grid(row=2, column=0, sticky="ew")
        self.bar.bind("<Configure>", lambda e: self._draw_bar())

        mid = ttk.Frame(right)
        mid.grid(row=3, column=0, sticky="ew", pady=8)
        mid.columnconfigure(1, weight=1)
        self.prob_tree = ttk.Treeview(mid, columns=("h", "d", "a"), height=4)
        self.prob_tree.heading("#0", text="Source")
        self.prob_tree.column("#0", width=200)
        for c, t in zip(("h", "d", "a"), ("Home", "Draw", "Away")):
            self.prob_tree.heading(c, text=t)
            self.prob_tree.column(c, width=80, anchor="center")
        self.prob_tree.grid(row=0, column=0, sticky="nw")
        self.stats_lbl = ttk.Label(mid, text="", justify="left", font=(FONT, 10))
        self.stats_lbl.grid(row=0, column=1, sticky="nw", padx=16)

        hbox = ttk.LabelFrame(right, text="Head-to-head", padding=6)
        hbox.grid(row=4, column=0, sticky="nsew")
        hbox.columnconfigure(0, weight=1)
        hbox.rowconfigure(1, weight=1)
        self.h2h_lbl = ttk.Label(hbox, text="")
        self.h2h_lbl.grid(row=0, column=0, sticky="w")
        self.h2h_tree = ttk.Treeview(hbox, columns=("date", "home", "score", "away"), show="headings", height=6)
        for c, t, w in (("date", "Date", 100), ("home", "Home", 180), ("score", "Score", 70), ("away", "Away", 180)):
            self.h2h_tree.heading(c, text=t)
            self.h2h_tree.column(c, width=w, anchor="center" if c in ("date", "score") else "w")
        self.h2h_tree.grid(row=1, column=0, sticky="nsew")

        abox = ttk.LabelFrame(right, text="Claude's preview", padding=6)
        abox.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        self.ai_text = ScrolledText(abox, height=7, wrap="word", font=(FONT, 10), relief="flat")
        self.ai_text.pack(fill="both", expand=True)

    def _build_fixtures_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Upcoming fixtures  ")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="League").pack(side="left")
        self.fx_cb = ttk.Combobox(top, values=["All supported leagues"] + list(self.league_names),
                                  state="readonly", width=36)
        self.fx_cb.set(self.code_to_name["E0"])
        self.fx_cb.pack(side="left", padx=6)
        ttk.Button(top, text="Load & predict", style="Accent.TButton",
                   command=self._load_fixtures).pack(side="left")
        ttk.Label(top, text="Double-click a fixture for the full analysis.",
                  foreground="#555").pack(side="left", padx=12)
        cols = ("date", "lg", "home", "away", "h", "d", "a", "pick", "book")
        self.fx_tree = ttk.Treeview(tab, columns=cols, show="headings")
        spec = (("date", "Date", 95), ("lg", "Lg", 45), ("home", "Home", 170), ("away", "Away", 170),
                ("h", "Home %", 70), ("d", "Draw %", 70), ("a", "Away %", 70),
                ("pick", "Prediction", 190), ("book", "Bookmaker H/D/A", 150))
        for c, t, w in spec:
            self.fx_tree.heading(c, text=t)
            self.fx_tree.column(c, width=w, anchor="w" if c in ("home", "away", "pick") else "center")
        self.fx_tree.pack(fill="both", expand=True, pady=(8, 0))
        self.fx_tree.bind("<Double-1>", self._open_fixture)
        self.fx_items: dict[str, tuple[str, str, str]] = {}

    def _build_ratings_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Team ratings  ")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        self.rt_cb = self._league_combo(top)
        self.rt_cb.pack(side="left")
        ttk.Button(top, text="Show", command=self._show_ratings).pack(side="left", padx=6)
        self.rt_info = ttk.Label(tab, text="Attack: goals scored vs league average (higher is better). "
                                           "Defence: goals conceded vs average (lower is better).",
                                 foreground="#555")
        self.rt_info.pack(anchor="w", pady=6)
        cols = ("rank", "team", "att", "def", "str")
        self.rt_tree = ttk.Treeview(tab, columns=cols, show="headings")
        for c, t, w in (("rank", "#", 40), ("team", "Team", 220), ("att", "Attack", 90),
                        ("def", "Defence", 90), ("str", "Strength", 90)):
            self.rt_tree.heading(c, text=t)
            self.rt_tree.column(c, width=w, anchor="w" if c == "team" else "center")
        self.rt_tree.pack(fill="both", expand=True)

    def _build_backtest_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Backtest  ")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        self.bt_cb = self._league_combo(top)
        self.bt_cb.pack(side="left")
        start = current_season_start()
        seasons = [season_code(y) for y in range(start, start - 6, -1)]
        self.bt_season = ttk.Combobox(top, values=[_season_label(s) for s in seasons], state="readonly", width=10)
        self.bt_season.set(_season_label(seasons[1]))
        self.bt_season.pack(side="left", padx=6)
        ttk.Button(top, text="Run backtest", style="Accent.TButton", command=self._run_backtest).pack(side="left")
        ttk.Label(tab, foreground="#555", wraplength=900, justify="left", text=(
            "Replays a past season week by week. Each week the model only sees matches played "
            "before it, then predicts that week. Lower Brier / log-loss is better. If the model "
            "doesn't beat 'League base rates' it has no skill; beating the bookmaker is very rare."
        )).pack(anchor="w", pady=8)
        self.bt_tree = ttk.Treeview(tab, columns=("n", "acc", "brier", "ll"), height=5)
        self.bt_tree.heading("#0", text="Method")
        self.bt_tree.column("#0", width=320)
        for c, t in (("n", "Matches"), ("acc", "Accuracy"), ("brier", "Brier"), ("ll", "Log loss")):
            self.bt_tree.heading(c, text=t)
            self.bt_tree.column(c, width=100, anchor="center")
        self.bt_tree.pack(anchor="w")
        self.bt_note = ttk.Label(tab, text="", wraplength=900, justify="left")
        self.bt_note.pack(anchor="w", pady=8)
        self.bt_apply = ttk.Button(tab, text="Use the best H2H weight", state="disabled",
                                   command=self._apply_best_weight)
        self.bt_apply.pack(anchor="w")
        self._best_weight: Optional[float] = None

    def _build_settings_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=16)
        nb.add(tab, text="  Settings  ")
        s = self.engine.settings
        self.var_seasons = tk.IntVar(value=s.seasons)
        self.var_half = tk.IntVar(value=s.half_life_days)
        self.var_prior = tk.DoubleVar(value=s.prior_strength)
        self.var_h2h = tk.DoubleVar(value=s.h2h_weight)
        self.var_h2hn = tk.IntVar(value=s.h2h_matches)
        rows = [
            ("Seasons of history", ttk.Spinbox(tab, from_=2, to=15, textvariable=self.var_seasons, width=8),
             "More seasons = longer head-to-head history. The model itself down-weights old games."),
            ("Form half-life (days)", ttk.Spinbox(tab, from_=60, to=1500, increment=30,
                                                  textvariable=self.var_half, width=8),
             "A match this many days old counts half as much as one played today."),
            ("Shrinkage (pseudo-matches)", ttk.Spinbox(tab, from_=0, to=20, increment=0.5,
                                                       textvariable=self.var_prior, width=8),
             "Pulls teams with few games toward average. Stops early-season overreaction."),
            ("Head-to-head matches", ttk.Spinbox(tab, from_=1, to=30, textvariable=self.var_h2hn, width=8),
             "How many past meetings to look at."),
        ]
        for i, (label, widget, help_) in enumerate(rows):
            ttk.Label(tab, text=label).grid(row=i, column=0, sticky="w", pady=6)
            widget.grid(row=i, column=1, sticky="w", padx=10)
            ttk.Label(tab, text=help_, foreground="#555").grid(row=i, column=2, sticky="w")
        i = len(rows)
        ttk.Label(tab, text="Head-to-head weight").grid(row=i, column=0, sticky="w", pady=6)
        wf = ttk.Frame(tab)
        wf.grid(row=i, column=1, sticky="w", padx=10)
        self.h2h_val = ttk.Label(wf, text=f"{s.h2h_weight:.2f}", width=5)
        ttk.Scale(wf, from_=0, to=0.6, variable=self.var_h2h, length=140,
                  command=lambda v: self.h2h_val.config(text=f"{float(v):.2f}")).pack(side="left")
        self.h2h_val.pack(side="left", padx=4)
        ttk.Label(tab, foreground="#555", text=(
            "0 = past results only. Use the Backtest tab to find what actually helps; "
            "above ~0.3 usually hurts.")).grid(row=i, column=2, sticky="w")
        ttk.Button(tab, text="Save settings", style="Accent.TButton",
                   command=self._save_settings).grid(row=i + 1, column=0, sticky="w", pady=16)
        ttk.Label(tab, text=f"Data cache: {CACHE_DIR}", foreground="#555").grid(
            row=i + 2, column=0, columnspan=3, sticky="w")
        ttk.Label(tab, text="Data: football-data.co.uk (free historical results and odds).",
                  foreground="#555").grid(row=i + 3, column=0, columnspan=3, sticky="w")

    # ---------- predict tab ----------

    def _league(self, cb: ttk.Combobox) -> str:
        return self.league_names[cb.get()]

    def _on_league_change(self) -> None:
        league = self._league(self.league_cb)

        def load() -> list[str]:
            self.engine.load(league, progress=self.report_progress)
            return self.engine.teams(league)

        def done(teams: list[str]) -> None:
            self.home_cb["values"] = teams
            self.away_cb["values"] = teams
            if teams:
                self.home_cb.set(teams[0])
                self.away_cb.set(teams[1] if len(teams) > 1 else teams[0])
            self.status.set(f"{len(teams)} teams loaded for {LEAGUES[league]}")

        self.run_task(f"Loading {LEAGUES[league]}...", load, done)

    def _refresh(self) -> None:
        league = self._league(self.league_cb)
        self.run_task("Refreshing...", lambda: self.engine.load(league, refresh=True,
                                                                  progress=self.report_progress),
                      lambda _: self.status.set("Latest results downloaded"))

    def _swap(self) -> None:
        h, a = self.home_cb.get(), self.away_cb.get()
        self.home_cb.set(a)
        self.away_cb.set(h)

    def _predict(self, odds: Optional[tuple[float, float, float]] = None) -> None:
        league, home, away = self._league(self.league_cb), self.home_cb.get(), self.away_cb.get()
        if not home or not away:
            messagebox.showinfo("Pick teams", "Choose a home and an away team first.")
            return
        if home == away:
            messagebox.showinfo("Pick teams", "Home and away team must be different.")
            return
        self.run_task(f"Predicting {home} vs {away}...",
                      lambda: self.engine.predict(league, home, away, odds, self.report_progress),
                      self._show_report)

    def _show_report(self, r: MatchReport) -> None:
        self.report = r
        p = r.prediction
        self.title_lbl.config(text=f"{r.home}  vs  {r.away}")
        self.verdict_lbl.config(text=f"Prediction: {r.verdict}  ({_pct(max(r.final_probs))})")
        self._bar_probs = r.final_probs
        self._draw_bar()

        self.prob_tree.delete(*self.prob_tree.get_children())
        rows = [("Model (past results)", r.model_probs),
                (f"Head-to-head ({len(r.h2h.matches)} games)", r.h2h_probs),
                (f"FINAL (H2H weight {r.h2h_weight:.2f})", r.final_probs)]
        if r.odds_probs:
            rows.append(("Bookmaker (for comparison)", r.odds_probs))
        for name, probs in rows:
            self.prob_tree.insert("", "end", text=name, values=tuple(_pct(x) for x in probs))

        scores = "\n".join(f"   {i}-{j}   {_pct(q)}" for i, j, q in p.top_scores)
        self.stats_lbl.config(text=(
            f"Expected goals:  {p.home_xg:.2f} - {p.away_xg:.2f}\n"
            f"Over 2.5 goals:  {_pct(p.over_2_5)}\n"
            f"Both teams score:  {_pct(p.btts)}\n"
            f"Most likely scores:\n{scores}"))

        forms = [(f"{r.home} - last {len(r.home_form.results)}", r.home_form),
                 (f"{r.home} - at home", r.home_form_at_home),
                 (f"{r.away} - last {len(r.away_form.results)}", r.away_form),
                 (f"{r.away} - away", r.away_form_away)]
        for (lbl, cv, extra), (title, f) in zip(self.form_rows, forms):
            lbl.config(text=title)
            cv.delete("all")
            for k, ch in enumerate(f.results):
                x = k * 24
                cv.create_rectangle(x + 1, 1, x + 21, 21, fill=FORM_C[ch], outline="")
                cv.create_text(x + 11, 11, text=ch, fill="white", font=(FONT, 9, "bold"))
            extra.config(text=f"{f.points} pts, {f.goals_for}-{f.goals_against}")
        (ha, hd), (aa, ad) = r.home_rating, r.away_rating
        self.ratings_lbl.config(text=(
            f"Ratings (1.00 = league average)\n"
            f"{r.home}: attack {ha:.2f}, defence {hd:.2f}\n"
            f"{r.away}: attack {aa:.2f}, defence {ad:.2f}"))

        h = r.h2h
        self.h2h_lbl.config(text=(
            f"{r.home} {h.home_wins} wins  -  {h.draws} draws  -  {h.away_wins} wins {r.away}"
            f"     (goals {h.home_goals}-{h.away_goals})" if h.matches else
            "No previous meetings in the loaded seasons."))
        self.h2h_tree.delete(*self.h2h_tree.get_children())
        for m in h.matches:
            self.h2h_tree.insert("", "end", values=(m.date.strftime("%d %b %Y"), m.home,
                                                    f"{m.home_goals} - {m.away_goals}", m.away))
        self.ai_text.delete("1.0", "end")

    def _draw_bar(self) -> None:
        cv = self.bar
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if sum(self._bar_probs) <= 0 or w < 10:
            return
        x = 0.0
        for p, color in zip(self._bar_probs, (HOME_C, DRAW_C, AWAY_C)):
            seg = p * w
            cv.create_rectangle(x, 0, x + seg, h, fill=color, outline="")
            if seg > 50:
                cv.create_text(x + seg / 2, h / 2, text=_pct(p), fill="white", font=(FONT, 11, "bold"))
            x += seg

    def _ask_claude(self) -> None:
        if not self.report:
            messagebox.showinfo("Predict first", "Run a prediction first, then ask Claude to explain it.")
            return
        report = self.report
        self.ai_text.delete("1.0", "end")
        self.ai_text.insert("1.0", "Asking Claude...")

        def done(text: str) -> None:
            self.ai_text.delete("1.0", "end")
            self.ai_text.insert("1.0", text)

        self.run_task("Asking Claude (via your subscription)...", lambda: ai.ask_claude(report), done)

    # ---------- other tabs ----------

    def _load_fixtures(self) -> None:
        choice = self.fx_cb.get()
        leagues = None if choice.startswith("All") else [self.league_names[choice]]

        def done(rows: list) -> None:
            self.fx_tree.delete(*self.fx_tree.get_children())
            self.fx_items.clear()
            for fx, r, err in rows:
                book = "/".join(f"{o:.2f}" for o in fx.odds) if fx.odds else ""
                if r:
                    vals = (fx.date.strftime("%a %d %b"), fx.league, fx.home, fx.away,
                            *(f"{x * 100:.0f}%" for x in r.final_probs), r.verdict, book)
                else:
                    vals = (fx.date.strftime("%a %d %b"), fx.league, fx.home, fx.away,
                            "", "", "", f"n/a ({err})", book)
                iid = self.fx_tree.insert("", "end", values=vals)
                self.fx_items[iid] = (fx.league, fx.home, fx.away)
            self.status.set(f"{len(rows)} upcoming fixtures")

        self.run_task("Loading fixtures...",
                      lambda: self.engine.fixtures(progress=self.report_progress, leagues=leagues), done)

    def _open_fixture(self, _event: Any) -> None:
        sel = self.fx_tree.selection()
        if not sel or sel[0] not in self.fx_items:
            return
        league, home, away = self.fx_items[sel[0]]
        self.league_cb.set(self.code_to_name[league])
        teams = self.engine.teams(league)
        self.home_cb["values"] = teams
        self.away_cb["values"] = teams
        self.home_cb.set(home)
        self.away_cb.set(away)
        self.nb.select(0)
        self._predict()

    def _show_ratings(self) -> None:
        league = self._league(self.rt_cb)

        def work() -> tuple:
            self.engine.load(league, progress=self.report_progress)
            return self.engine.table(league), self.engine.ratings(league)

        def done(res: tuple) -> None:
            table, r = res
            self.rt_tree.delete(*self.rt_tree.get_children())
            for i, (t, a, d, s) in enumerate(table, 1):
                self.rt_tree.insert("", "end", values=(i, t, f"{a:.2f}", f"{d:.2f}", f"{s:.2f}"))
            self.rt_info.config(text=(
                f"Home advantage x{r.home_adv:.2f}   |   avg away-team goals {r.base:.2f}   |   "
                f"low-score correction rho {r.rho:+.2f}   |   {r.matches_used} matches. "
                f"Attack higher = better, defence lower = better."))

        self.run_task("Fitting ratings...", work, done)

    def _run_backtest(self) -> None:
        league = self._league(self.bt_cb)
        label = self.bt_season.get()
        season = label[2:4] + label[5:7]

        def done(res: Any) -> None:
            self.bt_tree.delete(*self.bt_tree.get_children())
            for s in res.scores:
                vals = (s.n, "-", "-", "-") if s.n == 0 else \
                    (s.n, f"{s.accuracy * 100:.1f}%", f"{s.brier:.3f}", f"{s.log_loss:.3f}")
                self.bt_tree.insert("", "end", text=s.name, values=vals)
            model, blended, book, base = res.scores
            lines = []
            if model.n:
                skill = "has real skill" if model.log_loss < base.log_loss else "does NOT beat simple base rates"
                lines.append(f"The model {skill} on this season.")
            if book.n and model.n:
                gap = model.log_loss - book.log_loss
                lines.append("The model beat the bookmaker. Treat this with suspicion; it rarely holds up."
                             if gap < 0 else
                             f"The bookmaker was better by {gap:.3f} log-loss. That's normal for public-data models.")
            lines.append(f"Best head-to-head weight for this season: {res.best_h2h_weight:.2f} "
                         f"(current {self.engine.settings.h2h_weight:.2f}). Check a few seasons before trusting it.")
            if res.skipped:
                lines.append(f"{res.skipped} early matches skipped (not enough prior data).")
            self.bt_note.config(text="\n".join(lines))
            self._best_weight = res.best_h2h_weight
            self.bt_apply.config(state="normal")

        self.run_task(f"Backtesting {label}...",
                      lambda: self.engine.backtest(league, season, self.report_progress), done)

    def _apply_best_weight(self) -> None:
        if self._best_weight is None:
            return
        self.var_h2h.set(self._best_weight)
        self.h2h_val.config(text=f"{self._best_weight:.2f}")
        self._save_settings()

    def _save_settings(self) -> None:
        s = self.engine.settings
        try:
            seasons = int(self.var_seasons.get())
            s.half_life_days = max(30, int(self.var_half.get()))
            s.prior_strength = max(0.0, float(self.var_prior.get()))
            s.h2h_matches = max(1, int(self.var_h2hn.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("Settings", "Please enter valid numbers.")
            return
        s.h2h_weight = round(float(self.var_h2h.get()), 2)
        s.seasons = max(2, min(seasons, 20))
        s.save()
        self.engine.invalidate()
        self.status.set("Settings saved. Models will refit on next use.")
        self._on_league_change()


def main() -> int:
    if tk is None:
        print("tkinter is not available in this Python install.\n"
              "On Windows, reinstall Python from python.org and tick 'tcl/tk and IDLE'.\n"
              "You can still use the command line: python -m football_predictor --help")
        return 1
    App().mainloop()
    return 0
