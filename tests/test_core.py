"""Tests on synthetic leagues with known true strengths. Run: python -m unittest"""

import math
import random
import unittest
from datetime import date, timedelta
from unittest import mock

from football_predictor import data, engine
from football_predictor.analysis import blend_h2h, form, h2h_probs, head_to_head, implied_probs
from football_predictor.backtest import run_backtest
from football_predictor.config import Settings
from football_predictor.data import Match, parse_fixtures, parse_matches
from football_predictor.model import fit, predict, promoted_teams


def _poisson(rng: random.Random, lam: float) -> int:
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p < l:
            return k
        k += 1


def synthetic_league(seasons=3, n_teams=12, seed=1):
    rng = random.Random(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    # Team A strongest ... last team weakest.
    att = {t: 1.6 - 1.0 * i / (n_teams - 1) for i, t in enumerate(teams)}
    dfn = {t: 0.6 + 0.9 * i / (n_teams - 1) for i, t in enumerate(teams)}
    matches = []
    for s in range(seasons):
        d = date(2022 + s, 8, 6)
        code = data.season_code(2022 + s)
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for k, (h, a) in enumerate(fixtures):
            day = d + timedelta(days=7 * (k // (n_teams // 2)))
            lam, mu = 1.1 * 1.3 * att[h] * dfn[a], 1.1 * att[a] * dfn[h]
            matches.append(Match(day, "E0", code, h, a, _poisson(rng, lam), _poisson(rng, mu),
                                 (2.5, 3.3, 3.0)))
    matches.sort(key=lambda m: m.date)
    return matches, att, dfn


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches, cls.att, cls.dfn = synthetic_league()
        cls.ratings = fit(cls.matches, half_life_days=10_000, prior_strength=1)

    def test_recovers_team_order(self):
        strength = {t: self.ratings.attack[t] / self.ratings.defence[t] for t in self.att}
        ranked = sorted(strength, key=strength.get, reverse=True)
        # 396 matches is a small sample, so check broad agreement, not exact order.
        self.assertIn("Team A", ranked[:3])
        self.assertIn("Team L", ranked[-3:])
        true_rank = list(self.att)  # already strongest -> weakest
        d2 = sum((ranked.index(t) - true_rank.index(t)) ** 2 for t in true_rank)
        n = len(true_rank)
        spearman = 1 - 6 * d2 / (n * (n * n - 1))
        self.assertGreater(spearman, 0.85)
        self.assertGreater(self.ratings.home_adv, 1.1)
        self.assertLess(self.ratings.home_adv, 1.55)

    def test_probabilities_are_consistent(self):
        p = predict(self.ratings, "Team A", "Team L")
        self.assertAlmostEqual(p.p_home + p.p_draw + p.p_away, 1.0, places=9)
        self.assertGreater(p.p_home, 0.6)
        self.assertTrue(0 < p.over_2_5 < 1 and 0 < p.btts < 1)
        self.assertEqual(len(p.top_scores), 5)
        rev = predict(self.ratings, "Team L", "Team A")
        self.assertGreater(rev.p_away, rev.p_home)

    def test_unknown_team(self):
        with self.assertRaises(KeyError):
            predict(self.ratings, "Team A", "Nobody FC")

    def test_as_of_excludes_future(self):
        cutoff = self.matches[len(self.matches) // 2].date
        r = fit(self.matches, as_of=cutoff)
        self.assertEqual(r.matches_used, sum(m.date < cutoff for m in self.matches))

    def test_promoted_detection_and_prior(self):
        ms = list(self.matches)
        last = ms[-1].season
        ms += [Match(date(2024, 9, 1), "E0", last, "New FC", "Team A", 0, 2),
               Match(date(2024, 9, 8), "E0", last, "Team B", "New FC", 1, 1)]
        self.assertEqual(promoted_teams(ms), {"New FC"})
        r = fit(ms, promoted={"New FC"})
        self.assertLess(r.attack["New FC"], 1.0)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        d = date(2024, 1, 1)
        self.ms = [
            Match(d, "E0", "2324", "X", "Y", 2, 0),
            Match(d + timedelta(7), "E0", "2324", "Y", "X", 1, 1),
            Match(d + timedelta(14), "E0", "2324", "X", "Z", 0, 3),
            Match(d + timedelta(21), "E0", "2324", "X", "Y", 3, 1),
        ]

    def test_form(self):
        f = form(self.ms, "X")
        self.assertEqual(f.results, "WDLW")
        self.assertEqual((f.points, f.goals_for, f.goals_against), (7, 6, 5))
        self.assertEqual(form(self.ms, "X", venue="away").results, "D")

    def test_head_to_head_oriented_to_home_team(self):
        h = head_to_head(self.ms, "Y", "X")
        self.assertEqual(len(h.matches), 3)
        self.assertEqual((h.home_wins, h.draws, h.away_wins), (0, 1, 2))
        self.assertEqual(h.matches[0].date, self.ms[-1].date)  # most recent first

    def test_h2h_probs_shrink_toward_model(self):
        model = (0.4, 0.3, 0.3)
        h = head_to_head(self.ms, "X", "Y")
        hp = h2h_probs(h, model, date(2024, 2, 1))
        self.assertAlmostEqual(sum(hp), 1.0)
        self.assertGreater(hp[0], model[0])  # X won the meetings
        empty = h2h_probs(head_to_head(self.ms, "X", "Q"), model, date(2024, 2, 1))
        self.assertEqual(tuple(round(x, 9) for x in empty), model)
        self.assertEqual(blend_h2h(model, hp, 0), model)

    def test_implied_probs_remove_margin(self):
        p = implied_probs((2.0, 3.4, 3.8))
        self.assertAlmostEqual(sum(p), 1.0)


class DataTests(unittest.TestCase):
    CSV = ("﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
           "E0,16/08/2025,20:00,Team A,Team B,2,1,H,1.9,3.5,4.2,1.88,3.45,4.1\n"
           "E0,17/08/25,15:00,Team C,Team D,0,0,D,2.5,3.2,2.9,,,\n"
           "E0,18/08/2025,15:00,Team E,Team F,,,,2.5,3.2,2.9,,,\n"
           ",,,,,,,,,,,,,\n")

    def test_parse_matches(self):
        ms = parse_matches(data._decode(self.CSV.encode("utf-8")), "E0", "2526")
        self.assertEqual(len(ms), 2)
        self.assertEqual(ms[0].date, date(2025, 8, 16))
        self.assertEqual(ms[0].odds, (1.88, 3.45, 4.1))   # market average preferred
        self.assertEqual(ms[1].odds, (2.5, 3.2, 2.9))     # falls back to B365
        self.assertEqual(ms[1].date, date(2025, 8, 17))
        self.assertEqual(ms[1].result, "D")

    def test_parse_old_format_and_fixtures(self):
        old = "Div,Date,HomeTeam,AwayTeam,HG,AG\nE0,01/09/05,A,B,1,3\n"
        self.assertEqual(parse_matches(old, "E0", "0506")[0].away_goals, 3)
        fx = parse_fixtures("Div,Date,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\nE0,01/11/2026,A,B,2,3.4,3.9\n")
        self.assertEqual((fx[0].league, fx[0].odds), ("E0", (2.0, 3.4, 3.9)))

    def test_seasons(self):
        self.assertEqual(data.season_code(2025), "2526")
        self.assertEqual(data.season_code(1999), "9900")
        self.assertEqual(data.recent_seasons(3, date(2026, 10, 26)), ["2425", "2526", "2627"])
        self.assertEqual(data.recent_seasons(1, date(2026, 3, 1)), ["2526"])


class EngineAndBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches, _, _ = synthetic_league(seasons=3, seed=7)

    def test_backtest_runs_and_has_skill(self):
        res = run_backtest(self.matches, self.matches[-1].season, Settings())
        model, blended, book, base = res.scores
        self.assertGreater(model.n, 100)
        self.assertLess(model.log_loss, base.log_loss)  # real signal exists in the synthetic data
        self.assertIn(res.best_h2h_weight, [i / 20 for i in range(11)])
        self.assertEqual(book.n, model.n)

    def test_engine_predict(self):
        with mock.patch.object(engine, "load_league", return_value=self.matches):
            e = engine.Engine(Settings())
            teams = e.teams("E0")
            self.assertEqual(len(teams), 12)
            r = e.predict("E0", "Team A", "Team L", odds=(1.3, 5.5, 9.0))
            self.assertAlmostEqual(sum(r.final_probs), 1.0)
            self.assertEqual(r.verdict, "Team A win")
            self.assertGreater(len(r.h2h.matches), 0)
            self.assertEqual(len(e.table("E0")), 12)
            from football_predictor.ai import build_summary
            self.assertIn("Team A (home) vs Team L", build_summary(r))
            with self.assertRaises(ValueError):
                e.predict("E0", "Team A", "Team A")


if __name__ == "__main__":
    unittest.main()
