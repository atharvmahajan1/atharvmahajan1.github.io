"""
Decision logic for the NBA team-total tracker. Frozen on Nov 11, 2026.

Given one finished game (ESPN play-by-play with real-world timestamps) and the Kalshi
team-total markets for it (minute-by-minute prices), decide what the bot would have done:

  For every minute of the 4th quarter and every team-total line:
      p     = model's chance the team reaches the line, from the game state at that minute
      edge  = p - ask - fee
  The bot buys the FIRST minute where edge >= 4c and at least 100 contracts traded.
  One bet per game. (If two lines qualify in the same minute, it takes the bigger edge.)

Nothing in here is tuned after the start date. To try a different rule, start a second tracker.
"""
import re

import numpy as np
import pandas as pd

RULE = {"edge": 0.04, "min_volume": 100, "bets_per_game": 1, "stake": 100}
FEATS = ["need", "sec_left", "need_per_min", "margin", "has_ball", "own_fouls_q4", "opp_fouls_q4",
         "in_bonus", "own_to_left", "opp_to_left", "home", "team_ortg", "team_drtg", "opp_ortg",
         "opp_drtg", "exp_ppp", "exp_pace"]
K_SHRINK = 8  # team ratings lean on the league average until a team has played ~8 games


def kalshi_fee(price):
    """Kalshi taker fee per contract: 7% x price x (1 - price), rounded up to the cent."""
    return np.ceil(0.07 * price * (1 - price) * 100) / 100


def clock_to_sec(s):
    s = str(s)
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    try:
        return float(s)
    except ValueError:
        return np.nan


# ------------------------------------------------------------ plays -> events
def plays_to_events(plays, home_id, away_id):
    """ESPN plays (one game) -> event table with kind, side, score, possession and counters."""
    p = plays.copy()
    p["seq"] = pd.to_numeric(p["seq"], errors="coerce")
    p = p.sort_values("seq", kind="stable").reset_index(drop=True)
    ty = p["type"].fillna("").astype(str)
    tid = pd.to_numeric(p["team_id"], errors="coerce")
    side = np.where(tid == float(home_id), "H", np.where(tid == float(away_id), "A", ""))

    kind = np.full(len(p), "OTHER", dtype=object)
    is_ft = ty.str.startswith("Free Throw")
    truthy = lambda c: p[c].map(lambda v: str(v).lower() == "true" or v is True)
    shot = truthy("shooting_play") & ~is_ft
    made = truthy("scoring_play")
    kind[shot & made] = "FGM"
    kind[shot & ~made] = "FGX"
    kind[is_ft & made] = "FTM"
    kind[is_ft & ~made] = "FTX"
    m = ty.str.extract(r"(\d) of (\d)")
    ft_last = (is_ft & (m[0] == m[1]) & ~ty.str.contains("Technical")).to_numpy()
    kind[ty.str.contains("Rebound")] = "REB"
    kind[ty.str.contains("Turnover")] = "TOV"
    foul = ty.str.contains("Foul") & ~ty.str.contains("Turnover|Offensive|Technical|Delay|Too Many")
    kind[foul] = "FOUL"
    kind[ty.str.contains("Timeout")] = "TIMEOUT"
    kind[ty.eq("Substitution")] = "SUB"

    e = pd.DataFrame({
        "period": pd.to_numeric(p["period"], errors="coerce").to_numpy(),
        "sec_left": p["clock"].map(clock_to_sec).to_numpy(),
        "home_score": pd.to_numeric(p["home_score"], errors="coerce").to_numpy(),
        "away_score": pd.to_numeric(p["away_score"], errors="coerce").to_numpy(),
        "side": side, "kind": kind, "ft_last": ft_last,
        "wallclock": pd.to_datetime(p["wallclock"], utc=True, errors="coerce"),
    })
    e[["home_score", "away_score"]] = e[["home_score", "away_score"]].ffill().fillna(0)

    # offensive vs defensive rebound: offensive if the same team missed the last shot
    last_miss = e.side.where(e.kind.isin(["FGX", "FTX"])).ffill()
    reb = e.kind.eq("REB")
    e.loc[reb & (e.side == last_miss), "kind"] = "OREB"
    e.loc[reb & (e.side != last_miss), "kind"] = "DREB"

    # who has the ball after each play (approximate)
    other = e.side.map({"H": "A", "A": "H"})
    poss = pd.Series(np.nan, index=e.index, dtype=object)
    poss[e.kind.eq("FGM")] = other
    poss[e.kind.eq("FTM") & e.ft_last] = other
    poss[e.kind.isin(["OREB", "DREB"])] = e.side
    poss[e.kind.eq("TOV")] = other
    poss[e.kind.eq("OTHER") & e.sec_left.eq(720)] = "?"
    e["poss"] = poss.groupby(e.period).ffill().replace("?", np.nan)

    # running counts
    e["elapsed"] = np.where(e.period <= 4, (e.period - 1) * 720 + (720 - e.sec_left),
                            2880 + (e.period - 5) * 300 + (300 - e.sec_left))
    q4 = e.period.eq(4)
    for s in ["H", "A"]:
        is_s = e.side.eq(s)
        e[f"f4_{s}"] = (q4 & is_s & e.kind.eq("FOUL")).cumsum()
        e[f"f2_{s}"] = (q4 & is_s & e.kind.eq("FOUL") & e.sec_left.le(120)).cumsum()
        e[f"to_{s}"] = (is_s & e.kind.eq("TIMEOUT")).cumsum()
        e[f"to4_{s}"] = (q4 & is_s & e.kind.eq("TIMEOUT")).cumsum()
    e["ts"] = (e.wallclock - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    return e


def game_possessions(e):
    """Average possessions per team: FGA + 0.44 FTA - OREB + TOV."""
    out = []
    for s in ["H", "A"]:
        k = e.kind[e.side == s]
        fga = k.isin(["FGM", "FGX"]).sum()
        fta = k.isin(["FTM", "FTX"]).sum()
        out.append(fga + 0.44 * fta - k.eq("OREB").sum() + k.eq("TOV").sum())
    return float(np.mean(out))


# ------------------------------------------------------------ team ratings
def ratings_before(season_games, date):
    """Season-to-date offense, defense and pace for every team, from games before `date` only.
    season_games: columns date, home, away, home_final, away_final, poss."""
    g = season_games[(season_games.date < date) & season_games.poss.notna()].sort_values("date")
    tot = {}
    lg = [0, 0.0, 0.0]
    for r in g.itertuples():
        for team, f, a in [(r.home, r.home_final, r.away_final), (r.away, r.away_final, r.home_final)]:
            t = tot.setdefault(team, [0, 0.0, 0.0, 0.0])
            t[0] += 1; t[1] += f; t[2] += a; t[3] += r.poss
        lg[0] += 2; lg[1] += r.home_final + r.away_final; lg[2] += 2 * r.poss
    lg_ortg = 100 * lg[1] / lg[2] if lg[2] else 113.0
    lg_pace = lg[2] / lg[0] if lg[0] else 99.0

    def rate(team):
        n, pf, pa, ps = tot.get(team, [0, 0.0, 0.0, 0.0])
        return {"ortg": (100 * pf + K_SHRINK * lg_ortg * lg_pace) / (ps + K_SHRINK * lg_pace),
                "drtg": (100 * pa + K_SHRINK * lg_ortg * lg_pace) / (ps + K_SHRINK * lg_pace),
                "pace": (ps + K_SHRINK * lg_pace) / (n + K_SHRINK)}
    return rate


def matchup_features(rate_home, rate_away, side):
    me, opp = (rate_home, rate_away) if side == "H" else (rate_away, rate_home)
    return {"team_ortg": me["ortg"], "team_drtg": me["drtg"], "opp_ortg": opp["ortg"],
            "opp_drtg": opp["drtg"], "exp_ppp": (me["ortg"] + opp["drtg"]) / 200,
            "exp_pace": (me["pace"] + opp["pace"]) / 2}


# ------------------------------------------------------------ game state at a moment
STATE_COLS = ["ts", "period", "sec_left", "elapsed", "home_score", "away_score", "poss",
              "f4_H", "f4_A", "f2_H", "f2_A", "to_H", "to_A", "to4_H", "to4_A"]


def arrays(e):
    e = e.dropna(subset=["ts"]).sort_values("ts", kind="stable")
    return {c: e[c].to_numpy() for c in STATE_COLS}


def state_at(g, T, side):
    """Features for one team at real-world time T, using only plays logged by then."""
    if T >= g["ts"][-1]:  # game over: Kalshi keeps quoting stale prices until it settles
        return None
    i = np.searchsorted(g["ts"], T, side="right") - 1  # binary search for the latest play
    if i < 0 or g["period"][i] != 4:
        return None
    t = g["sec_left"][i]
    if t < 5:
        return None
    opp = "A" if side == "H" else "H"
    me = g["home_score"][i] if side == "H" else g["away_score"][i]
    them = g["away_score"][i] if side == "H" else g["home_score"][i]
    p = g["poss"][i]
    own_to = min(7 - g[f"to_{side}"][i], 4 - g[f"to4_{side}"][i])
    opp_to = min(7 - g[f"to_{opp}"][i], 4 - g[f"to4_{opp}"][i])
    return dict(sec_left=max(t, 1), pts=me, margin=me - them,
                has_ball=1.0 if p == side else (0.0 if p == opp else np.nan),
                own_fouls_q4=g[f"f4_{side}"][i], opp_fouls_q4=g[f"f4_{opp}"][i],
                in_bonus=int(g[f"f4_{opp}"][i] >= 4 or (t <= 120 and g[f"f2_{opp}"][i] >= 1)),
                own_to_left=max(own_to, 0), opp_to_left=max(opp_to, 0), home=int(side == "H"))


# ------------------------------------------------------------ the decision
def decide(events, markets, candles, model, rate_home, rate_away):
    """
    events : output of plays_to_events for this game
    markets: DataFrame with ticker, side ('H'/'A'), target (points needed to win YES), result ('yes'/'no')
    candles: DataFrame with ticker, ts, bid, ask, vol (one row per minute)
    Returns (bet or None, closest moment, chart series for the shown line).
    """
    g = arrays(events)
    if len(g["ts"]) == 0:
        return None, None, []
    px = candles.merge(markets, on="ticker")
    px = px[(px.ask > 0) & (px.ask < 1) & (px.bid > 0)]
    rows = []
    for r in px.itertuples():
        s = state_at(g, r.ts, r.side)
        if s is None:
            continue
        need = r.target - s["pts"]
        if need <= 0:
            continue  # already reached: effectively settled
        s.update(ticker=r.ticker, side=r.side, target=r.target, ts=r.ts, need=need,
                 ask=r.ask, bid=r.bid, vol=r.vol, result=r.result)
        s.update(matchup_features(rate_home, rate_away, r.side))
        rows.append(s)
    if not rows:
        return None, None, []
    d = pd.DataFrame(rows)
    d["need_per_min"] = d.need / (d.sec_left / 60)
    d["p"] = model.predict(d[FEATS])
    d["fee"] = kalshi_fee(d.ask)
    d["edge"] = d.p - d.ask - d.fee
    d["mid"] = (d.ask + d.bid) / 2

    ok = d[(d.edge >= RULE["edge"]) & (d.vol >= RULE["min_volume"])]
    pick = ok.sort_values(["ts", "edge"], ascending=[True, False]).head(1)
    liquid = d[d.vol >= RULE["min_volume"]]
    closest = liquid.sort_values("edge", ascending=False).head(1) if len(liquid) else d.sort_values("edge", ascending=False).head(1)

    def describe(x):
        x = x.iloc[0]
        won = int(x.result == "yes")
        cost = x.ask + x.fee
        return {"ticker": x.ticker, "side": x.side, "line": float(x.target) - 0.5, "pts": int(x.pts),
                "sec_left": int(round(x.sec_left)), "ask": round(float(x.ask), 2), "fee": round(float(x.fee), 2),
                "p": round(float(x.p), 3), "edge": round(float(x.edge), 3), "vol": int(x.vol), "ts": int(x.ts),
                "won": won, "profit": round(RULE["stake"] * ((1 - cost) if won else -cost) / cost, 2)}

    bet = describe(pick) if len(pick) else None
    near = describe(closest)
    show = (bet or near)["ticker"]
    s = d[d.ticker == show].sort_values("ts")
    series = [[round(float(a) / 60, 2), round(float(p), 3), round(float(m), 3)] for a, p, m in zip(s.sec_left, s.p, s.mid)]
    return bet, near, series
