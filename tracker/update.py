"""
Daily update for the NBA tracker. GitHub Actions runs this every morning.

For every finished game since START_DATE that isn't recorded yet:
  1. pull ESPN play-by-play (with the real-world time of each play)
  2. pull Kalshi team-total markets and minute-by-minute prices for the game
  3. run the frozen model and rule (core.py) to see if and when the bot would have bought
  4. add the game to data/tracker.json
Games whose Kalshi markets haven't settled yet are retried on the next run.

Run by hand:  python tracker/update.py
"""
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core  # noqa: E402

START_DATE = int(os.environ.get("TRACKER_START", "20261111"))  # first night of the forward test
SEASON_START = date(2026, 10, 20)                                # for season-to-date team ratings
OUT = os.environ.get("TRACKER_OUT", os.path.join(ROOT, "data", "tracker.json"))
CACHE = os.environ.get("TRACKER_CACHE", os.path.join(HERE, "season_games.csv"))
MODEL = os.path.join(HERE, "model.txt")

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
ESPN2KAL = {"NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
KAL2ESPN = {v: k for k, v in ESPN2KAL.items()}

session = requests.Session()
session.headers.update(HEADERS)


def get(url, params=None):
    err = None
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=20)
            if r.ok:
                return r.json()
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {err}")


# ------------------------------------------------------------ data sources
def scoreboard(day):
    """Finished regular-season games on a date (YYYYMMDD int)."""
    data = get(ESPN + "/scoreboard", {"dates": str(day)})
    out = []
    for ev in data.get("events", []):
        if not ev.get("status", {}).get("type", {}).get("completed"):
            continue
        if ev.get("season", {}).get("type") != 2:
            continue
        comp = ev["competitions"][0]["competitors"]
        home = next(c for c in comp if c["homeAway"] == "home")
        away = next(c for c in comp if c["homeAway"] == "away")
        out.append({"game_id": str(ev["id"]), "date": day,
                    "home": home["team"]["abbreviation"], "home_id": int(home["team"]["id"]),
                    "away": away["team"]["abbreviation"], "away_id": int(away["team"]["id"]),
                    "home_final": int(home["score"]), "away_final": int(away["score"])})
    return out


def plays(game_id):
    data = get(ESPN + "/summary", {"event": game_id})
    rows = []
    for p in data.get("plays", []):
        rows.append({"seq": p.get("sequenceNumber"), "period": p.get("period", {}).get("number"),
                     "clock": p.get("clock", {}).get("displayValue"), "wallclock": p.get("wallclock"),
                     "type": p.get("type", {}).get("text"), "team_id": (p.get("team") or {}).get("id"),
                     "home_score": p.get("homeScore"), "away_score": p.get("awayScore"),
                     "scoring_play": p.get("scoringPlay"), "shooting_play": p.get("shootingPlay")})
    return pd.DataFrame(rows)


def kalshi_markets(day, away, home):
    d = datetime.strptime(str(day), "%Y%m%d").strftime("%y%b%d").upper()
    event = f"KXNBATEAMTOTAL-{d}{ESPN2KAL.get(away, away)}{ESPN2KAL.get(home, home)}"
    return get(KALSHI + "/markets", {"event_ticker": event, "limit": 200}).get("markets", [])


def kalshi_candles(ticker, start, end):
    data = get(f"{KALSHI}/series/KXNBATEAMTOTAL/markets/{ticker}/candlesticks",
               {"start_ts": int(start), "end_ts": int(end), "period_interval": 1})
    return data.get("candlesticks", [])


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------ one game
def decide_game(g, ev, cache, model):
    base = {k: g[k] for k in ["game_id", "date", "away", "home", "away_final", "home_final"]}
    raw = kalshi_markets(g["date"], g["away"], g["home"])
    if not raw:
        return {**base, "status": "no_market"}
    mk = pd.DataFrame(raw)
    mk["team"] = mk.ticker.str.split("-").str[2].str.extract(r"^([A-Z]+)")[0].map(lambda x: KAL2ESPN.get(x, x))
    mk["side"] = np.where(mk.team == g["home"], "H", np.where(mk.team == g["away"], "A", ""))
    mk = mk[mk.side != ""].copy()
    mk["target"] = np.ceil(mk.floor_strike.astype(float)).astype(int)
    if not mk.result.isin(["yes", "no"]).all():
        return {**base, "status": "pending"}

    t0, t1 = ev.ts.min(), ev.ts.max()
    rows = []
    for t in mk.ticker:
        for c in kalshi_candles(t, t0 - 600, t1 + 1800):
            rows.append((t, c["end_period_ts"], num(c.get("yes_bid", {}).get("close_dollars")),
                         num(c.get("yes_ask", {}).get("close_dollars")), num(c.get("volume_fp"))))
        time.sleep(0.12)
    candles = pd.DataFrame(rows, columns=["ticker", "ts", "bid", "ask", "vol"])

    rate = core.ratings_before(cache, g["date"])
    bet, near, series = core.decide(ev, mk[["ticker", "side", "target", "result"]], candles, model,
                                    rate(g["home"]), rate(g["away"]))
    team = lambda x: g["home"] if x["side"] == "H" else g["away"]
    if bet:
        bet["team"] = team(bet)
    if near:
        near["team"] = team(near)
    return {**base, "status": "bet" if bet else "no_bet", "bet": bet, "closest": near,
            "markets": int(len(mk)), "series": series}


# ------------------------------------------------------------ main
def main():
    model = lgb.Booster(model_file=MODEL)
    cache = pd.read_csv(CACHE, dtype={"game_id": str}) if os.path.exists(CACHE) else pd.DataFrame(
        columns=["game_id", "date", "home", "away", "home_id", "away_id", "home_final", "away_final", "poss"])
    tracker = json.load(open(OUT)) if os.path.exists(OUT) else {}
    games = {g["game_id"]: g for g in tracker.get("games", [])}

    yesterday = datetime.now(ZoneInfo("America/Chicago")).date() - timedelta(days=1)
    cached_days = set(cache.date.astype(int))
    day = SEASON_START
    while day <= yesterday:
        d = int(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
        if d < START_DATE and d in cached_days:
            continue  # ratings history already stored for this day
        try:
            todays = scoreboard(d)
        except Exception as e:
            print(f"{d}: scoreboard failed ({e}), will retry next run")
            continue
        for g in todays:
            gid = g["game_id"]
            in_cache = gid in set(cache.game_id)
            rec = games.get(gid)
            needs_decision = d >= START_DATE and (rec is None or rec["status"] == "pending")
            if in_cache and not needs_decision:
                continue
            try:
                ev = core.plays_to_events(plays(gid), g["home_id"], g["away_id"])
                if needs_decision:
                    res = decide_game(g, ev, cache, model)
                    games[gid] = res
                    b = res.get("bet")
                    print(f"{d} {g['away']} @ {g['home']}: {res['status']}"
                          + (f" {b['team']} over {b['line']} at {b['ask']:.2f}, {'won' if b['won'] else 'lost'}" if b else ""))
                if not in_cache:
                    cache.loc[len(cache)] = [gid, d, g["home"], g["away"], g["home_id"], g["away_id"],
                                             g["home_final"], g["away_final"], core.game_possessions(ev)]
            except Exception as e:
                print(f"{d} {gid}: failed ({e}), will retry next run")
            time.sleep(0.2)

    # markets that never settle (postponed games) stop being retried after 3 days
    cutoff = int((yesterday - timedelta(days=3)).strftime("%Y%m%d"))
    for g in games.values():
        if g["status"] == "pending" and g["date"] < cutoff:
            g["status"] = "unsettled"

    done = sorted(games.values(), key=lambda g: (g["date"], g["game_id"]))
    bets = [g["bet"] for g in done if g["status"] == "bet"]
    tracker = {
        "start_date": f"{str(START_DATE)[:4]}-{str(START_DATE)[4:6]}-{str(START_DATE)[6:]}",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": {**core.RULE, "summary": "Buy the first 4th-quarter minute where model chance - Kalshi ask - fee is at least 4 cents, with 100+ contracts traded that minute. One $100 bet per game."},
        "model": "LightGBM, trained on 2021-22 through 2025-26. Frozen Nov 11, 2026.",
        "totals": {"games": sum(g["status"] in ("bet", "no_bet") for g in done), "bets": len(bets),
                   "wins": sum(b["won"] for b in bets), "profit": round(sum(b["profit"] for b in bets), 2),
                   "staked": core.RULE["stake"] * len(bets)},
        "games": done,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(tracker, f, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    cache.to_csv(CACHE, index=False)
    t = tracker["totals"]
    print(f"Tracked {t['games']} games, {t['bets']} bets, {t['wins']} wins, profit ${t['profit']:,.2f}")


if __name__ == "__main__":
    main()
