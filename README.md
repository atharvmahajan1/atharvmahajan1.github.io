# Atharv Mahajan

Personal site plus a daily tracker for my NBA 4th-quarter model.

- `index.html` – the site (Home, Projects, Contact).
- `data/tracker.json` – the tracker's results. Updated automatically every morning.
- `tracker/` – the code that grades the bot on each finished game.
  - `core.py` – the frozen model inputs and betting rule (locked Nov 11, 2026).
  - `update.py` – pulls finished games from ESPN and prices from Kalshi, decides what the bot would have done, and saves the result.
  - `model.txt` – the trained LightGBM model.
  - `season_games.csv` – season-to-date games, used for team ratings.
- `.github/workflows/update-tracker.yml` – runs `update.py` every morning at 8:17 AM Central and commits the results.

Run the tracker by hand:

```
pip install -r tracker/requirements.txt
python tracker/update.py
```
