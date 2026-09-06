# ZU Splash stats

Stats page built from Statto exports. `index.html` is the page GitHub Pages serves.

## Rebuild after adding games

    python build_dashboard.py

Inputs live in `data/`:

- `ZU_Splash_Players.csv`, `ZU_Splash_Games.csv` — team-level Statto exports
- `passes/Passes_vs_*.csv` — per-game pass logs
- `players_by_game/Player_Stats_vs_*.csv` — per-game player stats

Schedule, nicknames, MMP/FMP mapping and scoring conventions are at the top of `build_dashboard.py`.

## Deploy

Settings → Pages → Source: deploy from branch `main`, folder `/ (root)`.
