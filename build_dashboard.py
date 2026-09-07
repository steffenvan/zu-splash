"""Build a self-contained HTML stats page from Statto exports.

Usage:
    python build_dashboard.py

Inputs:
    PLAYERS_CSV, GAMES_CSV       the team-level exports (aggregate over all games)
    PASSES_DIR/Passes_vs_*.csv   per-game pass logs (one file per game)
    PLAYERS_BY_GAME_DIR/*.csv    per-game Player Stats exports, same columns as
                                 the team-level Players file. When present, the
                                 per-game views also get points played, blocks,
                                 points won and +/-.

Output is a single HTML file with inline SVG charts and a small script for the
game tabs and table sorting. No external dependencies, opens on any phone.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ADAPT: paths, labels, output
PLAYERS_CSV = Path("data/ZU_Splash_Players.csv")
GAMES_CSV = Path("data/ZU_Splash_Games.csv")
PASSES_DIR = Path("data/passes")
PLAYERS_BY_GAME_DIR = Path("data/players_by_game")
OUT_HTML = Path("index.html")
TEAM = "ZU Splash"
EVENT_LABEL = "WUCC 2026"

# ADAPT: short display names keyed by the exact "Player" value in the export.
# Anyone not listed is shown by first name, duplicates get their jersey number.
NICKNAMES = {"12 Anika Gnaedinger": "Ani", "54 Thanh Elsener": "Trissy"}

# ADAPT: scoring conventions and thresholds
A2_WEIGHT = 0.0         # credit for a hockey assist in +/-: 0 (common), 0.5 (fantasy-style) or 1
LONG_THROW_M = 20       # forward metres from which a throw counts as long (Statto's huck flag only fires at ~60 m)
EXPECT_BINS = [-1e9, 0, 10, 20, 30, 1e9]   # forward-distance bins for expected completion
PAIRS_TO_LIST = 12      # thrower->receiver pairs in the "most common connections" list
SUSPECT_GAMES: set[str] = set()   # opponents whose games were only partly tracked (greyed, left out of totals)

# ADAPT: which players are MMP and which FMP, keyed by the "Player" value in the export.
# Anyone missing is left out of the MMP/FMP breakdown and reported when the script runs.
# Confirmed by Steffen for the WUCC 2026 roster.
MATCHING = {
    "10 Cedric von": "MMP", "30 Quinn Bergeron": "MMP", "99 Luca Godenzini": "MMP", "70 Steffen Van": "MMP",
    "09 Jan Ghadamian": "MMP", "03 Jason Christian": "MMP", "50 Philip Arm": "MMP", "19 Moritz Roevekamp": "MMP",
    "18 Martin Fahndrich": "MMP", "34 Noah Kesseli": "MMP", "95 Benjamin Willi": "MMP", "42 Tim Mueller": "MMP",
    "01 Oscar Weber": "MMP",
    "28 Anika": "FMP", "12 Anika Gnaedinger": "FMP", "81 Emily Cai": "FMP", "27 Erica Lastufka": "FMP",
    "11 Ronja Seibert": "FMP", "59 Yael Grob": "FMP", "17 Alexandra Tutuian": "FMP", "32 Fiona Pacifico": "FMP",
    "08 Francesca Pietrafesa": "FMP", "51 Lilli Gerber": "FMP", "23 Lola Bardel": "FMP", "29 Malvina Voclova": "FMP",
    "54 Thanh Elsener": "FMP",
}
TOP_TARGETS = 3   # how many favourite targets / throwers to list per player
THROWER_SHARE = 0.5   # share of a completed pass's value credited to the thrower, rest to the receiver
ROSE_SECTORS = 12     # direction sectors in the tendency radials
# ADAPT: order of the sections on the page
SECTION_ORDER = ["games", "sheet", "playing", "throwing", "connections", "receiving", "scoring", "points_won", "models"]

# ADAPT: actual schedule, keyed by opponent as named in the Games export. Overrides the
# dates and times Statto recorded (which are when the game was tagged, not played) and
# fixes the order of the tabs and score strip. Stage is shown under each score.
SCHEDULE = {
    "BFD Redshot":      ("2026-08-15", "17:00", "Pool G"),
    "Hammers Bs As":    ("2026-08-16", "09:00", "Pool G"),
    "Rebel":            ("2026-08-16", "15:00", "Pool G"),
    "Avalon":           ("2026-08-17", "13:00", "Pool G"),
    "Shame":            ("2026-08-18", "13:00", "Pool G"),
    "Meclao":           ("2026-08-19", "09:00", "Playoff 1–32"),
    "Tiefseetaucher":   ("2026-08-19", "15:30", "Playoff 1–32, round 2"),
    "Union":            ("2026-08-20", "09:00", "Quarterfinal"),
    "Tartu Turbulence": ("2026-08-20", "15:00", "Semifinal"),
    "Monkey Grenoble":  ("2026-08-21", "09:00", "Final"),
}

# palette
INK = "#12283F"
SPLASH = "#1877D2"    # offence, primary
AMBER = "#E39A2E"     # defence, turnovers
TINT = "#E3F0FB"
MIST = "#8A99AB"
RULE = "#D5DEE8"

PLAYER_RENAMES = {
    "Points played total": "pts", "Offense points played": "o_pts", "Defense points played": "d_pts",
    "Offense points won": "o_won", "Defense points won": "d_won", "Throws": "throws", "Catches": "catches",
    "Assists": "assists", "Secondary assists": "a2", "Goals": "goals", "Thrower errors": "te",
    "Receiver errors": "re", "Defensive blocks": "blocks",
    "Total completed throw gain (m)": "throw_gain", "Average completed throw gain (m)": "gain_per_throw",
    "Average caught pass gain (m)": "gain_per_catch",
}
GAME_RENAMES = {
    "Our score": "us", "Opponent's score": "them", "Won points started on offense (holds)": "holds",
    "Won points started on defense (breaks)": "breaks", "Turnovers": "turns", "Defensive blocks": "blocks",
}


# ---------------------------------------------------------------- loading
def short_names(players: pd.Series) -> pd.Series:
    parts = players.str.extract(r"^(\d+)\s+(.*)$")
    number, name = parts[0], parts[1].fillna(players)
    short = players.map(NICKNAMES).fillna(name.str.split().str[0])
    dup = short.duplicated(keep=False)
    short = short.where(~dup, short + " #" + number.fillna("?"))
    return short


def finish_player_frame(p: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Works for the full frame and for pass-log-derived frames."""
    p = p.copy()
    p["short"] = short_names(p["Player"])
    p["turnovers"] = p["te"] + p["re"]
    p["completion"] = 1 - p["te"] / p["throws"].where(p["throws"] > 0, 1)
    if "blocks" in p:
        p["plus_minus"] = p["goals"] + p["assists"] + p["blocks"] - p["turnovers"] + A2_WEIGHT * p["a2"]
    if "pts" in p:
        p["won"] = p["o_won"] + p["d_won"]
        p["win_pct"] = p["won"] / p["pts"].where(p["pts"] > 0, 1)
        p["per10"] = 10 / p["pts"].where(p["pts"] > 0, 1)
    return p


def load_players(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, thousands="'").rename(columns=PLAYER_RENAMES)
    return finish_player_frame(p)


def load_games() -> pd.DataFrame:
    g = pd.read_csv(GAMES_CSV, thousands="'").rename(columns=GAME_RENAMES)
    g["stage"] = ""
    for opp, (date, time, stage) in SCHEDULE.items():
        hit = g["Opponent"] == opp
        g.loc[hit, ["Date", "Time", "stage"]] = [date, time, stage]
    missing = set(SCHEDULE) - set(g["Opponent"])
    if missing:
        print(f"warning: SCHEDULE names not found in the Games export: {sorted(missing)}")
    g["Date"] = pd.to_datetime(g["Date"])
    g = g.sort_values(["Date", "Time"]).reset_index(drop=True)
    g["suspect"] = g["Opponent"].isin(SUSPECT_GAMES)
    return g


def load_passes() -> pd.DataFrame:
    frames = []
    for f in sorted(PASSES_DIR.glob("Passes_vs_*.csv")):
        d = pd.read_csv(f)
        m = re.match(r"Passes_vs_+(.+?)_(\d{4}-\d{2}-\d{2})", f.name)
        d["game"] = m.group(1).replace("_", " ") if m else f.stem
        d["game_date"] = m.group(2) if m else ""
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["Thrower", "Receiver"])   # unattributable passes
    d = d.rename(columns={"Turnover?": "turn", "Thrower error?": "te", "Receiver error?": "re",
                          "Assist?": "assist", "Secondary assist?": "a2", "Huck?": "huck",
                          "Swing?": "swing", "Dump?": "dump", "Forward distance (m)": "fwd"})
    d["completed"] = 1 - d["turn"]
    d["kind"] = "upfield"
    d.loc[d["dump"] == 1, "kind"] = "dump"
    d.loc[d["swing"] == 1, "kind"] = "swing"
    d.loc[(d["fwd"] >= LONG_THROW_M) | (d["huck"] == 1), "kind"] = "long"
    # expected completion: team rate within the throw's forward-distance bin, over all logged games
    d["bin"] = pd.cut(d["fwd"], EXPECT_BINS, right=False)
    d["expected"] = d.groupby("bin", observed=True)["completed"].transform("mean")
    d = d.rename(columns=COORD_RENAMES)
    d, models = fit_models(d)
    d.attrs["models"] = models
    return d


COORD_RENAMES = {
    "Start X (0 -> 1 = left sideline -> right sideline)": "sx",
    "Start Y (0 -> 1 = back of opponent endzone -> back of own endzone)": "sy",
    "End X (0 -> 1 = left sideline -> right sideline)": "ex",
    "End Y (0 -> 1 = back of opponent endzone -> back of own endzone)": "ey",
    "Distance (m)": "dist", "Left-to-right distance (m)": "lat",
}


def _logit_fit(X, y, l2=1e-3, iters=60):
    """Ridge-regularised logistic regression by Newton's method (no sklearn needed)."""
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ (y - p) - l2 * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _logit_pred(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-X @ w))


def _xs_feats(y, x):
    return np.column_stack([y, y ** 2, np.abs(x - 0.5)])


def _cp_feats(d):
    return np.column_stack([d["dist"], d["dist"] ** 2 / 50, d["fwd"], np.abs(d["lat"]), d["sy"], np.abs(d["sx"] - 0.5)])


def fit_models(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Expected score by field position, expected completion probability, and per-pass
    expected contribution, all fitted on the logged games themselves.

    P(score | position): logistic in distance to the end zone (linear + quadratic) and
    distance from the centre line, fitted on every pass start position with the
    possession's outcome as the label.
    xCP: logistic in throw length, forward gain, lateral movement and start position.
    Contribution of a completed pass = P(score at end) - P(score at start), split
    THROWER_SHARE / (1 - THROWER_SHARE) between thrower and receiver. A turnover costs
    P(score at start): the thrower for a throwaway, the receiver for a drop, half each
    when Statto tagged both. The
    opponent's counter-value is left out because the logs hold no defensive events to
    balance it, so the team's numbers sum to about zero by construction."""
    d = d.copy()
    d["pid"] = d["game"] + "|" + d["Point"].astype(str) + "|" + d["Possession"].astype(str)
    d["scored"] = d["pid"].map(d.groupby("pid")["assist"].max())
    w_xs = _logit_fit(_xs_feats(d["sy"].values, d["sx"].values), d["scored"].values.astype(float))
    w_cp = _logit_fit(_cp_feats(d), d["completed"].values.astype(float), l2=1e-2)
    d["xs_start"] = _logit_pred(w_xs, _xs_feats(d["sy"].values, d["sx"].values))
    d["xs_end"] = np.where(d["assist"] == 1, 1.0, _logit_pred(w_xs, _xs_feats(d["ey"].values, d["ex"].values)))
    d["xcp"] = _logit_pred(w_cp, _cp_feats(d))
    gain = d["xs_end"] - d["xs_start"]
    # a turnover tagged as both throwaway and drop (shared fault) is charged half to each
    blame_t = d["te"] / (d["te"] + d["re"]).replace(0, 1)
    blame_r = d["re"] / (d["te"] + d["re"]).replace(0, 1)
    d["ec_thrower"] = np.where(d["completed"] == 1, THROWER_SHARE * gain, -blame_t * d["xs_start"])
    d["ec_receiver"] = np.where(d["completed"] == 1, (1 - THROWER_SHARE) * gain, -blame_r * d["xs_start"])
    info = {
        "possessions": int(d["pid"].nunique()), "goals": int(d.groupby("pid")["assist"].max().sum()),
        "xs_curve": [(y, float(_logit_pred(w_xs, _xs_feats(np.array([y]), np.array([0.5])))[0]),
                      float(_logit_pred(w_xs, _xs_feats(np.array([y]), np.array([0.05])))[0])) for y in (0.23, 0.38, 0.53, 0.68, 0.83)],
    }
    return d, info


def players_from_passes(d: pd.DataFrame) -> pd.DataFrame:
    """Per-player counts derivable from a pass log (no points played, blocks or points won)."""
    thr = d.groupby("Thrower").agg(throws=("completed", "size"), te=("te", "sum"), assists=("assist", "sum"),
                                   a2=("a2", "sum"))
    comp = d[d["completed"] == 1]
    thr_gain = comp.groupby("Thrower")["fwd"].agg(throw_gain="sum", gain_per_throw="mean")
    rec = d.groupby("Receiver").agg(re=("re", "sum"), goals=("assist", "sum"))
    cat = comp.groupby("Receiver")["fwd"].agg(catches="size", gain_per_catch="mean")
    p = thr.join(thr_gain, how="outer").join(rec, how="outer").join(cat, how="outer").fillna(0)
    p.index.name = "Player"
    return finish_player_frame(p.reset_index())


def load_players_by_game(games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(PLAYERS_BY_GAME_DIR.glob("*.csv")):
        m = re.match(r"Players?_?(?:Stats)?_vs_+(.+?)_(\d{4}-\d{2}-\d{2})", f.name)
        opp = m.group(1).replace("_", " ") if m else f.stem
        out[opp] = load_players(f)
    return out


def solve_o_points(ps: pd.DataFrame) -> dict[int, bool]:
    """Which points started on offence, from the per-game Player Stats file.

    Each player's "Offense points played" must equal the number of O points among the
    points they played. That is a small binary system; starting from a lineup guess and
    flipping points until every player's count is satisfied solves it exactly for these
    games (residual 0 on all ten)."""
    import numpy as np
    pts = sorted({int(x) for v in ps["Points played"] for x in str(v).split(",")})
    A = np.zeros((len(ps), len(pts)))
    for i, v in enumerate(ps["Points played"]):
        for x in str(v).split(","):
            A[i, pts.index(int(x))] = 1
    o_target = ps["o_pts"].values.astype(float)
    d_counts = ps["d_pts"].values
    x = np.zeros(len(pts))
    for j in range(len(pts)):
        on = A[:, j] == 1
        x[j] = (d_counts[on] == 0).sum() > (o_target[on] == 0).sum()

    def resid(v):
        return float(((A @ v - o_target) ** 2).sum())

    best, improved = resid(x), True
    while best > 0 and improved:
        improved = False
        for j in range(len(pts)):
            y = x.copy(); y[j] = 1 - y[j]
            if (r := resid(y)) < best:
                x, best, improved = y, r, True
        if not improved:
            for j in range(len(pts)):
                for k in range(j + 1, len(pts)):
                    y = x.copy(); y[j] = 1 - y[j]; y[k] = 1 - y[k]
                    if (r := resid(y)) < best:
                        x, best, improved = y, r, True
    if best > 0:
        print("warning: could not fully resolve O/D points for a game, residual", best)
    return {p: bool(v) for p, v in zip(pts, x)}


def point_summary(ps: pd.DataFrame, pl: pd.DataFrame) -> dict:
    """O points, D points, holds, clean holds (scored on our first possession) and breaks."""
    is_o = solve_o_points(ps)
    scored = set(pl.loc[pl["assist"] == 1, "Point"])
    turns = pl.groupby("Point")["turn"].sum()
    return {
        "o_points": sum(is_o.values()),
        "d_points": sum(1 for v in is_o.values() if not v),
        "holds_calc": sum(1 for p, o in is_o.items() if o and p in scored),
        "clean_holds": sum(1 for p, o in is_o.items() if o and p in scored and turns.get(p, 0) == 0),
        "breaks_calc": sum(1 for p, o in is_o.items() if not o and p in scored),
    }


# ---------------------------------------------------------------- svg helpers
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def hbars(rows, *, color=SPLASH, value_fmt="{:.0f}", note_fmt=None, max_value=None,
          label_w=118, bar_w=300, row_h=26) -> str:
    """rows = [(label, value, note)]"""
    if not rows:
        return '<p class="muted">Nothing to show.</p>'
    vmax = max_value or max(v for _, v, _ in rows) or 1
    h = row_h * len(rows)
    w = label_w + bar_w + 190
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" font-family="inherit" font-size="13">']
    for i, (label, val, note) in enumerate(rows):
        y = i * row_h
        bw = max(0.0, val / vmax) * bar_w
        out.append(f'<text x="{label_w - 8}" y="{y + 17}" text-anchor="end" fill="{INK}">{esc(label)}</text>')
        out.append(f'<rect x="{label_w}" y="{y + 5}" width="{bw:.1f}" height="{row_h - 10}" fill="{color}" rx="2"/>')
        txt = value_fmt.format(val)
        if note is not None and note_fmt:
            txt += f' <tspan fill="{MIST}">{esc(note_fmt.format(note))}</tspan>'
        out.append(f'<text x="{label_w + bw + 6:.1f}" y="{y + 17}" fill="{INK}">{txt}</text>')
    out.append("</svg>")
    return "\n".join(out)


def diverging_bars(rows, *, pos=SPLASH, neg=MIST, value_fmt="{:+.1f}", label_w=118, bar_w=260, row_h=26) -> str:
    """rows = [(label, value, note)], bars from a zero line, right for positive and left for negative."""
    if not rows:
        return '<p class="muted">Nothing to show.</p>'
    vmax = max(abs(v) for _, v, _ in rows) or 1
    h = row_h * len(rows)
    half = bar_w / 2
    zero = label_w + half
    w = label_w + bar_w + 200
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" font-family="inherit" font-size="13">',
           f'<line x1="{zero}" x2="{zero}" y1="0" y2="{h}" stroke="{RULE}"/>']
    for i, (label, val, note) in enumerate(rows):
        y = i * row_h
        bw = abs(val) / vmax * half
        x0 = zero if val >= 0 else zero - bw
        out.append(f'<text x="{label_w - 8}" y="{y + 17}" text-anchor="end" fill="{INK}">{esc(label)}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y + 5}" width="{bw:.1f}" height="{row_h - 10}" fill="{pos if val >= 0 else neg}" rx="2"/>')
        tx = zero + bw + 6 if val >= 0 else zero + 6
        out.append(f'<text x="{tx:.1f}" y="{y + 17}" fill="{INK}">{value_fmt.format(val)}'
                   f' <tspan fill="{MIST}">{esc(note)}</tspan></text>')
    out.append("</svg>")
    return "\n".join(out)


def stacked_bars(rows, *, colors=(SPLASH, AMBER), label_w=118, bar_w=300, row_h=22, marker=None) -> str:
    """rows = [(label, [v1, v2])]; marker = (value, label) draws a dashed vertical reference line."""
    vmax = max(sum(v) for _, v in rows) or 1
    top = 18 if marker else 0
    h = row_h * len(rows) + top
    w = label_w + bar_w + 60
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" font-family="inherit" font-size="13">']
    if marker:
        mx = label_w + marker[0] / vmax * bar_w
        out.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{top}" y2="{h}" stroke="{INK}" stroke-width="1" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{mx:.1f}" y="12" text-anchor="middle" fill="{INK}" font-size="12">{esc(marker[1])}</text>')
    for i, (label, vals) in enumerate(rows):
        y = i * row_h + top
        x = label_w
        out.append(f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" fill="{INK}">{esc(label)}</text>')
        for v, c in zip(vals, colors):
            bw = v / vmax * bar_w
            if bw > 0:
                out.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{bw:.1f}" height="{row_h - 8}" fill="{c}"/>')
            if bw >= 22:
                out.append(f'<text x="{x + bw / 2:.1f}" y="{y + 15}" text-anchor="middle" fill="#fff" font-size="11">{v:.0f}</text>')
            x += bw
        out.append(f'<text x="{x + 6:.1f}" y="{y + 15}" fill="{INK}">{sum(vals):.0f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def rose_svg(counts_throw, counts_catch, size=96) -> str:
    """Two direction roses side by side: throws (blue) and catches (amber). Forward is up."""
    import math
    n = len(counts_throw)
    out = [f'<svg viewBox="0 0 {2 * size} {size}" width="{2 * size}" height="{size}">']
    for k, (counts, color) in enumerate(((counts_throw, SPLASH), (counts_catch, AMBER))):
        cx, cy, r = k * size + size / 2, size / 2, size / 2 - 4
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{RULE}"/>')
        out.append(f'<line x1="{cx}" x2="{cx}" y1="{cy - r}" y2="{cy + r}" stroke="{RULE}"/>')
        out.append(f'<line x1="{cx - r}" x2="{cx + r}" y1="{cy}" y2="{cy}" stroke="{RULE}"/>')
        m = max(counts) or 1
        for i, c in enumerate(counts):
            if not c:
                continue
            rr = r * math.sqrt(c / m)
            a0 = (i - 0.5) * 2 * math.pi / n
            a1 = (i + 0.5) * 2 * math.pi / n
            x0, y0 = cx + rr * math.sin(a0), cy - rr * math.cos(a0)
            x1, y1 = cx + rr * math.sin(a1), cy - rr * math.cos(a1)
            out.append(f'<path d="M{cx:.1f},{cy:.1f} L{x0:.1f},{y0:.1f} A{rr:.1f},{rr:.1f} 0 0 1 {x1:.1f},{y1:.1f} Z" fill="{color}" fill-opacity="0.75"/>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- page pieces
def a2_note() -> str:
    return f" + {A2_WEIGHT:g} × A2" if A2_WEIGHT else ""


def score_cards(g: pd.DataFrame) -> str:
    cards = []
    for _, r in g.iterrows():
        cls = "game" + (" loss" if r["Result"] == "Loss" else "")
        meta_txt = ", ".join(x for x in [r["stage"], "partial data" if r["suspect"] else ""] if x)
        meta = f'<div class="meta">{esc(meta_txt)}</div>' if meta_txt else ""
        cards.append(f'<div class="{cls}"><div class="score">{r.us}<span> {r.them}</span></div>'
                     f'<div class="opp">{esc(r.Opponent)}</div>{meta}</div>')
    return f'<div class="strip">{"".join(cards)}</div>'


def games_table(g: pd.DataFrame) -> str:
    detailed = "clean_holds" in g.columns and g["clean_holds"].notna().all()
    rows = []
    for _, r in g.iterrows():
        cls = ' class="suspect"' if r["suspect"] else ""
        cells = [f'<td>{esc(r.Opponent)}<span class="stage">{esc(r.stage)}</span></td>', f"<td>{r.us}–{r.them}</td>"]
        if detailed:
            cells += [f"<td>{r.holds}/{r.o_points}</td>", f"<td>{r.clean_holds}</td>", f"<td>{r.breaks}/{r.d_points}</td>"]
        else:
            cells += [f"<td>{r.holds}</td>", f"<td>{r.breaks}</td>"]
        cells += [f"<td>{r.turns}</td>", f"<td>{r.blocks}</td>"]
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    head = (["Opponent", "Score"] + (["Holds / O pts", "Clean holds", "Breaks / D pts"] if detailed else ["Holds", "Breaks"])
            + ["Turns", "Blocks"])
    return ('<div class="scroll"><table class="games"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def build_view(p: pd.DataFrame, g: pd.DataFrame, passes: pd.DataFrame, *, single_game: bool, prefix: str) -> str:
    """One complete set of sections for a player frame, a games frame and a pass log."""
    full = "pts" in p.columns      # playing time, blocks, points won available
    has_passes = len(passes) > 0
    clean = g[~g["suspect"]]
    sec = {k: [] for k in SECTION_ORDER}
    add = lambda k, h: sec[k].append(h)

    # -- games
    detailed = "clean_holds" in g.columns and g["clean_holds"].notna().all()
    lead = "This game" if single_game else f"Over the {len(clean)} games"
    if detailed:
        o_n, d_n = int(clean.o_points.sum()), int(clean.d_points.sum())
        h, ch, b = int(clean.holds.sum()), int(clean.clean_holds.sum()), int(clean.breaks.sum())
        team_line = (f"{lead}: {h} holds from {o_n} O points ({h / o_n:.0%}), {ch} of them clean ({ch / o_n:.0%} of O points), "
                     f"{b} breaks from {d_n} D points ({b / d_n:.0%}), {int(clean.turns.sum())} turnovers, {int(clean.blocks.sum())} blocks.")
    else:
        team_line = (f"{lead}: {int(clean.holds.sum())} holds, {int(clean.breaks.sum())} breaks, "
                     f"{int(clean.turns.sum())} turnovers, {int(clean.blocks.sum())} blocks.")
    add("games", f"""
<h2>{'Game' if single_game else 'Games'}</h2>
<p class="muted">Holds are points we started on offence and won, breaks are points we started on defence and won.{' A clean hold is an O point scored on our first possession, without a turnover. Which points started on offence is worked out from the lineups in the per-game player stats and checked against Statto\'s hold and break counts.' if detailed else ''}{' Greyed rows were only partly recorded.' if SUSPECT_GAMES and not single_game else ''}</p>
{games_table(g)}
<p class="muted">{esc(team_line)}</p>""")

    if not full:
        add("games", f"""
<p class="muted">Per-game numbers below come from the pass log. Points played, blocks, points won and +/- need the per-game Players export from Statto, which isn't included yet.</p>""")

    # -- scoring
    def ranked(col, color=SPLASH):
        d = p[p[col] > 0].sort_values([col, "pts" if full else "throws"], ascending=[False, True])
        if full:
            return hbars([(r.short, r[col], r[col] * r.per10) for _, r in d.iterrows()],
                         color=color, note_fmt="({:.1f} per 10 pts)")
        return hbars([(r.short, r[col], None) for _, r in d.iterrows()], color=color)

    add("scoring", f"""
<h2>Scoring</h2>
<p class="muted">Raw counts for everyone with at least one{', with the rate per 10 points played in grey so heavy playing time does not dominate' if full else ''}.</p>
<h3>Goals</h3>
{ranked('goals')}
<h3>Assists</h3>
{ranked('assists')}
<h3>Hockey assists (A2)</h3>
{ranked('a2')}""")
    if full:
        pm = p.sort_values(["plus_minus", "pts"], ascending=[False, True])
        pm_rows = [(r.short, r.plus_minus,
                    f"({r.goals:g}G {r.assists:g}A {r.blocks:g}B − {r.turnovers:g}T" + (f" + {A2_WEIGHT:g}×{r.a2:g}A2)" if A2_WEIGHT else ")"))
                   for _, r in pm.iterrows()]
        add("scoring", f"""
<h3>Blocks</h3>
{ranked('blocks', AMBER)}
<h3>Plus/minus</h3>
<p class="muted">Goals + assists + blocks − turnovers{a2_note()}. All players. It rewards volume, so heavy-touch handlers swing furthest both ways.</p>
{diverging_bars(pm_rows, value_fmt="{:+g}")}""")

    # -- throwing
    thr = p[p["throws"] > 0].sort_values("gain_per_throw", ascending=False)
    gain_svg = hbars([(r.short, r.gain_per_throw, f"{r.throws:g} throws, {r.completion:.0%}") for _, r in thr.iterrows()],
                     value_fmt="{:.1f} m", note_fmt="({})")
    add("throwing", f"""
<h2>Throwing</h2>
<p>Not every throw is equal, so completion percentage alone is misleading: a reset handler at 97% and a cutter throwing hucks at 85% aren't doing the same job. The lists below split throwing into how much field each completed throw gains and how hard the throws were. Everyone is included, so treat the numbers for players with few throws as anecdotes.</p>
<h3>Metres gained per completed throw</h3>
<p class="muted">Average downfield gain on completed throws, from Statto's tapped positions. Throw count and completion rate in grey.</p>
{gain_svg}""")

    if has_passes:
        short = dict(zip(p["Player"], p["short"]))
        nm = lambda x: short.get(x, re.sub(r"^\d+\s+", "", str(x)).split()[0])
        pl = passes.assign(t=passes["Thrower"].map(nm), r=passes["Receiver"].map(nm))

        if "xcp" in pl.columns:
            ce = pl.groupby("t").agg(n=("completed", "size"), comp=("completed", "sum"), exp=("xcp", "sum"), xcp=("xcp", "mean"))
            ce["count"] = ce.comp - ce.exp
            ce["pts"] = 100 * ce["count"] / ce.n
            by_pts = ce.sort_values("pts", ascending=False)
            by_cnt = ce.sort_values("count", ascending=False)
            pts_svg = diverging_bars([(t, r.pts, f"({int(r.comp)}/{int(r.n)}, expected {r.exp:.1f}, xCP {r.xcp:.0%})") for t, r in by_pts.iterrows()],
                                     value_fmt="{:+.1f} pts")
            cnt_svg = diverging_bars([(t, r["count"], f"({int(r.comp)}/{int(r.n)}, expected {r.exp:.1f}, {r.pts:+.1f} pts)") for t, r in by_cnt.iterrows()],
                                     value_fmt="{:+.1f}")
            add("throwing", f"""
<h3>Completion probability over expected (CPOE)</h3>
<p class="muted">How much better a player completes than a typical teammate would on the same throws. For each throw a model fitted on this event's pass logs gives a completion chance from its length, direction and start position, without knowing who threw it. Summing those gives the expected completions, shown in grey next to the actual count. The average of those chances is the player's xCP: high means safe throw selection, low means ambitious. It adjusts for the throw, not the situation, so D-line throwers after a block are held to the same bar as the O-line. Because the model looks at more than distance, it also credits throws that start in awkward spots. Over the whole event Quinn is the clearest case: his throws are longer than the team's (13.9 m against 12.2 m on average), start slightly further from the end zone and a little more often near a sideline, so the model expects about 1.7 fewer completions from him than throw distance alone would, and his CPOE is higher for it.</p>
<p class="muted">The same numbers are shown two ways. Per throw, in percentage points: the player's completion rate minus the expected rate, so 32/32 with an xCP of 93% is +7 pts. This is fair between a handler and a cutter but swings wildly on few throws. As a count: completions minus expected completions, so the same 32/32 is +2.1. This rewards volume and is far less noisy. A reset handler tends to look better per throw, a workhorse better by count. "pts" means percentage points, the plain difference between two rates, not a relative change.</p>
<h3 style="font-size:14px; margin-top:14px;">Per throw</h3>
{pts_svg}
<h3 style="font-size:14px;">As a count</h3>
{cnt_svg}""")
        else:
            ce = pl.groupby("t").agg(n=("completed", "size"), comp=("completed", "sum"), exp=("expected", "sum"))
            ce["coe"] = ce.comp - ce.exp
            ce = ce.sort_values("coe", ascending=False)
            coe_svg = diverging_bars([(t, r.coe, f"({int(r.comp)}/{int(r.n)}, expected {r.exp:.1f})") for t, r in ce.iterrows()])
            add("throwing", f"""
<h3>Completions over expected</h3>
<p class="muted">Each throw is compared with the team completion rate for throws of the same forward distance, and the differences are summed per player. Positive means completing harder throws than a typical teammate would.</p>
{coe_svg}""")

        # connections
        pairs = (pl.groupby(["t", "r"]).agg(n=("completed", "size"), comp=("completed", "sum"))
                   .reset_index().sort_values(["n", "comp"], ascending=False))
        pair_rows = [(f"{a} → {b}", n, f"{c} completed") for a, b, n, c in
                     zip(pairs.t, pairs.r, pairs.n, pairs.comp)][:PAIRS_TO_LIST]
        pairs_svg = hbars(pair_rows, note_fmt="({})", label_w=150, bar_w=260)

        thr_ct, rec_ct = pl.groupby("t").size(), pl.groupby("r").size()
        keep = sorted(set(thr_ct.index) | set(rec_ct.index), key=lambda n: -(thr_ct.get(n, 0) + rec_ct.get(n, 0)))
        agg = pl.groupby(["t", "r"]).agg(n=("completed", "size"), te=("te", "sum"), re=("re", "sum")).reset_index()
        def piv(col):
            return agg.pivot(index="t", columns="r", values=col).reindex(index=keep, columns=keep).fillna(0).astype(int)
        mat, te_m, re_m = piv("n"), piv("te"), piv("re")
        vmax = int(mat.values.max()) or 1
        head = "<th></th>" + "".join(f'<th><span>{esc(c)}</span></th>' for c in keep)
        rows = []
        for t in keep:
            cells = []
            for r in keep:
                v, te, re_ = int(mat.at[t, r]), int(te_m.at[t, r]), int(re_m.at[t, r])
                if not v:
                    cells.append("<td></td>")
                    continue
                bad = te + re_
                tip = f"{t} → {r}: {v} attempts, {v - bad} completed"
                if bad:
                    tip += f", {te} throwaway{'s' if te != 1 else ''}, {re_} drop{'s' if re_ != 1 else ''}"
                sup = f"<sup>{bad}</sup>" if bad else ""
                cells.append(f'<td style="background:rgba(24,119,210,{0.08 + 0.72 * v / vmax:.2f})" title="{esc(tip)}">{v}{sup}</td>')
            rows.append(f'<tr><th>{esc(t)}</th>{"".join(cells)}<td class="tot">{int(mat.loc[t].sum())}</td></tr>')
        matrix = (f'<div class="scroll"><table class="matrix"><thead><tr>{head}<th>Total</th></tr></thead>'
                  f'<tbody>{"".join(rows)}</tbody></table></div>'
                  '<p class="muted matrix-note">Tap a cell to see the split.</p>')

        # favourite targets and throwers per player
        tgt = pl.groupby(["t", "r"]).size().reset_index(name="n")
        order = [nm(x) for x in p.sort_values("throws", ascending=False)["Player"]]
        def fav(col_from, col_to, who):
            d = tgt[tgt[col_from] == who].sort_values("n", ascending=False).head(TOP_TARGETS)
            return ", ".join(f"{r_[col_to]} {int(r_.n)}" for _, r_ in d.iterrows())
        fav_rows = "".join(
            f"<tr><td>{esc(who)}</td><td>{esc(fav('t', 'r', who))}</td><td>{esc(fav('r', 't', who))}</td></tr>"
            for who in order if who in set(tgt.t) | set(tgt.r))
        fav_table = ('<div class="scroll"><table class="fav"><thead><tr><th>Player</th><th>Throws most to</th>'
                     f'<th>Catches most from</th></tr></thead><tbody>{fav_rows}</tbody></table></div>')

        # MMP / FMP breakdown
        mm = pl.assign(tm=pl["Thrower"].map(MATCHING), rm=pl["Receiver"].map(MATCHING))
        unmapped = sorted(set(pl.loc[mm.tm.isna(), "Thrower"]) | set(pl.loc[mm.rm.isna(), "Receiver"]))
        mm = mm.dropna(subset=["tm", "rm"])
        combo = mm.groupby(["tm", "rm"]).agg(n=("completed", "size"), comp=("completed", "mean"), fwd=("fwd", "mean"))
        combo["fwd_c"] = mm[mm["completed"] == 1].groupby(["tm", "rm"])["fwd"].mean()
        combo_rows = "".join(
            f"<tr><td>{a} → {b}</td><td>{int(r_.n)}</td><td>{r_.n / len(mm):.0%}</td><td>{r_.comp:.0%}</td>"
            f"<td>{r_.fwd:+.1f}</td><td>{r_.fwd_c:+.1f}</td></tr>"
            for (a, b), r_ in combo.reindex([("MMP", "MMP"), ("MMP", "FMP"), ("FMP", "MMP"), ("FMP", "FMP")]).dropna().iterrows())
        combo_table = ('<table><thead><tr><th>Thrower → receiver</th><th>Throws</th><th>Share</th><th>Completed</th>'
                       f'<th>Fwd m attempted</th><th>Fwd m completed</th></tr></thead><tbody>{combo_rows}</tbody></table>')
        per_m = mm.groupby(["t", "rm"]).size().unstack(fill_value=0).reindex(columns=["MMP", "FMP"], fill_value=0)
        per_m["tot"] = per_m.sum(axis=1)
        per_m = per_m.sort_values("tot", ascending=False)
        split_svg = stacked_bars([(t, [r_.MMP, r_.FMP]) for t, r_ in per_m.iterrows()])
        unmapped_note = (f' Not in the MMP/FMP list, so left out here: {", ".join(nm(x) for x in unmapped)}.' if unmapped else "")

        kinds = ["long", "upfield", "swing", "dump"]
        tk = pl.groupby("kind").agg(n=("completed", "size"), comp=("completed", "sum"), fwd=("fwd", "mean")).reindex(kinds)
        tk["fwd_c"] = pl[pl["completed"] == 1].groupby("kind")["fwd"].mean()
        tk_rows = "".join(f"<tr><td>{k.capitalize()}</td><td>{int(r.n)}</td><td>{r.comp / r.n:.0%}</td>"
                          f"<td>{r.fwd:+.1f}</td><td>{r.fwd_c:+.1f}</td></tr>"
                          for k, r in tk.iterrows() if r.n > 0)
        team_kind_table = ('<table><thead><tr><th>Throw type</th><th>Throws</th><th>Completed</th>'
                           '<th>Fwd m attempted</th><th>Fwd m completed</th></tr></thead>'
                           f'<tbody>{tk_rows}</tbody></table>')
        per = pl.groupby(["t", "kind"]).agg(n=("completed", "size"), comp=("completed", "sum"))
        pk_rows = []
        for t in thr_ct.sort_values(ascending=False).index:
            cells = []
            for k in kinds:
                if (t, k) in per.index:
                    n, c = per.loc[(t, k)]
                    cells.append(f"<td>{int(c)}/{int(n)}</td>")
                else:
                    cells.append("<td></td>")
            pk_rows.append(f"<tr><td>{esc(t)}</td>{''.join(cells)}<td>{int(thr_ct[t])}</td></tr>")
        per_kind_table = ('<div class="scroll"><table><thead><tr><th>Player</th>'
                          + "".join(f"<th>{k.capitalize()}</th>" for k in kinds)
                          + f'<th>All</th></tr></thead><tbody>{"".join(pk_rows)}</tbody></table></div>')

        add("connections", f"""
<h2>Who throws to whom</h2>
<h3>Most common connections</h3>
<p class="muted">All attempts between a thrower and receiver, completions in grey.</p>
{pairs_svg}
<h3>Pass attempts, thrower by receiver</h3>
<p class="muted">Rows throw, columns catch, everyone included, ordered by touches. The small amber number is how many of those attempts were turnovers. Statto logs whether each was a throwaway or a drop, and tapping a cell shows the split below the table. Scroll sideways on a phone.</p>
{matrix}
<h3>Favourite targets</h3>
<p class="muted">Each player's top {TOP_TARGETS} receivers and top {TOP_TARGETS} throwers, by attempts.</p>
{fav_table}
<h3>MMP and FMP throwing</h3>
<p class="muted">Where the disc goes between matching groups. Share is the fraction of all logged throws. Forward metres are shown twice: averaged over every attempt, which measures how ambitious the throws were, and over completions only, which is the ground actually gained. Turnovers are on average much longer throws than completions, so the two differ.{unmapped_note}</p>
{combo_table}
<p class="legend">Throws per player, most first<i style="background:{SPLASH}"></i>to MMP<i style="background:{AMBER}"></i>to FMP</p>
{split_svg}
<h3>Completion by throw type</h3>
<p class="muted">Categories from the tapped positions. Dump is any pass that goes backwards, however far sideways. Swing is a pass that moves the disc at least 11 m across the field while staying within about 8 m forward or back. Long is any throw gaining {LONG_THROW_M} m or more. Upfield is everything else, from a short give-and-go up to just under {LONG_THROW_M} m. Splitting throws this way makes throwers comparable, since a dump and a long throw are different jobs. Forward metres are positive when the disc moves toward the end zone we are attacking, shown once averaged over every attempt and once over completions only.</p>
{team_kind_table}
<p class="muted">Per player, completed/attempted.</p>
{per_kind_table}""")

    # -- model-based section (Shown Space style)
    if has_passes and "ec_thrower" in passes.columns:
        info = passes.attrs.get("models", {})
        short = dict(zip(p["Player"], p["short"]))
        nm = lambda x: short.get(x, re.sub(r"^\d+\s+", "", str(x)).split()[0])
        pm_ = passes.assign(t=passes["Thrower"].map(nm), r=passes["Receiver"].map(nm))
        thr = pm_.groupby("t").agg(throws=("completed", "size"), ec_t=("ec_thrower", "sum"), xcp=("xcp", "mean"),
                                   comp=("completed", "sum"), xcomp=("xcp", "sum"))
        thr["cpoe"] = thr["comp"] - thr["xcomp"]
        rec = pm_.groupby("r").agg(ec_r=("ec_receiver", "sum"))
        ev = thr.join(rec, how="outer").fillna(0)
        ev["total"] = ev["ec_t"] + ev["ec_r"]
        ev = ev.sort_values("total", ascending=False)
        ec_svg = diverging_bars([(t, r_.total, f"(thrower {r_.ec_t:+.1f}, receiver {r_.ec_r:+.1f})") for t, r_ in ev.iterrows()],
                                value_fmt="{:+.2f}")
        xs_rows = "".join(f"<tr><td>{int(round((y - 0.18) * 100))} m from the end zone we attack</td><td>{mid:.0%}</td><td>{side:.0%}</td></tr>"
                          for y, mid, side in info.get("xs_curve", []))
        xs_table = ('<table><thead><tr><th>Disc position</th><th>Middle</th><th>Sideline</th></tr></thead>'
                    f'<tbody>{xs_rows}</tbody></table>')

        ang = np.degrees(np.arctan2(pm_["lat"], pm_["fwd"]))
        sector = ((ang + 360 / ROSE_SECTORS / 2) % 360 // (360 / ROSE_SECTORS)).astype(int)
        pm_ = pm_.assign(sector=sector)
        order = [nm(x) for x in p.sort_values("throws", ascending=False)["Player"]]
        cards = []
        for who in order:
            th = pm_[pm_["t"] == who]["sector"].value_counts().reindex(range(ROSE_SECTORS), fill_value=0).tolist()
            ca = pm_[(pm_["r"] == who) & (pm_["completed"] == 1)]["sector"].value_counts().reindex(range(ROSE_SECTORS), fill_value=0).tolist()
            if not sum(th) and not sum(ca):
                continue
            cards.append(f'<div class="rose"><div class="rose-name">{esc(who)}</div>{rose_svg(th, ca)}'
                         f'<div class="muted">{sum(th)} throws · {sum(ca)} catches</div></div>')
        roses = ("" if single_game else
                 '<h3>Tendency radials</h3>\n<p class="muted">The direction of each player\'s throws (blue) and of the throws they catch (amber), with forward at the top and the field\'s right on the right. Wedge size grows with the share of throws in that direction, so a deep cutter shows a tall amber wedge and a reset handler a wide blue base.</p>\n'
                 '<div class="roses">' + "".join(cards) + "</div>")

        add("models", f"""
<h2>Expected contribution</h2>
<p>A simplified version of the Shown Space approach used for the UFA. Two models are fitted on the logged games themselves. The first gives the probability that a possession ends in a goal from where the disc is, from {info.get('possessions', 0)} possessions and {info.get('goals', 0)} goals. The second gives the probability that a throw is completed from its length, direction and start position, which is what the CPOE chart in the Throwing section uses. With ten games of data these are rough, so read the rankings as a first look, not a verdict.</p>
<h3>Chance of scoring by disc position</h3>
<p class="muted">Probability that our possession ends in a goal when the disc is at this spot, in the middle of the field or near a sideline.</p>
{xs_table}
<h3>Expected contribution</h3>
<p class="muted">Every completed pass is worth the change in scoring chance from where it started to where it was caught, shared {THROWER_SHARE:.0%} to the thrower and {1 - THROWER_SHARE:.0%} to the receiver. A turnover costs the scoring chance the possession had, charged to the thrower for a throwaway and to the receiver for a drop, and half each when Statto tagged both. Units are goals. A pure position model marks resets slightly negative, since they move the disc away from the end zone, so reset handlers sit lower than their value to the offence.</p>
{ec_svg}
{roses}""")

    # -- receiving
    rc = p[p["catches"] > 0].sort_values("gain_per_catch", ascending=False)
    add("receiving", f"""
<h2>Receiving</h2>
<p class="muted">Average field gained on each catch. High numbers are the deep cutters.</p>
{hbars([(r.short, r.gain_per_catch, r.catches) for _, r in rc.iterrows()], value_fmt="{:.1f} m", note_fmt="({:g} catches)")}""")

    # -- playing time and points won
    if full:
        pt = p.sort_values("pts", ascending=False)
        median_pts = float(p["pts"].median())
        wp = p[p["pts"] > 0].sort_values("win_pct", ascending=False)
        add("playing", f"""
<h2>Playing time</h2>
<p class="legend">Points played, most first, dashed line at the squad median<i style="background:{SPLASH}"></i>offence<i style="background:{AMBER}"></i>defence</p>
{stacked_bars([(r.short, [r.o_pts, r.d_pts]) for _, r in pt.iterrows()], marker=(median_pts, f"median {median_pts:g}"))}

""")
        add("points_won", f"""
<h2>Points won while on the field</h2>
<p class="muted">Share of points we scored with this player on the line. Mostly reflects which line you play on and who you play with, so treat it as a curiosity rather than a rating.</p>
{hbars([(r.short, r.win_pct * 100, f"{r.won:g}/{r.pts:g}") for _, r in wp.iterrows()], value_fmt="{:.0f}%", note_fmt="{}", max_value=100)}""")

    # -- full stat sheet
    cols = [("Player", "short")]
    if full:
        cols += [("Points played", "pts"), ("+/-", "plus_minus")]
    cols += [("Goals", "goals"), ("Assists", "assists"), ("A2", "a2")]
    if full:
        cols += [("Blocks", "blocks")]
    cols += [("Throwaways", "te"), ("Drops", "re"), ("Turnovers", "turnovers")]
    sheet = p.assign(invol=p["goals"] + p["assists"] + p["a2"]).sort_values(["invol", "throws"], ascending=[False, True])
    head = "".join(f"<th>{esc(h)}</th>" for h, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(r[c]) if c == 'short' else f'{r[c]:g}'}</td>" for _, c in cols) + "</tr>"
                   for _, r in sheet.iterrows())
    tot = {c: p[c].sum() for _, c in cols[1:] if c not in ("plus_minus", "pts")}
    tot_line = ", ".join(f"{tot[c]:g} {h if h == 'A2' else h.lower()}" for h, c in cols[1:] if c in tot)
    add("sheet", f"""
<h2>Full stat sheet</h2>
<p class="muted">Every player. Tap a column heading to sort by it. A2 is the hockey assist, the pass before the assist. Turnovers are throwaways plus drops, so a turnover Statto blamed on both players counts once for each.{' +/- is goals + assists + blocks − turnovers' + a2_note() + '.' if full else ''} Sorted by goals + assists + hockey assists.</p>
<div class="scroll"><table class="sheet"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<p class="muted">Team totals: {tot_line}.</p>""")

    body = "\n".join(h for k in SECTION_ORDER for h in sec[k])
    # anchor every section heading and build the contents list
    titles = []
    def anchor(m):
        title = m.group(1)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        titles.append((title, f"{prefix}-{slug}"))
        return f'<h2 id="{prefix}-{slug}">{title}</h2>'
    body = re.sub(r"<h2>([^<]*)</h2>", anchor, body)
    toc = ('<nav class="toc">' + "".join(f'<a href="#{i}">{esc(t)}</a>' for t, i in titles)
           + '<a href="#definitions">Definitions and sources</a></nav>')
    return toc + body


CSS = f"""
:root {{ --ink:{INK}; --splash:{SPLASH}; --amber:{AMBER}; --tint:{TINT}; --mist:{MIST}; --rule:{RULE}; }}
* {{ box-sizing:border-box; }}
html {{ -webkit-text-size-adjust:100%; }}
body {{ margin:0; background:#fff; color:var(--ink); font-family:"Barlow","Helvetica Neue",Arial,sans-serif; font-size:16px; line-height:1.45; }}
main {{ max-width:680px; margin:0 auto; padding:20px 18px 56px; }}
h1 {{ font-size:34px; font-weight:600; letter-spacing:-0.01em; margin:0; line-height:1.1; }}
h1 small {{ display:block; font-size:16px; font-weight:400; color:var(--mist); margin-top:4px; }}
h2 {{ font-size:22px; font-weight:600; margin:44px 0 4px; scroll-margin-top:64px; }}
h3 {{ font-size:16px; font-weight:600; margin:26px 0 6px; }}
p {{ margin:6px 0 14px; max-width:60ch; }}
.muted {{ color:var(--mist); font-size:14px; }}
.strip {{ display:flex; overflow-x:auto; padding:18px 0 8px 18px; margin:0 -18px; scrollbar-width:none; }}
.strip::-webkit-scrollbar {{ display:none; }}
.game {{ flex:0 0 auto; min-width:104px; margin-right:10px; padding:10px 12px 8px; border-radius:6px; background:var(--tint); }}
.game.loss {{ background:#F1F4F7; }}
.game .score {{ font-family:"Barlow Semi Condensed","Barlow",sans-serif; font-size:40px; font-weight:600; line-height:1; letter-spacing:-0.02em; }}
.game .score span {{ color:var(--mist); font-size:24px; font-weight:500; }}
.game .opp {{ font-size:13px; margin-top:6px; white-space:nowrap; }}
.game .meta {{ font-size:12px; color:var(--mist); }}
.record {{ display:flex; flex-wrap:wrap; margin:10px 0 0; }}
.record > div {{ margin:0 28px 8px 0; font-size:13px; color:var(--mist); }}
.record div b {{ display:block; font-size:28px; font-weight:600; line-height:1.1; color:var(--ink); }}
.tabs {{ position:sticky; top:0; z-index:2; background:rgba(255,255,255,0.96); margin:22px -18px 0; padding:8px 18px; border-bottom:1px solid var(--rule); overflow-x:auto; white-space:nowrap; scrollbar-width:none; }}
.tabs::-webkit-scrollbar {{ display:none; }}
.tabs button {{ font:inherit; font-size:14px; color:var(--ink); background:none; border:1px solid var(--rule); border-radius:999px; padding:5px 12px; margin:0 6px 0 0; cursor:pointer; }}
@media (min-width: 600px) {{
  .tabs {{ white-space:normal; overflow:visible; display:flex; flex-wrap:wrap; }}
  .tabs button {{ margin:0 6px 6px 0; }}
}}
.tabs button.on {{ background:var(--ink); color:#fff; border-color:var(--ink); }}
.view {{ display:none; }}
.view.on {{ display:block; }}
.view-title {{ font-size:14px; color:var(--mist); margin:18px 0 0; }}
.toc {{ display:flex; flex-wrap:wrap; margin:14px 0 4px; }}
.toc a {{ color:var(--ink); text-decoration:none; font-size:14px; padding:4px 10px; margin:0 6px 6px 0; background:var(--tint); border-radius:4px; }}
.toc a:hover {{ background:#D3E6F8; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 12px; }}
th, td {{ text-align:right; padding:5px 4px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; padding-left:0; }}
th {{ font-weight:500; color:var(--mist); }}
tr.suspect td {{ color:var(--mist); }}
td .stage {{ display:block; font-size:11px; color:var(--mist); line-height:1.1; }}
.scroll {{ overflow-x:auto; margin:0 -18px; padding:0 18px; }}
.games td, .games th {{ white-space:nowrap; }}
.sheet th {{ cursor:pointer; user-select:none; white-space:normal; line-height:1.2; vertical-align:bottom; }}
.sheet th.on {{ color:var(--ink); border-bottom:2px solid var(--ink); }}
.sheet td:nth-child(even), .sheet th:nth-child(even) {{ background:#F3F6F9; }}
.fav td:first-child {{ white-space:nowrap; }}
.fav td {{ white-space:normal; text-align:left; font-size:13px; }}
.fav th {{ text-align:left; }}
.matrix {{ font-size:12px; }}
.matrix th, .matrix td {{ padding:4px 3px; text-align:center; min-width:26px; border:0; }}
.matrix thead th {{ height:78px; vertical-align:bottom; padding-bottom:6px; }}
.matrix thead th span {{ writing-mode:vertical-rl; transform:rotate(180deg); display:inline-block; }}
.matrix tbody th {{ text-align:left; font-weight:500; color:var(--ink); padding-left:0; white-space:nowrap; }}
.matrix td.tot {{ color:var(--mist); }}
.matrix td sup {{ font-size:9px; color:var(--amber); font-weight:600; margin-left:1px; }}
.matrix td[title] {{ cursor:pointer; }}
.matrix td.picked {{ outline:2px solid var(--ink); outline-offset:-2px; }}
.matrix-note {{ min-height:1.4em; margin-top:8px; }}
.roses {{ margin-top:6px; }}
.rose {{ display:inline-block; vertical-align:top; width:200px; margin:0 8px 14px 0; font-size:12px; }}
.rose-name {{ font-weight:600; margin-bottom:2px; }}
.rose .muted {{ font-size:11px; margin:0; }}
.legend {{ font-size:13px; color:var(--mist); margin:0 0 6px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 5px 0 12px; vertical-align:-1px; }}
svg text {{ font-family:inherit; }}
footer {{ margin-top:56px; padding-top:16px; border-top:1px solid var(--rule); font-size:13px; color:var(--mist); }}
footer p {{ max-width:70ch; }}
footer a {{ color:var(--splash); }}
footer code {{ font-size:12px; }}
"""

JS = """
(function () {
  var tabs = document.querySelectorAll('.tabs button'), views = document.querySelectorAll('.view');
  tabs.forEach(function (b) {
    b.addEventListener('click', function () {
      tabs.forEach(function (x) { x.classList.remove('on'); }); b.classList.add('on');
      views.forEach(function (v) { v.classList.toggle('on', v.dataset.view === b.dataset.view); });
      window.scrollTo(0, 0);
    });
  });
  document.querySelectorAll('table.matrix').forEach(function (t) {
    var note = t.parentNode.nextElementSibling;
    t.addEventListener('click', function (e) {
      var td = e.target.closest('td[title]'); if (!td) return;
      t.querySelectorAll('td.picked').forEach(function (c) { c.classList.remove('picked'); });
      td.classList.add('picked');
      if (note) note.textContent = td.getAttribute('title');
    });
  });
  document.querySelectorAll('table.sheet').forEach(function (t) {
    var ths = t.querySelectorAll('th'), tb = t.tBodies[0], dir = {};
    ths.forEach(function (th, i) {
      th.addEventListener('click', function () {
        var num = i > 0, d = dir[i] = -(dir[i] || (num ? 1 : -1));
        var rows = Array.prototype.slice.call(tb.rows);
        rows.sort(function (a, b) {
          var x = a.cells[i].textContent, y = b.cells[i].textContent;
          return num ? d * (parseFloat(y) - parseFloat(x)) : d * y.localeCompare(x);
        });
        rows.forEach(function (r) { tb.appendChild(r); });
        ths.forEach(function (h) { h.classList.remove('on'); }); th.classList.add('on');
      });
    });
  });
})();
"""


def build(p, g, passes, by_game) -> str:
    wins = int((g["Result"] == "Win").sum())
    losses = len(g) - wins
    game_names = list(g["Opponent"])
    for col in ("o_points", "d_points", "clean_holds"):
        g[col] = pd.NA
    for opp in game_names:
        pi = passes[passes["game"] == opp] if len(passes) else passes
        if opp in by_game and len(pi):
            ps_ = by_game[opp]
            summ = point_summary(ps_, pi)
            if summ["holds_calc"] != int(g.loc[g["Opponent"] == opp, "holds"].iloc[0]):
                print(f"warning: derived holds differ from Statto for {opp}")
            for col in ("o_points", "d_points", "clean_holds"):
                g.loc[g["Opponent"] == opp, col] = summ[col]

    views = [f'<section class="view on" data-view="all">{build_view(p, g, passes, single_game=False, prefix="all")}</section>']
    tabs = ['<button class="on" data-view="all">All games</button>']
    for i, opp in enumerate(game_names):
        gi = g[g["Opponent"] == opp]
        pi = passes[passes["game"] == opp] if len(passes) else passes
        if opp in by_game:
            frame = by_game[opp]
        elif len(pi):
            frame = players_from_passes(pi)
        else:
            continue
        r = gi.iloc[0]
        views.append(f'<section class="view" data-view="g{i}"><p class="view-title">vs {esc(opp)}, {r.us}–{r.them}</p>'
                     f'{build_view(frame, gi, pi, single_game=True, prefix=f"g{i}")}</section>')
        tabs.append(f'<button data-view="g{i}">{esc(opp)}</button>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(TEAM)} — {esc(EVENT_LABEL)} stats</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{esc(TEAM)}<small>{esc(EVENT_LABEL)} · stats from Statto, tagged from the game videos</small></h1>
{score_cards(g)}
<div class="record">
  <div><b>{wins}–{losses}</b>record</div>
  <div><b>{int(g.us.sum())}–{int(g.them.sum())}</b>points for–against</div>
  <div><b>{int(g.breaks.sum())}</b>breaks</div>
</div>
<nav class="tabs">{''.join(tabs)}</nav>
{''.join(views)}
<footer id="definitions">
<h2 style="font-size:18px; margin-top:0;">Definitions and sources</h2>
<p><b>How the data was recorded.</b> Stats tagged in Statto by rewatching the game videos, so they are more complete than live stats but still hand-counted, and small errors are possible. Throw and catch gains are Statto's own distance estimates from the tapped field positions.</p>
<p><b>Box-score stats.</b> Goals, assists, hockey assists (A2, the pass before the assist), blocks, throwaways, drops, holds and breaks follow the usual ultimate definitions, the same as in Statto, Ultiworld and UltiAnalytics. Turnovers in the stat sheet are throwaways plus drops, which is Statto's per-player count. It is higher than the team's turnover count because Statto sometimes tags one turnover as both a throwaway and a drop, when thrower and receiver share the blame. +/- is goals + assists + blocks − turnovers{a2_note()}, a common convention rather than a rule. A clean hold is an O point scored on our first possession. Which points started on offence is reconstructed from the lineups in the per-game player stats and matches Statto's hold and break counts in every game.</p>
<p><b>Throw categories.</b> Swing and dump are Statto's flags. From this data they work out as: dump is any pass that goes backwards, swing moves the disc at least 11 m across the field within about 8 m forward or back. Long (20 m or more forward) is our own threshold, chosen because the completion rate drops sharply there.</p>
<p><b>CPOE (Throwing section).</b> A standard observed-minus-expected construction: each throw's expected completion probability (xCP) comes from a model fitted on this event's pass logs. It is shown per throw, as the player's actual completion rate minus the expected one in percentage points (written pts, a plain difference of two rates, not a relative increase, so 100% against an expected 93% is +7 pts), and as a count, completions minus expected completions, which for the same player is +2.1 over 32 throws. If a player were exactly average, the standard deviation of their completion count over n throws is about √(n·p·(1−p)), roughly 3.5 completions at 130 throws, or 2.7 pts, so small differences are noise.</p>
<p><b>Expected contribution section.</b> Concepts borrowed from <a href="https://shownspace.com/">Shown Space</a>, the analytics project built on UFA data (see their <a href="https://shownspace.substack.com/p/welcome-to-shown-space-your-field">introduction</a> and the <a href="https://www.watchufa.com/league/news/2026-ufa-shown-space-stats-analytics-introduction">UFA article</a>): aEC accumulates the change in the team's expected scoring probability over a player's actions, xCP is a throw's expected completion probability from distance, angle and field position, CPOE is completion rate against that expectation, and tendency radials show throw directions. The models on this page are not theirs. They are two small ridge logistic regressions fitted on this event's own {passes.attrs.get("models", {}).get("possessions", 0) if len(passes) else 0} possessions: scoring chance from distance to the end zone (linear and squared) and distance from the centre line, and completion chance from throw length, forward gain, lateral movement and start position. A completed pass is credited with the change in scoring chance, split {THROWER_SHARE:.0%} thrower and {1 - THROWER_SHARE:.0%} receiver, and a turnover costs the scoring chance the possession had, split when both players were blamed. The opponent's value after a turnover is left out, since there are no defensive events in the log to balance it, so the team's total is about zero by construction. The "over expected" idea itself comes from football analytics, where <a href="https://www.nfeloapp.com/analysis/over-expected-explained-what-are-cpoe-ryoe-and-yacoe/">this explainer</a> makes the fair point that such numbers are model errors and inherit the model's blind spots. Full formulas are in the docstring of <code>fit_models</code> in the build script.</p>
</footer>
</main>
<script>{JS}</script>
</body>
</html>
"""


if __name__ == "__main__":
    players = load_players(PLAYERS_CSV)
    games = load_games()
    pass_log = load_passes()
    by_game = load_players_by_game(games)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build(players, games, pass_log, by_game), encoding="utf-8")
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} kB)")
