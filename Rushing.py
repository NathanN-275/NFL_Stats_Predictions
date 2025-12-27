# RB_weekly_model_with_snaps_and_week_eval.py
# ------------------------------------------------------------
# Predict RB rushing yards, receiving yards, total yards
# using weekly player stats + RB-vs-defense rolling features
# + snap % (from weekly snap counts)
# Evaluate by WEEK (walk-forward), not random split.
# ------------------------------------------------------------

import os
import glob
import pandas as pd
import nfl_data_py as nfl

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# USER SETTINGS
# -----------------------------
SEASONS = [2024, 2025]
DATA_DIR = "."  # folder where nflverse CSVs live

# You can use one combined weekly file 
PLAYER_WEEKLY_FILE = "stats_player_week.csv"

# Snap counts weekly file name varies sometimes; script will try to auto-find if this is None
SNAP_WEEKLY_FILE = None  # e.g. "snap_counts_week.csv"

ROLL_WINDOW_PLAYER = 3     # player form window
ROLL_WINDOW_DEF = 3        # defense vs RB window
RIDGE_ALPHA = 1.0

# -----------------------------
# Helpers
# -----------------------------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def require_cols(df, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}\nAvailable: {list(df.columns)}")

def week_index(df):
    # sortable time key across seasons
    return df["season"].astype(int) * 100 + df["week"].astype(int)

def shifted_roll_mean(df, group_col, value_col, window):
    return (
        df.groupby(group_col)[value_col]
        .shift(1)
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

def find_snap_file(data_dir):
    # try common snap filenames in nflverse-data
    patterns = [
        "snap_counts_week*.csv",
        "snap_counts_weekly*.csv",
        "snaps_week*.csv",
        "*snap*week*.csv",
    ]
    hits = []
    for p in patterns:
        hits.extend(glob.glob(os.path.join(data_dir, p)))
    hits = sorted(set(hits))
    return hits[0] if hits else None

# -----------------------------
# LOAD WEEKLY PLAYER STATS
# -----------------------------
player_path = os.path.join(DATA_DIR, PLAYER_WEEKLY_FILE)
if not os.path.exists(player_path):
    raise FileNotFoundError(
        f"Could not find {player_path}\n"
        f"Put '{PLAYER_WEEKLY_FILE}' in {DATA_DIR} or change PLAYER_WEEKLY_FILE."
    )

print(f"Loading weekly player stats: {player_path}")
weekly = pd.read_csv(player_path)
print(f"Weekly rows: {len(weekly):,}")
print("Weekly columns (sample):", list(weekly.columns)[:30])

# Map columns (weekly stats file can vary slightly)
season_col   = pick_col(weekly, ["season", "year"])
week_col     = pick_col(weekly, ["week"])
team_col     = pick_col(weekly, ["team", "recent_team", "posteam"])
opp_col      = pick_col(weekly, ["opponent_team", "opponent", "opp", "defteam"])
pos_col      = pick_col(weekly, ["position", "pos"])
name_col     = pick_col(weekly, ["player_name", "name", "full_name", "player_display_name"])
id_col       = pick_col(weekly, ["player_id", "gsis_id", "esb_id", "pfr_id", "sportradar_id"])

rush_att_col = pick_col(weekly, ["rushing_attempts", "rush_attempts", "carries", "rush_att"])
rush_yds_col = pick_col(weekly, ["rushing_yards", "rush_yards", "rush_yds"])
rec_yds_col  = pick_col(weekly, ["receiving_yards", "rec_yards", "rec_yds"])
targets_col  = pick_col(weekly, ["targets", "tgt"])
recs_col     = pick_col(weekly, ["receptions", "rec"])

require_cols(weekly, [season_col, week_col, team_col, pos_col, name_col, rush_att_col, rush_yds_col, rec_yds_col], "weekly player stats")

# Keep seasons requested
weekly = weekly[weekly[season_col].isin(SEASONS)].copy()

# Filter to RBs
weekly = weekly[weekly[pos_col] == "RB"].copy()

# Create stable player_id if missing
if id_col is None:
    weekly["player_id"] = weekly[name_col].astype(str) + "_" + weekly[team_col].astype(str)
    id_col = "player_id"

# Normalize names
weekly = weekly.rename(columns={
    season_col: "season",
    week_col: "week",
    team_col: "team",
    name_col: "player_name",
    id_col: "player_id",
})
if opp_col is not None:
    weekly = weekly.rename(columns={opp_col: "opponent"})
else:
    # If you don't have opponent, defense features won't work well
    weekly["opponent"] = "UNK"

# Numeric stat columns
weekly["rush_att"] = pd.to_numeric(weekly[rush_att_col], errors="coerce").fillna(0)
weekly["rush_yds"] = pd.to_numeric(weekly[rush_yds_col], errors="coerce").fillna(0)
weekly["rec_yds"]  = pd.to_numeric(weekly[rec_yds_col], errors="coerce").fillna(0)
weekly["targets"]  = (
    pd.to_numeric(weekly[targets_col], errors="coerce").fillna(0)
    if targets_col is not None
    else (pd.to_numeric(weekly[recs_col], errors="coerce").fillna(0) if recs_col is not None else 0)
)
weekly["total_yards"] = weekly["rush_yds"] + weekly["rec_yds"]

# RB1/RB2 proxy by rush attempts
weekly["rb_rank"] = (
    weekly.groupby(["season", "week", "team"])["rush_att"]
    .rank(method="first", ascending=False)
)
rb = weekly[weekly["rb_rank"] <= 2].copy()

# -----------------------------
# LOAD SNAP COUNTS + MERGE SNAP %
# -----------------------------
snap_path = os.path.join(DATA_DIR, SNAP_WEEKLY_FILE) if SNAP_WEEKLY_FILE else find_snap_file(DATA_DIR)
if snap_path is None or not os.path.exists(snap_path):
    print("\nWARNING: Could not find a weekly snap counts file in DATA_DIR.")
    print("Snap % will be set to 0. If you have it, set SNAP_WEEKLY_FILE to the filename.\n")
    rb["snap_pct"] = 0.0
else:
    print(f"Loading snap counts: {snap_path}")
    snaps = pd.read_csv(snap_path)
    print(f"Snap rows: {len(snaps):,}")
    print("Snap columns (sample):", list(snaps.columns)[:30])

    s_season = pick_col(snaps, ["season", "year"])
    s_week   = pick_col(snaps, ["week"])
    s_team   = pick_col(snaps, ["team", "posteam", "recent_team"])
    s_pos    = pick_col(snaps, ["position", "pos"])
    s_name   = pick_col(snaps, ["player", "player_name", "name", "full_name", "player_display_name"])
    s_id     = pick_col(snaps, ["player_id", "gsis_id", "esb_id", "pfr_id", "sportradar_id"])

    # Offense snap pct columns vary a bit
    s_snap_pct = pick_col(snaps, ["offense_snap_pct", "off_snap_pct", "snap_pct", "off_pct", "pct_offense_snaps"])

    require_cols(snaps, [s_season, s_week, s_team, s_pos], "snap counts")

    snaps = snaps[snaps[s_season].isin(SEASONS)].copy()
    snaps = snaps[snaps[s_pos] == "RB"].copy()

    # Normalize
    snaps = snaps.rename(columns={
        s_season: "season",
        s_week: "week",
        s_team: "team",
    })

    # Create a player_id in snaps matching rb["player_id"] if possible
    # Prefer true id; otherwise fallback to name+team
    if s_id is not None:
        snaps = snaps.rename(columns={s_id: "player_id"})
    else:
        # fallback uses name col
        if s_name is None:
            print("\nWARNING: Snap file has no player id or name. Snap % will be 0.\n")
            rb["snap_pct"] = 0.0
        else:
            snaps["player_id"] = snaps[s_name].astype(str) + "_" + snaps["team"].astype(str)

    if "player_id" in snaps.columns and s_snap_pct is not None:
        snaps["snap_pct"] = pd.to_numeric(snaps[s_snap_pct], errors="coerce").fillna(0.0)

        snap_keep = snaps[["season", "week", "team", "player_id", "snap_pct"]].drop_duplicates()
        rb = rb.merge(snap_keep, on=["season", "week", "team", "player_id"], how="left")
        rb["snap_pct"] = rb["snap_pct"].fillna(0.0)
    else:
        print("\nWARNING: Could not find a snap % column in snap file. Snap % will be 0.\n")
        rb["snap_pct"] = 0.0

# -----------------------------
# DEFENSE vs RB FEATURES (rolling allowed)
# Build a defense-week table from RB weekly outcomes vs that defense.
# -----------------------------
# This requires opponent to be meaningful. If opponent == "UNK", these features will be weak.
def_rb_week = (
    rb.groupby(["season", "week", "opponent"], as_index=False)
      .agg(
          rb_rush_yds_allowed=("rush_yds", "sum"),
          rb_rec_yds_allowed=("rec_yds", "sum"),
          rb_total_yds_allowed=("total_yards", "sum"),
          rb_rush_att_allowed=("rush_att", "sum"),
          rb_targets_allowed=("targets", "sum"),
      )
      .rename(columns={"opponent": "defteam"})
)

# Rolling allowed features by defense
def_rb_week = def_rb_week.sort_values(["defteam", "season", "week"]).reset_index(drop=True)
def_rb_week["tkey"] = week_index(def_rb_week)

for col in ["rb_rush_yds_allowed", "rb_rec_yds_allowed", "rb_total_yds_allowed", "rb_rush_att_allowed", "rb_targets_allowed"]:
    def_rb_week[f"{col}_avg_{ROLL_WINDOW_DEF}"] = shifted_roll_mean(def_rb_week, "defteam", col, ROLL_WINDOW_DEF)

# Merge defense rolling features into player-week rb table
rb_games = rb.copy()
rb_games["tkey"] = week_index(rb_games)

rb_games = rb_games.merge(
    def_rb_week[[
        "season", "week", "defteam",
        f"rb_rush_yds_allowed_avg_{ROLL_WINDOW_DEF}",
        f"rb_rec_yds_allowed_avg_{ROLL_WINDOW_DEF}",
        f"rb_total_yds_allowed_avg_{ROLL_WINDOW_DEF}",
        f"rb_rush_att_allowed_avg_{ROLL_WINDOW_DEF}",
        f"rb_targets_allowed_avg_{ROLL_WINDOW_DEF}",
    ]].rename(columns={"defteam": "opponent"}),
    on=["season", "week", "opponent"],
    how="left"
)

for c in [
    f"rb_rush_yds_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_rec_yds_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_total_yds_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_rush_att_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_targets_allowed_avg_{ROLL_WINDOW_DEF}",
]:
    rb_games[c] = rb_games[c].fillna(0.0)

# -----------------------------
# PLAYER FORM ROLLING FEATURES (shifted)
# -----------------------------
rb_games = rb_games.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

for col in ["rush_yds", "rec_yds", "total_yards", "rush_att", "targets", "snap_pct"]:
    rb_games[f"{col}_avg_{ROLL_WINDOW_PLAYER}"] = shifted_roll_mean(rb_games, "player_id", col, ROLL_WINDOW_PLAYER)

# Drop rows without enough player history (and optionally defense history)
need = [f"rush_yds_avg_{ROLL_WINDOW_PLAYER}", f"rec_yds_avg_{ROLL_WINDOW_PLAYER}"]
rb_games = rb_games.dropna(subset=need).copy()

print(f"\nTraining rows after rolling history: {len(rb_games):,}")

# -----------------------------
# FEATURES / TARGETS
# -----------------------------
features = [
    # player form
    f"rush_yds_avg_{ROLL_WINDOW_PLAYER}",
    f"rec_yds_avg_{ROLL_WINDOW_PLAYER}",
    f"total_yards_avg_{ROLL_WINDOW_PLAYER}",
    f"rush_att_avg_{ROLL_WINDOW_PLAYER}",
    f"targets_avg_{ROLL_WINDOW_PLAYER}",
    f"snap_pct_avg_{ROLL_WINDOW_PLAYER}",
    "rb_rank",

    # defense vs RB (rolling allowed)
    f"rb_rush_yds_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_rec_yds_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_total_yds_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_rush_att_allowed_avg_{ROLL_WINDOW_DEF}",
    f"rb_targets_allowed_avg_{ROLL_WINDOW_DEF}",
]

X_all = rb_games[features].copy()
y_rush  = rb_games["rush_yds"].copy()
y_rec   = rb_games["rec_yds"].copy()
y_total = rb_games["total_yards"].copy()

# -----------------------------
# 6) EVALUATE BY WEEK (WALK-FORWARD)
#    Train on all weeks < test_week, test on that week.
# -----------------------------
print("\nWalk-forward evaluation by week...")

rb_games_eval = rb_games.copy()
rb_games_eval["tkey"] = week_index(rb_games_eval)

all_tkeys = sorted(rb_games_eval["tkey"].unique())
min_train_tkey = all_tkeys[0]
# Start testing after you have at least a little training history
start_test_idx = max(5, 0)  # you can increase this if you want more training before evaluation

rows = []
for i in range(start_test_idx, len(all_tkeys)):
    test_tkey = all_tkeys[i]
    train = rb_games_eval[rb_games_eval["tkey"] < test_tkey]
    test  = rb_games_eval[rb_games_eval["tkey"] == test_tkey]

    if len(train) < 200 or len(test) < 10:
        continue

    X_train = train[features]
    X_test  = test[features]

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)

    m_rush  = Ridge(alpha=RIDGE_ALPHA)
    m_rec   = Ridge(alpha=RIDGE_ALPHA)
    m_total = Ridge(alpha=RIDGE_ALPHA)

    m_rush.fit(X_train_s, train["rush_yds"])
    m_rec.fit(X_train_s, train["rec_yds"])
    m_total.fit(X_train_s, train["total_yards"])

    pr = m_rush.predict(X_test_s)
    pc = m_rec.predict(X_test_s)
    pt = m_total.predict(X_test_s)

    mae_r = mean_absolute_error(test["rush_yds"], pr)
    mae_c = mean_absolute_error(test["rec_yds"], pc)
    mae_t = mean_absolute_error(test["total_yards"], pt)

    season = int(test["season"].iloc[0])
    week = int(test["week"].iloc[0])

    rows.append({"season": season, "week": week, "n": len(test), "mae_rush": mae_r, "mae_rec": mae_c, "mae_total": mae_t})

eval_df = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)

if len(eval_df) == 0:
    print("No evaluation weeks produced (not enough training/testing rows).")
else:
    print("\nPer-week MAE (last 10 rows):")
    print(eval_df.tail(10).to_string(index=False))
    print("\nOverall MAE:")
    print("Rush :", round(eval_df["mae_rush"].mean(), 2))
    print("Rec  :", round(eval_df["mae_rec"].mean(), 2))
    print("Total:", round(eval_df["mae_total"].mean(), 2))

# -----------------------------
# TRAIN FINAL MODELS ON ALL BUT LATEST WEEK, PREDICT LATEST WEEK
# -----------------------------
print("\nTraining final models and projecting latest week in file...")

# week 18 of 2025 prediction
TARGET_WEEK = 18
TARGET_SEASON = 2025

# train on all real data you have
train = rb_games_eval.copy()

# use each player's most recent real game as the "week 18 input"
latest = (
    rb_games_eval
    .sort_values(["player_id", "week"])
    .groupby("player_id")
    .tail(1)
    .copy()
)


try:
    rosters = nfl.import_rosters([TARGET_SEASON])
    # rosters uses gsis_id; your df uses player_id (you already normalized to "player_id")
    roster_team = (
        rosters[["gsis_id", "team"]]
        .dropna()
        .drop_duplicates("gsis_id")
        .rename(columns={"gsis_id": "player_id", "team": "team_roster"})
    )
    latest = latest.merge(roster_team, on="player_id", how="left")
    latest["team"] = latest["team_roster"].fillna(latest["team"])
    latest = latest.drop(columns=["team_roster"])
except Exception as e:
    print("WARNING: roster team update failed (continuing):", e)

# 2) Set OPPONENT from the actual Week 18 schedule (fixes mixed-up opponents)
sched = nfl.import_schedules([TARGET_SEASON])

# keep regular season, target week
wk = sched[(sched["season"] == TARGET_SEASON) & (sched["week"] == TARGET_WEEK)]
if "game_type" in wk.columns:
    wk = wk[wk["game_type"].isin(["REG", "R"])]

home_map = dict(zip(wk["home_team"], wk["away_team"]))
away_map = dict(zip(wk["away_team"], wk["home_team"]))

latest["opponent"] = latest["team"].map(home_map).fillna(latest["team"].map(away_map)).fillna("UNK")

latest["week"] = TARGET_WEEK
latest["season"] = TARGET_SEASON
latest["tkey"] = TARGET_SEASON * 100 + TARGET_WEEK

X_train = train[features]
X_latest = latest[features]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_latest_s = scaler.transform(X_latest)

rush_model  = Ridge(alpha=RIDGE_ALPHA).fit(X_train_s, train["rush_yds"])
rec_model   = Ridge(alpha=RIDGE_ALPHA).fit(X_train_s, train["rec_yds"])
total_model = Ridge(alpha=RIDGE_ALPHA).fit(X_train_s, train["total_yards"])

latest["proj_rush_yds"]  = rush_model.predict(X_latest_s)
latest["proj_rec_yds"]   = rec_model.predict(X_latest_s)
latest["proj_total_yds"] = total_model.predict(X_latest_s)

for c in ["proj_rush_yds", "proj_rec_yds", "proj_total_yds"]:
    latest[c] = latest[c].clip(lower=0).round(1)

output = latest[[
    "season", "week", "team", "opponent", "rb_rank", "player_name",
    "snap_pct", f"snap_pct_avg_{ROLL_WINDOW_PLAYER}",
    "proj_rush_yds", "proj_rec_yds", "proj_total_yds"
]].copy()

output = output.sort_values(["team", "proj_total_yds"], ascending=[True, False]).reset_index(drop=True)

def ansi(s, code):
    return f"\033[{code}m{s}\033[0m"

def format_table_with_colors(df, high_total=100.0):
    # Column order you want (like the bottom table)
    cols = list(df.columns)

    # Format everything as strings first (so we can compute widths)
    def fmt_value(col, v):
        if pd.isna(v):
            return ""
        # numeric formatting
        if col in {"snap_pct", "snap_pct_avg_3"}:
            try:
                return f"{float(v):.1f}"
            except:
                return str(v)
        if col.startswith("proj_"):
            try:
                return f"{float(v):.1f}"
            except:
                return str(v)
        if col == "rb_rank":
            try:
                return f"{float(v):.1f}"
            except:
                return str(v)
        return str(v)

    str_rows = []
    for _, r in df.iterrows():
        row = {c: fmt_value(c, r[c]) for c in cols}
        str_rows.append(row)

    # Compute widths (based on uncolored strings)
    widths = {}
    for c in cols:
        widths[c] = max(len(c), max(len(row[c]) for row in str_rows))

    # Alignment: numeric right, text left
    numeric_cols = set([c for c in cols if c.startswith("proj_")] + ["season", "week", "rb_rank", "snap_pct", "snap_pct_avg_3"])
    def pad(col, text):
        w = widths[col]
        return text.rjust(w) if col in numeric_cols else text.ljust(w)

    # Header
    header = " ".join(pad(c, c) for c in cols)
    line = "-" * len(header)

    # Colors
    GREEN = "92"   # RB1
    CYAN  = "96"   # RB2
    YELLOW_BG = "30;103"  # black text, yellow background
    DIM = "90"

    out_lines = [header, line]
    prev_team = None

    for row in str_rows:
        rb_rank = row.get("rb_rank")
        team = row.get("team")
        if prev_team is not None and team != prev_team:
            out_lines.append("")
        prev_team = team
        try:
            rb_rank_f = float(rb_rank)
        except:
            rb_rank_f = None

        # Base row color by RB rank
        base_code = None
        if rb_rank_f == 1.0:
            base_code = GREEN
        elif rb_rank_f == 2.0:
            base_code = CYAN

        # High total highlight (overrides for the total cell only)
        try:
            total_val = float(row.get("proj_total_yds", "0") or 0)
        except:
            total_val = 0.0

        parts = []
        for c in cols:
            txt = pad(c, row[c])

            # Dim zeros in snap columns (optional)
            if c in {"snap_pct", "snap_pct_avg_3"} and row[c] in {"0.0", "0"}:
                txt = ansi(txt, DIM)

            # Highlight high totals
            if c == "proj_total_yds" and total_val >= high_total:
                txt = ansi(txt, YELLOW_BG)

            # Apply RB color to the whole row *except* highlighted cell already colored
            if base_code and not (c == "proj_total_yds" and total_val >= high_total):
                txt = ansi(txt, base_code)

            parts.append(txt)

        out_lines.append(" ".join(parts))

    return "\n".join(out_lines)

print("\n=== RB Projections (latest week in file) ===")

# Make sure columns are in the "bottom table" order you want
output = output[[
    "season", "week", "team", "opponent", "rb_rank", "player_name",
    "snap_pct", f"snap_pct_avg_{ROLL_WINDOW_PLAYER}",
    "proj_rush_yds", "proj_rec_yds", "proj_total_yds"
]].copy()

# If your rolling window is 3, rename to match the display function
output = output.rename(columns={f"snap_pct_avg_{ROLL_WINDOW_PLAYER}": "snap_pct_avg_3"})

print("\n=== RB Projections (latest week in file) ===\n")
print(format_table_with_colors(output, high_total=100.0))


output.to_csv("rb_projections.csv", index=False)