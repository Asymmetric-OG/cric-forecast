import glob
import os   
import json
import pandas as pd
import numpy as np

match_files = glob.glob(r"D:\Amogh\cricpred\data\t20_master\*.json")

def load_match(path):
    with open(path) as f:
        return json.load(f)
matches = {}

for f in match_files:
    data = load_match(f)
    data["match_id"] = os.path.basename(f)
    matches[os.path.basename(f)] = data

def extract_balls(match_json):
    match_id = match_json["match_id"]
    meta = match_json["info"]

    rows = []
    for inning_i, inning in enumerate(match_json["innings"], start=1):
        batting_team = inning["team"]
        for over in inning["overs"]:
            over_number = over["over"]
            for ball in over["deliveries"]:
                row = {
                    "match_id": match_id,
                    "inning": inning_i,
                    "over": over_number,
                    "batter": ball.get("batter"),
                    "bowler": ball.get("bowler"),
                    "runs_batter": ball["runs"]["batter"],
                    "runs_extras": ball["runs"]["extras"],
                    "runs_total": ball["runs"]["total"],
                    "wicket_type": None,
                    "wicket_player": None,
                    "batting_team": batting_team,
                    "wicket_fielder": None,
                }

                if "wickets" in ball:
                    w = ball["wickets"][0]
                    row["wicket_type"] = w["kind"]
                    row["wicket_player"] = w["player_out"]
                    fielder_name = None

                    if "fielders" in w and len(w["fielders"]) > 0:
                        f = w["fielders"][0]

                        if isinstance(f, str):
                            fielder_name = f

                        elif isinstance(f, dict) and "name" in f:
                            fielder_name = f["name"]

                    if row["wicket_type"] == "caught and bowled":
                        if fielder_name is None:
                            fielder_name = row["bowler"]

                    row["wicket_fielder"] = fielder_name

                rows.append(row)
    return rows
all_rows = []
for name, match_json in matches.items():
    all_rows += extract_balls(match_json)
balls = pd.DataFrame(all_rows)

batting = balls.groupby(["match_id", "batter"]).agg(
    runs=("runs_batter", "sum"),
    balls_faced=("runs_batter", "count"),
    fours=("runs_batter", lambda x: (x == 4).sum()),
    sixes=("runs_batter", lambda x: (x == 6).sum()),
)
non_bowler_wickets = ['run out', 'retired hurt', 'retired out',  'obstructing the field']
bowling = balls.groupby(["match_id", "bowler"]).agg(
    runs_conceded=("runs_total", "sum"),
    balls_bowled=("runs_total", "count"),
    dot_balls = ("runs_total", lambda x : (x==0).sum()),
    wickets=("wicket_type", lambda x : (~x.isin(non_bowler_wickets) & x.notna()).sum()),
    lbw = ("wicket_type", lambda x : (x=='lbw').sum()),
    bowled = ("wicket_type", lambda x : (x=='bowled').sum()),
)
maidens = (
    balls.groupby(["match_id", "bowler", "inning", "over"])
         .runs_total.sum()
         .eq(0)
         .groupby(level=[0, 1])
         .sum()
)
bowling["maiden_overs"] = maidens.fillna(0).astype(int)
cols = list(bowling.columns)
cols.remove("maiden_overs")
insert_pos = cols.index("dot_balls") + 1
cols.insert(insert_pos, "maiden_overs")
bowling = bowling[cols]

fielding = balls.groupby(["match_id", "wicket_fielder"]).agg(
    f_catches=("wicket_type", lambda x : (x=='caught').sum()),
    f_caught_and_bowled=("wicket_type", lambda x : (x=='caught and bowled').sum()),
    f_stumpings=("wicket_type", lambda x : (x=='stumped').sum()),
    f_run_out = ("wicket_type", lambda x : (x=='run out').sum()),
)

battingn = batting.reset_index()
bowlingn = bowling.reset_index()
fieldingn = fielding.reset_index()
battingn = battingn.rename(columns={"batter": "player"})
bowlingn = bowlingn.rename(columns={"bowler": "player"})
fieldingn = fieldingn.rename(columns={"wicket_fielder": "player"})

player_match = pd.merge(battingn, bowlingn,
                        on=["match_id", "player"],
                        how="outer")

player_match = pd.merge(player_match, fieldingn,
                        on=["match_id", "player"],
                        how="outer")

player_match = player_match.fillna({
    "runs": 0,
    "balls_faced": 0,
    "fours": 0,
    "sixes": 0,
    "runs_conceded": 0,
    "balls_bowled": 0,
    "wickets": 0,
    "lbw":0,
    "bowled":0,
    "dot_balls":0,
    "maiden_overs":0,
    "f_catches": 0,
    "f_caught_and_bowled": 0,
    "f_stumpings": 0,
    "f_run_out": 0
})
player_match["match_id"] = (
    player_match["match_id"]
    .astype(str)
    .str.replace(".json", "", regex=False)
)
player_match["match_id"] = player_match["match_id"].astype(int)
player_match = player_match.sort_values("match_id", ascending=False).reset_index(drop=True)
player_match["strike_rate"] = (
    (player_match["runs"] * 100 / player_match["balls_faced"])
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)
player_match["overs_bowled"] = (player_match["balls_bowled"] / 6).astype(int)
player_match["economy"] = (
    (player_match["runs_conceded"] / player_match["overs_bowled"])
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

from fp_calculation import calculate_fantasy_points

player_match = calculate_fantasy_points(player_match)
os.makedirs(r"D:\Amogh\cricpred\data", exist_ok=True)
player_match.to_csv(r"D:\Amogh\cricpred\data\t20_master_aggregated.csv", index=False)
print("Data successfully preprocessed and saved to CSV!")
