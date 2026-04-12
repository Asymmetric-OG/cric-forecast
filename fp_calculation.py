import pandas as pd
import numpy as np

def calculate_fantasy_points(df):

    df["batting_fp"] = 0.0
    df["bowing_fp"] = 0.0
    df["fielding_fp"] = 0.0
    df["fp"] = 0.0

    # Batting Points
    df["batting_fp"] += df["runs"] * 1
    df["batting_fp"] += df["fours"] * 4
    df["batting_fp"] += df["sixes"] * 6

    df["batting_fp"] += np.where(df["runs"] >= 100, 16,
                                np.where(df["runs"] >= 75, 12,
                                np.where(df["runs"] >= 50, 8,
                                np.where(df["runs"] >= 25, 4, 0))))

    df["batting_fp"] += np.where((df["runs"] == 0) & (df["balls_faced"] > 0), -2, 0)

    mask_sr = (df["balls_faced"] >= 10) & (df["balls_bowled"] == 0)

    sr_bonus_values = np.where(df["strike_rate"] > 170, 6,
                               np.where(df["strike_rate"] > 150, 4,
                               np.where(df["strike_rate"] >= 130, 2,
                               np.where(df["strike_rate"] >= 60, 0,
                               np.where(df["strike_rate"] >= 50, -2,
                               np.where(df["strike_rate"] >= 0, -4, 0))))))
    df.loc[mask_sr, "batting_fp"] += sr_bonus_values[mask_sr]


    # Bowling Points
    df["bowing_fp"] += df["wickets"] * 30

    df["bowing_fp"] += (df["lbw"] + df["bowled"]) * 8

    df["bowing_fp"] += np.where(df["wickets"] >= 5, 12,
                                np.where(df["wickets"] == 4, 8,
                                np.where(df["wickets"] == 3, 4, 0)))

    if "dot_balls" in df.columns:
        df["bowing_fp"] += df["dot_balls"] * 1

    if "maiden_overs" in df.columns:
        df["bowing_fp"] += df["maiden_overs"] * 12

    mask_economy = df["overs_bowled"] >= 2

    economy_bonus_values = np.where( df["economy"] < 5, 6,
                                           np.where(df["economy"] < 6, 4,
                                           np.where(df["economy"] < 7, 2,
                                           np.where(df["economy"] >= 12, -6,
                                           np.where(df["economy"] >= 11, -4,
                                           np.where(df["economy"] >= 10, -2, 0))))))
    df.loc[mask_economy, "bowing_fp"] += economy_bonus_values[mask_economy]


    #Fielding Points

    df["fielding_fp"] += df["f_catches"] * 8
    df["fielding_fp"] += df["f_caught_and_bowled"] * 8
    df["fielding_fp"] += df["f_stumpings"] * 12
    df["fielding_fp"] += df["f_run_out"] * 6

    df["fielding_fp"] += np.where(df["f_catches"] >= 3, 4, 0)

    df.drop(columns=["overs_bowled"], inplace=True, errors='ignore')
    df["fp"] = df["batting_fp"] + df["bowing_fp"] + df["fielding_fp"]

    return df