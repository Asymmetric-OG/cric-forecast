import pandas as pd
import numpy as np
import torch
import sys
import os
from transformers import PatchTSTConfig, PatchTSTForPrediction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fp_calculation import calculate_fantasy_points

K = 10
BAT_COLS = ['runs', 'balls_faced', 'fours', 'sixes']
BOWL_COLS = ['runs_conceded', 'balls_bowled', 'dot_balls', 'wickets', 'lbw', 'bowled', 'maiden_overs']
FIELD_COLS = ['f_catches', 'f_caught_and_bowled', 'f_stumpings', 'f_run_out']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Loads PyTorch models from disk. (Called once by app.py)"""
    bat_config = PatchTSTConfig(context_length=K, prediction_length=1, num_input_channels=len(BAT_COLS))
    bat_model = PatchTSTForPrediction(bat_config)
    bat_model.load_state_dict(torch.load("models/bat_model.pt", map_location=DEVICE, weights_only=True))
    
    bowl_config = PatchTSTConfig(context_length=K, prediction_length=1, num_input_channels=len(BOWL_COLS))
    bowl_model = PatchTSTForPrediction(bowl_config)
    bowl_model.load_state_dict(torch.load("models/bowl_model.pt", map_location=DEVICE, weights_only=True))
    
    field_config = PatchTSTConfig(context_length=K, prediction_length=1, num_input_channels=len(FIELD_COLS))
    field_model = PatchTSTForPrediction(field_config)
    field_model.load_state_dict(torch.load("models/field_model.pt", map_location=DEVICE, weights_only=True))
    
    return bat_model.to(DEVICE).eval(), bowl_model.to(DEVICE).eval(), field_model.to(DEVICE).eval()

def get_padded_sequence(df, player_id, cols, scaler_dict):
    """Pads missing history with zeros and applies Z-score normalization."""
    if player_id in df.index:
        data = df.loc[[player_id]][cols].tail(K).to_numpy(np.float32)
    else:
        data = np.empty((0, len(cols)), dtype=np.float32)

    if len(data) < K:
        padding = np.zeros((K - len(data), len(cols)), dtype=np.float32)
        data = np.vstack([padding, data]) if len(data) > 0 else padding

    mean, std = np.array(scaler_dict["mean"]), np.array(scaler_dict["std"])
    X_norm = (data - mean) / std
    return torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

def generate_match_predictions(team1_name, team2_name, hist_df, team_registry, player_registry, scalers, models):
    """Prediction engine that inputs k sequences into the model to get k+1"""
    bat_model, bowl_model, field_model = models
    
    # The CSV is indexed by player NAME, not ID
    indexed_df = hist_df.sort_values(['player', 'match_id']).set_index('player')
    
    player_ids = team_registry.get(team1_name, []) + team_registry.get(team2_name, [])
    predictions = []

    with torch.no_grad():
        for raw_pid in player_ids:
            pid = str(raw_pid).strip()
            
            # 1. Translate the ID into the Name using the JSON
            p_name = player_registry.get(pid, player_registry.get(raw_pid, "Unknown")).strip()
            
            row_pred = {'player_id': pid, 'player_name': p_name}
            
            # 2. THE FIX: Search the CSV using the NAME (p_name), not the ID
            X_bat = get_padded_sequence(indexed_df, p_name, BAT_COLS, scalers['bat'])
            pred_bat = (bat_model(past_values=X_bat).prediction_outputs.squeeze().cpu().numpy() * np.array(scalers['bat']['std'])) + np.array(scalers['bat']['mean'])
            for i, col in enumerate(BAT_COLS): row_pred[col] = max(0, float(pred_bat[i]))

            X_bowl = get_padded_sequence(indexed_df, p_name, BOWL_COLS, scalers['bowl'])
            pred_bowl = (bowl_model(past_values=X_bowl).prediction_outputs.squeeze().cpu().numpy() * np.array(scalers['bowl']['std'])) + np.array(scalers['bowl']['mean'])
            for i, col in enumerate(BOWL_COLS): row_pred[col] = max(0, float(pred_bowl[i]))

            X_field = get_padded_sequence(indexed_df, p_name, FIELD_COLS, scalers['field'])
            pred_field = (field_model(past_values=X_field).prediction_outputs.squeeze().cpu().numpy() * np.array(scalers['field']['std'])) + np.array(scalers['field']['mean'])
            for i, col in enumerate(FIELD_COLS): row_pred[col] = max(0, float(pred_field[i]))

            predictions.append(row_pred)

    pred_df = pd.DataFrame(predictions)
    pred_df["strike_rate"] = (pred_df["runs"] * 100 / pred_df["balls_faced"]).replace([np.inf, -np.inf], 0).fillna(0)
    pred_df["overs_bowled"] = (pred_df["balls_bowled"] / 6).astype(int)
    pred_df["economy"] = (pred_df["runs_conceded"] / pred_df["overs_bowled"]).replace([np.inf, -np.inf], 0).fillna(0)
    
    return calculate_fantasy_points(pred_df)

def calculate_player_risk(pred_df, hist_df):
    """Calculates Coefficient of Variation (CV) to determine reliability."""
    metrics = []
    
    # We must iterate through the rows so we have access to the Player Name
    for _, row in pred_df.iterrows():
        pid = row["player_id"]
        p_name = row["player_name"]
        
        # THE FIX: Search the historical CSV using the NAME, not the ID
        player_hist = hist_df[hist_df["player"] == p_name]
        
        cv = (player_hist["fp"].std() / (player_hist["fp"].mean() + 1e-6)) if len(player_hist) > 0 else 0.5
        risk_tag = "Safe" if cv < 0.75 else "Moderate" if cv < 1.0 else "High Risk"
        
        metrics.append({"player_id": pid, "consistency": cv, "risk_tag": risk_tag})
    
    return pd.merge(pred_df, pd.DataFrame(metrics), on="player_id")