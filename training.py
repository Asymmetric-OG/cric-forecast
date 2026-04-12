import pandas as pd
import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction
from torch.optim import AdamW
from tqdm import tqdm

K = 10  
PRED_LEN = 1 
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

BAT_COLS = ['runs', 'balls_faced', 'fours', 'sixes']
BOWL_COLS = ['runs_conceded', 'balls_bowled', 'dot_balls', 'wickets', 'lbw', 'bowled', 'maiden_overs']
FIELD_COLS = ['f_catches', 'f_caught_and_bowled', 'f_stumpings', 'f_run_out']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def create_sequences(df, cols):
    """Slices player histories into sliding windows of length K (input) + 1 (target)"""

    X, Y = [], []
    for _, player_df in df.groupby('player'):
        data = player_df[cols].to_numpy(dtype=np.float32)
        if len(data) > K:
            for i in range(len(data) - K):
                X.append(data[i : i+K])
                Y.append(data[i+K])
    return np.array(X), np.array(Y)

def scale_data(X, Y, name, scalers_dict):
    """Calculates Z-scores, scales the data, and saves to the dictionary"""
   
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    std[std == 0] = 1e-6  # Prevent division by zero
    
    X_scaled = (X - mean) / std
    Y_scaled = (Y - mean) / std
    
    scalers_dict[name] = {"mean": mean.tolist(), "std": std.tolist()}
    return X_scaled, Y_scaled

class T20Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_model(model, dataloader, model_name):
    """Normal pytorch training loop, num(epochs) for each model"""
    print(f"\n--- Training {model_name.upper()} Model ---")
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(past_values=batch_x, future_values=batch_y.unsqueeze(1))
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(dataloader):.4f}")
        
    save_path = f"models/{model_name}_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved {model_name} model to {save_path}")

# Main function for first time execution
def main():
    print(f"Using device: {DEVICE}")
    print("Loading data...")
    df = pd.read_csv("data/t20_master_aggregated.csv")
    df = df.sort_values(['player', 'match_id'])
    
    scalers = {}

    # BATTING
    X_bat, Y_bat = create_sequences(df, BAT_COLS)
    X_bat, Y_bat = scale_data(X_bat, Y_bat, 'bat', scalers)
    bat_loader = DataLoader(T20Dataset(X_bat, Y_bat), batch_size=BATCH_SIZE, shuffle=True)
    
    bat_config = PatchTSTConfig(context_length=K, prediction_length=PRED_LEN, num_input_channels=len(BAT_COLS))
    bat_model = PatchTSTForPrediction(bat_config)
    train_model(bat_model, bat_loader, "bat")

    # BOWLING
    X_bowl, Y_bowl = create_sequences(df, BOWL_COLS)
    X_bowl, Y_bowl = scale_data(X_bowl, Y_bowl, 'bowl', scalers)
    bowl_loader = DataLoader(T20Dataset(X_bowl, Y_bowl), batch_size=BATCH_SIZE, shuffle=True)
    
    bowl_config = PatchTSTConfig(context_length=K, prediction_length=PRED_LEN, num_input_channels=len(BOWL_COLS))
    bowl_model = PatchTSTForPrediction(bowl_config)
    train_model(bowl_model, bowl_loader, "bowl")

    # FIELDING
    X_field, Y_field = create_sequences(df, FIELD_COLS)
    X_field, Y_field = scale_data(X_field, Y_field, 'field', scalers)
    field_loader = DataLoader(T20Dataset(X_field, Y_field), batch_size=BATCH_SIZE, shuffle=True)
    
    field_config = PatchTSTConfig(context_length=K, prediction_length=PRED_LEN, num_input_channels=len(FIELD_COLS))
    field_model = PatchTSTForPrediction(field_config)
    train_model(field_model, field_loader, "field")

    # Scalers for denormalization later on
    with open("data/scalers.json", "w") as f:
        json.dump(scalers, f, indent=4)
    print("\n✅ Saved normalization scalers to data/scalers.json")
    print("🎉 All training complete! You can now run `streamlit run app.py`")

if __name__ == "__main__":
    main()