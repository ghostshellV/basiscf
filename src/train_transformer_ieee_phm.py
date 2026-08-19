import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ieee_phm.Transformer_new_Model import TransformerModel

# --- GPU Optimization for A100 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROCESSED_DIR = project_root / "data" / "processed" / "ieee_phm" / "Transformer_new"
SAVE_MODEL_DIR = project_root / "outputs" / "ieee_phm_bearing" / "Transformer_new"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

with open(PROCESSED_DIR / "hyperparams.json") as f:
    hyperparams = json.load(f)

data = np.load(PROCESSED_DIR / "ieee_phm_sequences.npz")
BATCH_SIZE = hyperparams['BATCH_SIZE']
N_FEATURES = hyperparams['N_FEATURES']

train_loader = DataLoader(TensorDataset(torch.tensor(data["X_train"]), torch.tensor(data["y_train"]).view(-1, 1)), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(torch.tensor(data["X_val"]), torch.tensor(data["y_val"]).view(-1, 1)), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on {device} (A100 Optuna Accelerated)")

# ===============================================================
# OPTUNA OBJECTIVE FUNCTION
# ===============================================================
def objective(trial):
    # Suggest Hyperparameters
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    
    # Mathematical constraint: d_model must be divisible by nhead for MultiheadAttention
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    num_layers = trial.suggest_int("num_layers", 1, 3)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    # Initialize Model
    model = TransformerModel(
        input_size=N_FEATURES, d_model=d_model, nhead=nhead, 
        num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    best_val_loss = float('inf')

    # Fast evaluation loop (15 epochs per trial)
    for epoch in range(15):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

# Run Optuna Study
print("🔍 Starting Optuna Hyperparameter Search...")
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=30) # 30 trials is a good balance of time and search space

best_params = study.best_params
print(f"\n✅ Optuna Search Complete! Best Params: {best_params}")

# ===============================================================
# FINAL MODEL TRAINING (Using Best Params)
# ===============================================================
print("\n🚀 Training Final Model with Best Hyperparameters...")
EPOCHS = 50

final_model = TransformerModel(
    input_size=N_FEATURES, 
    d_model=best_params["d_model"], 
    nhead=best_params["nhead"], 
    num_layers=best_params["num_layers"], 
    dim_feedforward=best_params["dim_feedforward"], 
    dropout=best_params["dropout"]
).to(device)

optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.HuberLoss()

checkpoint_path = SAVE_MODEL_DIR / 'transformer_best_model.pth'
best_val_loss = float('inf')
patience, early_stop_counter = 12, 0

for epoch in range(EPOCHS):
    final_model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    train_loss /= len(train_loader.dataset)

    final_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = final_model(batch_X)
            val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(final_model.state_dict(), checkpoint_path)
        
        # Save best params alongside model for evaluation reference
        with open(SAVE_MODEL_DIR / "best_optuna_params.json", "w") as f:
            json.dump(best_params, f)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}.")
            break

print(f"🎉 Final model saved to {checkpoint_path}")
