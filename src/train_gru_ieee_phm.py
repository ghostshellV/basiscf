import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ieee_phm.GRU_BI_Model import GRUBiRULModel

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    PROCESSED_DIR = project_root / "data" / "processed" / "ieee_phm" / "gru_bi"
    SAVE_MODEL_DIR = project_root / "outputs" / "ieee_phm_bearing" / "gru_bi"
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    with open(PROCESSED_DIR / "hyperparams.json") as f:
        hyperparams = json.load(f)

    data = np.load(PROCESSED_DIR / "ieee_phm_sequences.npz")
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    
    train_loader = DataLoader(TensorDataset(torch.tensor(data["X_train"]), torch.tensor(data["y_train"]).view(-1, 1)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(data["X_val"]), torch.tensor(data["y_val"]).view(-1, 1)), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUBiRULModel(input_dim=hyperparams['N_FEATURES'], seq_len=hyperparams['GRU_SEQ_LEN']).to(device)
    
    EPOCHS = 150
    LEARNING_RATE = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7)
    criterion = nn.HuberLoss() 

    patience = 8
    early_stop_counter = 0
    best_val_loss = float('inf')
    checkpoint_path = SAVE_MODEL_DIR / 'gru_bi_best_model.pth'

    print("\n🚀 STARTING PYTORCH TRAINING (Bi-GRU + LayerNorm)")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [*] Best model saved -> {checkpoint_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    main()