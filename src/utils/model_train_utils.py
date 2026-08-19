import json
from tqdm.auto import tqdm
import torch
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch with progress bar"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Train", leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    progress_bar.close()
    return avg_loss


def evaluate(model, test_loader, criterion, device, epoch, num_epochs):
    """Evaluate model with progress bar"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{num_epochs} Val  ", leave=False)
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # RUL-specific score (NASA scoring function)
    def score_func(y_true, y_pred):
        diff = y_pred - y_true
        score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
        return score
    
    score = score_func(actuals, predictions)
    
    metrics = {
        'loss': total_loss / len(test_loader),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'score': score
    }
    
    progress_bar.close()
    return metrics, predictions, actuals


def save_history_to_json(history, save_path, model_name):
    """Save training history to JSON file"""
    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(save_path, f"{model_name}_history.json")
    
    # Convert numpy types to Python types for JSON serialization
    history_serializable = {}
    for key, values in history.items():
        history_serializable[key] = [float(v) for v in values]
    
    with open(json_path, 'w') as f:
        json.dump(history_serializable, f, indent=4)
    
    print(f"Training history saved to: {json_path}")
