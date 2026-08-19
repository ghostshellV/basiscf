"""
Visualization Utilities for RUL Prediction Models

This module provides comprehensive plotting functions for visualizing training
progress, model predictions, and performance comparisons.
"""

import os
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def ensure_output_dir(save_path: str = '../outputs/visualizations') -> None:
    """Ensure output directory exists"""
    os.makedirs(save_path, exist_ok=True)


def plot_training_history(
    histories: List[Dict],
    model_names: List[str],
    save_path: str = '../outputs/visualizations'
) -> None:
    """
    Plot training histories for multiple models.
    
    Args:
        histories: List of training history dictionaries
        model_names: List of model names
        save_path: Directory to save plots
    """
    ensure_output_dir(save_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Training Comparison', fontsize=16, fontweight='bold')
    
    # Updated metric names to match new Trainer
    metrics = [
        'train_loss', 
        'val_loss', 
        'val_rmse', 
        'val_mae', 
        'val_r2', 
        'val_nasa_score'
    ]
    titles = [
        'Training Loss', 
        'Validation Loss', 
        'Validation RMSE',
        'Validation MAE', 
        'Validation R²', 
        'NASA Score'
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for history, name, color in zip(histories, model_names, colors[:len(model_names)]):
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=name, linewidth=2.5, 
                       color=color, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add best value annotation for validation metrics
        if 'val' in metric and histories:
            for history, name, color in zip(histories, model_names, colors[:len(model_names)]):
                if metric in history:
                    if metric == 'val_nasa_score':
                        best_val = min(history[metric])
                        best_idx = history[metric].index(best_val)
                    elif metric == 'val_r2':
                        best_val = max(history[metric])
                        best_idx = history[metric].index(best_val)
                    else:
                        best_val = min(history[metric])
                        best_idx = history[metric].index(best_val)
    
    plt.tight_layout()
    save_file = os.path.join(save_path, 'training_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Training comparison plot saved to: {save_file}")
    plt.show()


def plot_predictions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
    model_name: str,
    num_samples: Optional[int] = None,
    save_path: str = '../outputs/visualizations'
) -> None:
    """
    Plot predictions vs actual values for a model.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device model is on
        model_name: Name of the model
        num_samples: Number of samples to plot (None for all)
        save_path: Directory to save plots
    """
    ensure_output_dir(save_path)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data).squeeze()
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    if num_samples is not None:
        predictions = predictions[:num_samples]
        actuals = actuals[:num_samples]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{model_name} Model - Prediction Analysis', 
                 fontsize=14, fontweight='bold')
    
    # 1. Predictions vs Actuals scatter plot
    axes[0].scatter(actuals, predictions, alpha=0.6, s=50, c='blue', edgecolors='black')
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 
                 'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.8)
    axes[0].set_xlabel('Actual RUL', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Predicted RUL', fontsize=11, fontweight='bold')
    axes[0].set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Add R² to the plot
    from sklearn.metrics import r2_score
    r2 = r2_score(actuals, predictions)
    axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=axes[0].transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Error distribution histogram
    errors = predictions - actuals
    axes[1].hist(errors, bins=40, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2.5, 
                   label='Zero Error', alpha=0.8)
    axes[1].axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean Error = {errors.mean():.2f}', alpha=0.8)
    axes[1].set_xlabel('Prediction Error (Predicted - Actual)', 
                      fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Time series of predictions
    sample_indices = range(len(actuals))
    axes[2].plot(sample_indices, actuals, 'o-', label='Actual', 
                linewidth=2, markersize=4, alpha=0.7)
    axes[2].plot(sample_indices, predictions, 's-', label='Predicted', 
                linewidth=2, markersize=4, alpha=0.7)
    axes[2].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('RUL', fontsize=11, fontweight='bold')
    axes[2].set_title('Prediction Sequence', fontsize=12, fontweight='bold')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_file = os.path.join(save_path, f'{model_name.lower()}_predictions.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Prediction plot saved to: {save_file}")
    plt.show()
    
    # Print statistics
    print(f"\n{model_name} Prediction Statistics:")
    print(f"  Mean Error: {errors.mean():.4f}")
    print(f"  Std Error: {errors.std():.4f}")
    print(f"  RMSE: {np.sqrt((errors**2).mean()):.4f}")
    print(f"  MAE: {np.abs(errors).mean():.4f}")
    print(f"  R²: {r2:.4f}")


def plot_model_comparison(
    histories: Dict[str, Dict],
    model_names: List[str],
    save_path: str = '../outputs/visualizations'
) -> None:
    """
    Create a comprehensive comparison of model performances.
    
    Args:
        histories: Dictionary of training histories keyed by model name
        model_names: List of model names
        save_path: Directory to save plots
    """
    ensure_output_dir(save_path)
    
    # Extract final metrics for each model
    metrics_summary = {
        'Model': [],
        'Final Train Loss': [],
        'Final Val Loss': [],
        'Best RMSE': [],
        'Best MAE': [],
        'Best R²': [],
        'Best NASA Score': []
    }
    
    for name in model_names:
        history = histories[name] if isinstance(histories, dict) else histories[model_names.index(name)]
        
        metrics_summary['Model'].append(name.upper())
        metrics_summary['Final Train Loss'].append(history['train_loss'][-1])
        metrics_summary['Final Val Loss'].append(history['val_loss'][-1])
        metrics_summary['Best RMSE'].append(min(history['val_rmse']))
        metrics_summary['Best MAE'].append(min(history['val_mae']))
        metrics_summary['Best R²'].append(max(history['val_r2']))
        metrics_summary['Best NASA Score'].append(min(history['val_nasa_score']))
    
    # Create bar plots for comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Performance Comparison - Best Metrics', 
                 fontsize=16, fontweight='bold')
    
    comparison_metrics = [
        ('Final Train Loss', 'lower'),
        ('Final Val Loss', 'lower'),
        ('Best RMSE', 'lower'),
        ('Best MAE', 'lower'),
        ('Best R²', 'higher'),
        ('Best NASA Score', 'lower')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (metric, better) in enumerate(comparison_metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = metrics_summary[metric]
        bars = ax.bar(metrics_summary['Model'], values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        # Highlight best performer
        if better == 'lower':
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgreen')
        bars[best_idx].set_linewidth(3)
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_file = os.path.join(save_path, 'model_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {save_file}")
    plt.show()
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'Train Loss':<12} {'Val Loss':<12} {'RMSE':<10} "
          f"{'MAE':<10} {'R²':<10} {'NASA Score':<12}")
    print("-"*80)
    for i, name in enumerate(metrics_summary['Model']):
        print(f"{name:<15} "
              f"{metrics_summary['Final Train Loss'][i]:<12.4f} "
              f"{metrics_summary['Final Val Loss'][i]:<12.4f} "
              f"{metrics_summary['Best RMSE'][i]:<10.4f} "
              f"{metrics_summary['Best MAE'][i]:<10.4f} "
              f"{metrics_summary['Best R²'][i]:<10.4f} "
              f"{metrics_summary['Best NASA Score'][i]:<12.2f}")
    print("="*80 + "\n")


def plot_learning_rate_schedule(
    history: Dict,
    model_name: str,
    save_path: str = '../outputs/visualizations'
) -> None:
    """
    Plot learning rate schedule over training.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_path: Directory to save plots
    """
    ensure_output_dir(save_path)
    
    if 'learning_rate' not in history:
        print("Learning rate not tracked in history")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history['learning_rate']) + 1)
    ax.plot(epochs, history['learning_rate'], linewidth=2.5, 
           color='darkblue', marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Learning Rate Schedule', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    save_file = os.path.join(save_path, f'{model_name.lower()}_lr_schedule.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Learning rate schedule plot saved to: {save_file}")
    plt.show()
