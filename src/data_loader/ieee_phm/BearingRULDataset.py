import numpy as np
import torch
from torch.utils.data import Dataset

class BearingRULDataset(Dataset):
    """
    PyTorch Dataset for bearing RUL prediction.
    Returns tensors of shape:
        X: (seq_len, n_features)
        y: scalar (RUL)
    When batched by DataLoader -> (batch, seq_len, n_features)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, seq_len, n_features) float32
            y: (n_samples,) float32
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

