import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """GRU for RUL prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.activations = {}
        
    def forward(self, x, return_hidden=False):
        # x: (batch, seq_len, features)
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state
        out = self.fc(gru_out[:, -1, :])
        
        if return_hidden:
            return out, {'gru_out': gru_out, 'hidden': hidden}
        return out