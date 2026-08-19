import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM for RUL prediction with instrumentation for XAI"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Storage for activations (for XAI)
        self.activations = {}
        self.gradients = {}
        
    def forward(self, x, return_hidden=False):
        # x: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        
        if return_hidden:
            return out, {'lstm_out': lstm_out, 'hidden': hidden, 'cell': cell}
        return out
    
    def register_hooks(self):
        """Register forward hooks to capture activations"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        self.lstm.register_forward_hook(get_activation('lstm'))

