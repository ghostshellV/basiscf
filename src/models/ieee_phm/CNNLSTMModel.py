import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid for RUL prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.activations = {}
        
    def forward(self, x, return_hidden=False):
        # x: (batch, seq_len, features)
        # Permute for CNN: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Permute back for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        
        if return_hidden:
            return out, {'lstm_out': lstm_out, 'hidden': hidden, 'cell': cell}
        return out

