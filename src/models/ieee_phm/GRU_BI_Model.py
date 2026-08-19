import torch
import torch.nn as nn

class GRUBiRULModel(nn.Module):
    def __init__(self, input_dim=8, seq_len=512):
        super(GRUBiRULModel, self).__init__()
        
        # First Bi-GRU Block (Output dim: 128 * 2 = 256)
        self.gru1 = nn.GRU(input_dim, 128, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(256) 
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Bi-GRU Block (Output dim: 64 * 2 = 128)
        self.gru2 = nn.GRU(256, 64, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense Layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x, _ = self.gru1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        # Second Bi-GRU
        _, h_n = self.gru2(x)
        
        # Merge final forward state and final backward state
        x = torch.cat((h_n[-2], h_n[-1]), dim=1) 
        
        x = self.ln2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        
        return x