import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder for RUL prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.2, output_activation=None):
        super().__init__()
        self.output_activation = output_activation
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.activations = {}
        
    def forward(self, x, return_hidden=False):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Use last token
        out = self.fc(transformer_out[:, -1, :])
        
        # if self.output_activation is not None:
        #     if self.output_activation == "sigmoid":
        #         out = torch.sigmoid(out)
        #     elif self.output_activation == "relu":
        #         out = torch.relu(out)
        #     elif self.output_activation == "tanh":
        #         out = torch.tanh(out)

        if return_hidden:
            return out, {'transformer_out': transformer_out}
        
        return out


