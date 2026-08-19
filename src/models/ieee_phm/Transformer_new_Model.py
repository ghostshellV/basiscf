# import torch
# import torch.nn as nn
# import numpy as np

# class PositionalEncoding(nn.Module):
#     """Positional encoding for transformer"""
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


# class TransformerModel(nn.Module):
#     """Transformer encoder for RUL prediction"""
#     def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, 
#                  num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.2, output_activation=None):
#         super().__init__()
#         self.output_activation = output_activation
        
#         # Input projection
#         self.input_proj = nn.Linear(input_size, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
#         # Output layers
#         self.fc = nn.Sequential(
#             nn.Linear(d_model, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1)
#         )
        
#         self.activations = {}
        
#     def forward(self, x, return_hidden=False):
#         # x: (batch, seq_len, features)
#         x = self.input_proj(x)
#         x = self.pos_encoder(x)
        
#         # Transformer encoding
#         transformer_out = self.transformer_encoder(x)
        
#         # Use last token
#         out = self.fc(transformer_out[:, -1, :])
        
#         if return_hidden:
#             return out, {'transformer_out': transformer_out}
        
#         return out

### THIS UPPER ONE WAS THE OLD TRANSFORMER MODEL. 

##=====================================================================================================
# NOW WE ARE USING THE NEW TRANSFORMER MODEL WHICH HAS GLOBAL POOLING AND OTHER IMPROVEMENTS.
##=====================================================================================================

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
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
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 3, dim_feedforward: int = 512, dropout: float = 0.2):
        super().__init__()
        
        # 1. Higher Capacity Input Projection
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 2. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Improved stability for RUL
        )
        
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers, 
            enable_nested_tensor=False
        )


        # 3. GLOBAL POOLING (The Secret Sauce from the Colab code)
        # Instead of just taking the last token, we average across time
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Final Prediction Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.BatchNorm1d(64), # Adds stability to the learning process
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_hidden=False):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        transformer_out = self.transformer_encoder(x)
        
        # REPLACED: Use Global Average Pooling instead of [:, -1, :]
        # transformer_out is (Batch, Seq_Len, d_model) -> Permute to (Batch, d_model, Seq_Len)
        pooled_out = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        
        out = self.fc(pooled_out)
        
        if return_hidden:
            return out, {'transformer_out': transformer_out}
        return out