import torch
import torch.nn as nn


class GRURULModel(nn.Module):
    """
    Improved GRU model for RUL prediction.
    
    Improvements over original Keras version:
    - Bidirectional GRU option for richer temporal context
    - Residual connection between GRU layers
    - Layer normalization (more stable than BatchNorm for sequences)
    - Separate weight decay (L2) in optimizer instead of per-layer regularizers
    """

    def __init__(self, n_features: int = 8, hidden1: int = 128, hidden2: int = 64,
                 fc_dim: int = 64, dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU Layer 1
        self.gru1 = nn.GRU(
            input_size=n_features,
            hidden_size=hidden1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.ln1 = nn.LayerNorm(hidden1 * self.num_directions)
        self.dropout1 = nn.Dropout(dropout)

        # GRU Layer 2
        self.gru2 = nn.GRU(
            input_size=hidden1 * self.num_directions,
            hidden_size=hidden2,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.ln2 = nn.LayerNorm(hidden2 * self.num_directions)
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection (if dimensions differ)
        gru1_out_dim = hidden1 * self.num_directions
        gru2_out_dim = hidden2 * self.num_directions
        if gru1_out_dim != gru2_out_dim:
            self.residual_proj = nn.Linear(gru1_out_dim, gru2_out_dim)
        else:
            self.residual_proj = nn.Identity()

        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(gru2_out_dim, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Linear(fc_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, 1) — predicted RUL
        """
        # GRU Layer 1
        out1, _ = self.gru1(x)              # (B, T, hidden1 * dirs)
        out1 = self.ln1(out1)
        out1 = self.dropout1(out1)

        # GRU Layer 2
        out2, _ = self.gru2(out1)            # (B, T, hidden2 * dirs)
        out2 = self.ln2(out2)
        out2 = self.dropout2(out2)

        # Residual connection (last timestep)
        residual = self.residual_proj(out1[:, -1, :])   # (B, hidden2*dirs)
        last_hidden = out2[:, -1, :] + residual          # (B, hidden2*dirs)

        # MLP Head
        return self.head(last_hidden)                    # (B, 1)

