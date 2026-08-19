"""
STAR: Two-Stage Attention-Based Hierarchical Transformer (PyTorch)
-----------------------------------------------------------------

This file provides a practical implementation of the STAR-style architecture
described in the Sensors 2024 paper (two-stage attention + hierarchical transformer)
for turbofan engine RUL prediction (CMAPSS).

Key ideas implemented:
- Two-stage attention: temporal attention per sensor, then sensor-wise attention per patch.
- Hierarchical / multiscale modelling via patch merging.
- Encoder-decoder with two-stage blocks and cross-attention.
- Optional sigmoid output scaling to keep predictions in [0, max_rul] (avoids negative RUL).

Usage:
    from STARModel import STARModel
    model = STARModel(input_dim=14, seq_len=32, patch_len=4, num_scales=3, max_rul=125.0)
    y = model(x)  # x: (B, 32, 14) -> y: (B,)

"""

import math
from typing import List
import torch
import torch.nn as nn


def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return sinusoidal PE of shape (length, dim)."""
    position = torch.arange(length, device=device).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStageAttentionBlock(nn.Module):
    """
    Input/Output: (B, K, D, E)
    Stage-1: temporal attention per sensor (attend along K)
    Stage-2: sensor-wise attention per patch (attend along D)
    """
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sensor_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_t2 = nn.LayerNorm(d_model)
        self.norm_s1 = nn.LayerNorm(d_model)
        self.norm_s2 = nn.LayerNorm(d_model)

        self.ff_t = FeedForward(d_model, ff_dim, dropout)
        self.ff_s = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, D, E = x.shape

        # Stage 1: temporal attention per sensor
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * D, K, E)  # (B*D, K, E)
        attn_out, _ = self.temporal_attn(xt, xt, xt, need_weights=False)
        xt = self.norm_t1(xt + attn_out)
        xt = self.norm_t2(xt + self.ff_t(xt))
        x = xt.view(B, D, K, E).permute(0, 2, 1, 3).contiguous()

        # Stage 2: sensor-wise attention per patch
        xs = x.view(B * K, D, E)  # (B*K, D, E)
        attn_out, _ = self.sensor_attn(xs, xs, xs, need_weights=False)
        xs = self.norm_s1(xs + attn_out)
        xs = self.norm_s2(xs + self.ff_s(xs))
        x = xs.view(B, K, D, E)

        return x


class PatchEmbedding(nn.Module):
    """(B, T, D) -> (B, K, D, E) using non-overlapping patches of length patch_len."""
    def __init__(self, input_dim: int, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_len = patch_len

        self.proj = nn.Linear(patch_len, d_model)
        self.sensor_emb = nn.Parameter(torch.zeros(1, 1, input_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {D}, expected {self.input_dim}")

        K = T // self.patch_len
        T_use = K * self.patch_len
        x = x[:, :T_use, :]  # truncate to full patches

        x = x.view(B, K, self.patch_len, D).permute(0, 1, 3, 2).contiguous()  # (B,K,D,patch)
        x = self.proj(x)  # (B,K,D,E)

        pe_k = sinusoidal_positional_encoding(K, self.d_model, x.device).view(1, K, 1, self.d_model)
        x = x + self.sensor_emb + pe_k
        return self.drop(x)


class PatchMerging(nn.Module):
    """Merge adjacent patches in time: K -> floor(K/2)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.reduction = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, D, E = x.shape
        K2 = (K // 2) * 2
        x = x[:, :K2, :, :]
        x0 = x[:, 0::2, :, :]
        x1 = x[:, 1::2, :, :]
        x = torch.cat([x0, x1], dim=-1)  # (B,K/2,D,2E)
        return self.reduction(x)         # (B,K/2,D,E)


class DecoderBlock(nn.Module):
    """Two-stage self-attn + cross-attn + FFN at a given scale."""
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_two_stage = TwoStageAttentionBlock(d_model, nhead, ff_dim, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        x_dec = self.self_two_stage(x_dec)

        B, K, D, E = x_dec.shape
        q = x_dec.view(B, K * D, E)
        kv = x_enc.view(B, K * D, E)

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ff(q))
        return q.view(B, K, D, E)


class STARModel(nn.Module):
    """
    STAR hierarchical encoder-decoder.

    output_activation:
      - "sigmoid": returns cycles in [0, max_rul]
      - "none": raw output (handle scaling/clipping externally)
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 32,
        patch_len: int = 4,
        num_scales: int = 3,
        d_model: int = 128,
        nhead: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_rul: float = 125.0,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_scales = num_scales
        self.d_model = d_model
        self.max_rul = float(max_rul)
        self.output_activation = output_activation

        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len.")
        K0 = seq_len // patch_len
        min_K = K0 // (2 ** (num_scales - 1))
        if min_K < 1:
            raise ValueError(
                f"Bad config: seq_len={seq_len}, patch_len={patch_len}, num_scales={num_scales}. "
                f"K0={K0} becomes <1 after merging."
            )

        self.embed = PatchEmbedding(input_dim, d_model, patch_len, dropout)

        self.enc_blocks = nn.ModuleList([TwoStageAttentionBlock(d_model, nhead, ff_dim, dropout) for _ in range(num_scales)])
        self.enc_merge = nn.ModuleList([PatchMerging(d_model) for _ in range(num_scales - 1)])

        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, nhead, ff_dim, dropout) for _ in range(num_scales)])
        self.dec_merge = nn.ModuleList([PatchMerging(d_model) for _ in range(num_scales - 1)])

        self.head = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _init_decoder_state(self, K: int, D: int, device: torch.device) -> torch.Tensor:
        pe_k = sinusoidal_positional_encoding(K, self.d_model, device).view(1, K, 1, self.d_model)
        return pe_k.repeat(1, 1, D, 1)  # (1,K,D,E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got D={D}")

        h = self.embed(x)  # (B,K0,D,E)

        enc_outs: List[torch.Tensor] = []
        for s in range(self.num_scales):
            h = self.enc_blocks[s](h)
            enc_outs.append(h)
            if s < self.num_scales - 1:
                h = self.enc_merge[s](h)

        dec = None
        pools: List[torch.Tensor] = []
        for s in range(self.num_scales):
            enc_s = enc_outs[s]
            _, K, D, _ = enc_s.shape
            if s == 0:
                dec = self._init_decoder_state(K, D, x.device).repeat(B, 1, 1, 1)
            else:
                dec = self.dec_merge[s - 1](dec)

            dec = self.dec_blocks[s](dec, enc_s)
            pools.append(dec.mean(dim=(1, 2)))  # (B,E)

        z = torch.cat(pools, dim=-1)  # (B,E*num_scales)
        y = self.head(z).squeeze(-1)

        if self.output_activation == "sigmoid":
            y = torch.sigmoid(y) * self.max_rul
        elif self.output_activation == "none":
            pass
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")
        return y


# if __name__ == "__main__":
#     # Quick smoke test
#     B, T, D = 4, 32, 14
#     x = torch.randn(B, T, D)
#     model = STARModel(input_dim=D, seq_len=T, patch_len=4, num_scales=3, d_model=128, nhead=1, max_rul=125.0)
#     y = model(x)
#     print("Output:", y.shape, "min/max:", float(y.min()), float(y.max()))