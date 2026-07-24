"""
FutureTST_windownorm — native (B, T, C) FutureTST with per-window instance normalization.

这是 FutureTST 原 paper 的设定：在 forward 入口对 endogenous Y_history 沿 time 维
求 mean/std 做 instance-norm，出口反归一化。本质是 RevIN 思想，让模型只学
"窗口内的相对走势"，不被窗口间的整体尺度漂移干扰。

与 src/models/FutureTST.py 的区别
    本文件                : 带 TimeSeriesNormalizer（per-window instance-norm）
    FutureTST.py (无 norm): 不带，只依赖 pipeline 层的全局 z-score

输入 / 输出约定（与 FutureTST.py 完全一致）
    forward(x):  x.shape == (B, T, C)
                 T = context_window_size + pred_size
                 C = input_channels (last channel = endogenous Y)
    return:      predictions.shape == (B, pred_size, 1)
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ───────── Instance normalization (沿 time 维, 即 dim=1) ─────────

class TimeSeriesNormalizer:
    """
    Per-window instance-norm。原版 FutureTST 在 (B, C, T) 上沿 dim=2 求统计量；
    本版在 (B, T, C) 上沿 dim=1 求。means / stdev 形状 (B, 1, C)，可与 (B, T, C) 直接广播。
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def normalize(self, x):
        means = x.mean(1, keepdim=True).detach()                                    # (B, 1, C)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.epsilon).detach()
        x = x / stdev
        return x, means, stdev

    def denormalize(self, x_normalized, means, stdev):
        return x_normalized * stdev + means


# ───────── 与原版完全相同的部件（不依赖布局） ─────────

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda y: self.self_attention_block(y, y, y, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda y: self.self_attention_block(y, y, y, tgt_mask))
        x = self.residual_connections[1](
            x, lambda y: self.cross_attention_block(y, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


def build_transformer(d_model: int = 512, N: int = 6, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    encoder_blocks = []
    for _ in range(N):
        attn = MultiHeadAttentionBlock(d_model, h, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, attn, ff, dropout))
    decoder_blocks = []
    for _ in range(N):
        self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(d_model, self_attn, cross_attn, ff, dropout))
    transformer = Transformer(
        Encoder(d_model, nn.ModuleList(encoder_blocks)),
        Decoder(d_model, nn.ModuleList(decoder_blocks)),
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


# ───────── 布局相关：ExtractPatches / LinearProjectionLayer ─────────

class ExtractPatches(nn.Module):
    """沿 time 维（dim=1）切片成 patches。输入 (B, T, C) → 输出 (B, num_patches, patch_size, C)。"""

    def __init__(self, patch_size: int, stride_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride_len = stride_len

    def forward(self, x):
        T = x.shape[1]
        num_patches = int(np.floor((T - self.patch_size) / self.stride_len) + 2)
        last_slice = x[:, -1:, :].repeat(1, self.stride_len, 1)
        x_extended = torch.cat((x, last_slice), dim=1)
        patches = []
        for i in range(num_patches):
            start = i * self.stride_len
            end = start + self.patch_size
            patches.append(x_extended[:, start:end, :])
        return torch.stack(patches, dim=1)


class LinearProjectionLayer(nn.Module):
    """把 patch 内 patch_size 个时间步投影到 d_model。输入 (B, N, P, C) → 输出 (B, N, C, d_model)。"""

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model, bias=False)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        return self.proj(x)


class LinearHead(nn.Module):
    """(B, num_patches, d_model) → flatten → Linear → (B, pred_size)。"""

    def __init__(self, d_model: int, num_patch: int, pred_size: int):
        super().__init__()
        self.linearhead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * num_patch, pred_size),
        )

    def forward(self, x):
        return self.linearhead(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal PE，作用在 (batch, seq_len, d_model)。"""

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# ───────── FutureTSTWindowNorm 主模型（带 per-window instance-norm） ─────────

class FutureTSTWindowNorm(nn.Module):
    """
    Native (B, T, C) FutureTST + per-window instance normalization。
    对齐原 FutureTST paper 设定。
    """

    def __init__(self, context_window_size: int = 365, patch_size: int = 16, stride_len: int = 8,
                 d_model: int = 256, num_transformer_layers: int = 2, mlp_size: int = 128,
                 num_heads: int = 8, mlp_dropout: float = 0.2, pred_size: int = 20,
                 embedding_dropout: float = 0.1, input_channels: int = 0):
        super().__init__()
        self.patch_size = patch_size
        self.stride_len = stride_len
        self.pred_size = pred_size
        self.context_window_size = context_window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        self.embedding_dropout = embedding_dropout
        self.input_channels = input_channels

        self.num_endoPatch = int(np.floor((context_window_size - patch_size) / stride_len) + 2)

        self.normalizer = TimeSeriesNormalizer()
        self.extract_patches = ExtractPatches(patch_size, stride_len)
        self.exogeneous_feature_projection = nn.Linear(context_window_size + pred_size, d_model)
        self.linear_projection_layer = LinearProjectionLayer(patch_size, d_model)
        self.endo_positional_encoding = PositionalEncoding(d_model, self.num_endoPatch, embedding_dropout)
        self.exo_positional_encoding = PositionalEncoding(d_model, input_channels - 1, embedding_dropout)
        self.transformer = build_transformer(
            d_model=d_model, N=num_transformer_layers, h=num_heads,
            dropout=mlp_dropout, d_ff=mlp_size,
        )
        self.linear_head = LinearHead(d_model, self.num_endoPatch, pred_size)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)，T = context_window_size + pred_size，C = input_channels
            x[:, :, :-1]                     = 外生 X（含 history+future）  (B, T, C-1)
            x[:, :context_window_size, -1:]  = 内生 Y_history               (B, context, 1)
        """
        # 1) 拆通道
        exogeneous_data = x[:, :, :-1]                                                # (B, T, C-1)
        endogeneous_data = x[:, :self.context_window_size, -1:]                       # (B, context, 1)

        # 2) 内生 instance-norm（沿 time 维 dim=1）— paper 设定，必启用
        endogeneous_data, means, stdev = self.normalizer.normalize(endogeneous_data)  # (B, context, 1)

        # 3) 内生切 patch
        endo_patches = self.extract_patches(endogeneous_data)                          # (B, num_patches, patch_size, 1)

        # 4) 内生 patch_size → d_model
        endo_patches = self.linear_projection_layer(endo_patches)                      # (B, num_patches, 1, d_model)

        # 5) 外生：把 time 维投影到 d_model
        exo_t = exogeneous_data.permute(0, 2, 1)                                       # (B, C-1, T)
        exo_proj = self.exogeneous_feature_projection(exo_t)                           # (B, C-1, d_model)

        # 6) 折叠内生的 channel=1，得到 transformer 输入 (B, num_patches, d_model)
        endo_for_pe = endo_patches.squeeze(2)                                          # (B, num_patches, d_model)

        # 7) 位置编码
        endo_for_pe = self.endo_positional_encoding(endo_for_pe)                       # (B, num_patches, d_model)
        exo_for_pe = self.exo_positional_encoding(exo_proj)                            # (B, C-1, d_model)

        # 8) Transformer encoder/decoder
        encoder_output = self.transformer.encode(exo_for_pe, None)                     # (B, C-1, d_model)
        decoder_output = self.transformer.decode(encoder_output, None, endo_for_pe, None)
        # decoder_output: (B, num_patches, d_model)

        # 9) 投影到 pred_size
        predictions = self.linear_head(decoder_output)                                 # (B, pred_size)
        predictions = predictions.unsqueeze(-1)                                        # (B, pred_size, 1)

        # 10) 反归一化（paper 设定，必执行）
        predictions = self.normalizer.denormalize(predictions, means, stdev)           # (B, pred_size, 1)

        return predictions
