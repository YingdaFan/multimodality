"""
Wave Fusion Modules for Bidirectional Diffusion

Physical Analogy:
- Two waves propagate from opposite ends of the sequence
- They meet, interfere, and create standing wave patterns
- Information from both directions is fused at each position

Three fusion strategies:
1. WaveSuperposition: Position-dependent weighted sum + interference term
2. WaveScatterFusion: Cross-attention based wave scattering
3. WaveScatterInterference: Combination of both approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class WaveSuperposition(nn.Module):
    """
    Wave Superposition Fusion (Approach 1: Superposition + Interference)

    Physical formula: y(t) = α(t)·h_fwd + β(t)·h_bwd + γ(t)·(h_fwd ⊙ h_bwd)

    - α(t): Forward wave strength, decreases with position (wave travels from start)
    - β(t): Backward wave strength, increases with position (wave travels from end)
    - γ(t): Interference term, strongest at middle where waves collide
    - ⊙: Element-wise product representing nonlinear wave collision
    """

    def __init__(self, d_model, seq_len, learnable_decay=True):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        if learnable_decay:
            # Learnable position-dependent wave strengths
            self.alpha = nn.Parameter(torch.linspace(1.0, 0.0, seq_len))  # Forward: 1→0
            self.beta = nn.Parameter(torch.linspace(0.0, 1.0, seq_len))   # Backward: 0→1
            # Interference: strongest at middle (sine wave pattern)
            self.gamma = nn.Parameter(torch.sin(torch.linspace(0, np.pi, seq_len)))
        else:
            # Fixed decay patterns
            self.register_buffer('alpha', torch.linspace(1.0, 0.0, seq_len))
            self.register_buffer('beta', torch.linspace(0.0, 1.0, seq_len))
            self.register_buffer('gamma', torch.sin(torch.linspace(0, np.pi, seq_len)))

        # Interference projection: transform the collision product
        self.interference_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h_fwd, h_bwd):
        """
        Args:
            h_fwd: (B, T, D) Forward encoded representation
            h_bwd: (B, T, D) Backward encoded representation (already flipped back to original order)

        Returns:
            h_fused: (B, T, D) Fused representation
        """
        B, T, D = h_fwd.shape

        # Ensure sequence length matches (handle variable lengths)
        if T != self.seq_len:
            # Interpolate decay patterns for different sequence lengths
            alpha = F.interpolate(self.alpha.unsqueeze(0).unsqueeze(0), size=T, mode='linear', align_corners=True).squeeze()
            beta = F.interpolate(self.beta.unsqueeze(0).unsqueeze(0), size=T, mode='linear', align_corners=True).squeeze()
            gamma = F.interpolate(self.gamma.unsqueeze(0).unsqueeze(0), size=T, mode='linear', align_corners=True).squeeze()
        else:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma

        # Normalize to [0, 1] range
        alpha = torch.sigmoid(alpha).view(1, T, 1)  # (1, T, 1)
        beta = torch.sigmoid(beta).view(1, T, 1)
        gamma = torch.sigmoid(gamma).view(1, T, 1)

        # Wave superposition with interference
        # h_fwd weighted by forward wave strength
        # h_bwd weighted by backward wave strength
        # Interference term: element-wise product (nonlinear collision)
        interference = self.interference_proj(h_fwd * h_bwd)

        h_fused = alpha * h_fwd + beta * h_bwd + gamma * interference

        return self.out_proj(h_fused)


class WaveScatterFusion(nn.Module):
    """
    Wave Scatter Fusion (Approach 3: Cross-Attention Scattering)

    Physical Analogy:
    - When waves collide, they exchange energy/information
    - Cross-attention models this information exchange
    - Forward wave "queries" backward wave and vice versa
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Forward wave queries backward wave
        self.fwd_queries_bwd = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Backward wave queries forward wave
        self.bwd_queries_fwd = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Layer norms for stability
        self.norm_fwd = nn.LayerNorm(d_model)
        self.norm_bwd = nn.LayerNorm(d_model)

        # Gating mechanism: learn how much to trust each direction
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, h_fwd, h_bwd):
        """
        Args:
            h_fwd: (B, T, D) Forward encoded representation
            h_bwd: (B, T, D) Backward encoded representation

        Returns:
            h_fused: (B, T, D) Fused representation
        """
        # Wave scattering: each wave gathers information from the other
        h_fwd_scattered, _ = self.fwd_queries_bwd(
            self.norm_fwd(h_fwd),
            self.norm_bwd(h_bwd),
            h_bwd
        )
        h_bwd_scattered, _ = self.bwd_queries_fwd(
            self.norm_bwd(h_bwd),
            self.norm_fwd(h_fwd),
            h_fwd
        )

        # Residual connections
        h_fwd_enriched = h_fwd + h_fwd_scattered
        h_bwd_enriched = h_bwd + h_bwd_scattered

        # Gated fusion: learn position-dependent weighting
        gate = self.gate(torch.cat([h_fwd_enriched, h_bwd_enriched], dim=-1))
        h_fused = gate * h_fwd_enriched + (1 - gate) * h_bwd_enriched

        return self.out_proj(h_fused)


class WaveScatterInterference(nn.Module):
    """
    Combined Wave Scatter + Interference Fusion (Approach 1+3 Combined)

    Two-stage fusion:
    1. First, waves scatter and exchange information (Cross-Attention)
    2. Then, scattered waves superpose with interference (Position-dependent)

    This captures both:
    - Information exchange at collision points
    - Position-dependent interference patterns
    """

    def __init__(self, d_model, seq_len, n_heads=4, dropout=0.1):
        super().__init__()

        # Stage 1: Wave scattering (information exchange)
        self.scatter = WaveScatterFusion(d_model, n_heads, dropout)

        # Stage 2: Wave superposition with interference
        self.superposition = WaveSuperposition(d_model, seq_len)

        # Layer norm between stages
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_fwd, h_bwd):
        """
        Args:
            h_fwd: (B, T, D) Forward encoded representation
            h_bwd: (B, T, D) Backward encoded representation

        Returns:
            h_fused: (B, T, D) Fused representation
        """
        # Stage 1: Scatter (waves exchange information)
        h_scattered = self.scatter(h_fwd, h_bwd)

        # The scattered result becomes the "modified" forward wave
        # Original h_bwd becomes the "modified" backward wave
        h_fwd_modified = self.norm(h_scattered)
        h_bwd_modified = self.norm(h_bwd + h_scattered - h_fwd)  # Residual adjustment

        # Stage 2: Superposition with interference
        h_fused = self.superposition(h_fwd_modified, h_bwd_modified)

        return h_fused


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional Encoder Wrapper

    Wraps any encoder to process sequences in both directions
    and fuses the results using specified fusion strategy.
    """

    def __init__(self, encoder, d_model, seq_len, fusion_type='superposition',
                 n_heads=4, dropout=0.1, share_weights=True):
        """
        Args:
            encoder: Base encoder module
            d_model: Model dimension
            seq_len: Sequence length
            fusion_type: 'superposition', 'scatter', or 'scatterinterference'
            share_weights: If True, use same encoder for both directions
        """
        super().__init__()

        self.encoder_fwd = encoder
        self.share_weights = share_weights

        if not share_weights:
            # Create a separate encoder for backward pass
            import copy
            self.encoder_bwd = copy.deepcopy(encoder)
        else:
            self.encoder_bwd = encoder

        # Select fusion strategy
        if fusion_type == 'superposition':
            self.fusion = WaveSuperposition(d_model, seq_len)
        elif fusion_type == 'scatter':
            self.fusion = WaveScatterFusion(d_model, n_heads, dropout)
        elif fusion_type == 'scatterinterference':
            self.fusion = WaveScatterInterference(d_model, seq_len, n_heads, dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.fusion_type = fusion_type

    def forward(self, x, x_mark=None):
        """
        Args:
            x: (B, T, D) Input sequence
            x_mark: (B, T, D_mark) Time encoding (optional)

        Returns:
            h_fused: (B, T, D) Fused bidirectional representation
        """
        # Forward pass: original order
        if x_mark is not None:
            h_fwd = self.encoder_fwd(x, x_mark)
        else:
            h_fwd = self.encoder_fwd(x)

        # Backward pass: flip input, encode, flip output back
        x_flip = x.flip(dims=[1])
        if x_mark is not None:
            x_mark_flip = x_mark.flip(dims=[1])
            h_bwd_flip = self.encoder_bwd(x_flip, x_mark_flip)
        else:
            h_bwd_flip = self.encoder_bwd(x_flip)

        # Flip backward output to align with forward
        h_bwd = h_bwd_flip.flip(dims=[1])

        # Fuse forward and backward representations
        h_fused = self.fusion(h_fwd, h_bwd)

        return h_fused


# Convenience functions for creating fusion modules
def create_wave_fusion(fusion_type, d_model, seq_len, n_heads=4, dropout=0.1):
    """Factory function to create wave fusion modules"""
    if fusion_type == 'superposition':
        return WaveSuperposition(d_model, seq_len)
    elif fusion_type == 'scatter':
        return WaveScatterFusion(d_model, n_heads, dropout)
    elif fusion_type == 'scatterinterference':
        return WaveScatterInterference(d_model, seq_len, n_heads, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
