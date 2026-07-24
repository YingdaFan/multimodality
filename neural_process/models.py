"""
Neural Process Models for Zero-Shot Time Series Reconstruction

Implements (in increasing complexity):
- CNP  (Conditional Neural Process)     - Garnelo et al. 2018a
- NP   (Neural Process, latent)         - Garnelo et al. 2018b
- ANP  (Attentive Neural Process)       - Kim et al. 2019
- GNP  (Graph Neural Process)           - Hu et al. 2023

Each location has a time series of exogenous X and target Y.
Context = observed locations with (X, Y), Target = unobserved locations with X only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    """KL(q || p) for diagonal Gaussians."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + var_q / var_p + (mu_q - mu_p) ** 2 / var_p - 1)
    return kl.sum(-1).mean()


class TemporalEncoder(nn.Module):
    """Bidirectional LSTM encoder: maps a time series to a fixed-size vector."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        x: (n_locations, seq_len, input_dim)
        Returns: (n_locations, hidden_dim)  — mean-pooled over time
        """
        output, _ = self.lstm(x)  # (n_loc, seq_len, hidden_dim)
        return output.mean(dim=1)  # (n_loc, hidden_dim)


# ---------------------------------------------------------------------------
# CNP — Conditional Neural Process (Garnelo et al. 2018a)
# ---------------------------------------------------------------------------

class ConditionalNeuralProcess(nn.Module):
    """
    CNP: no latent variable, no attention — simplest NP baseline.

    Context representations are mean-aggregated and broadcast to all targets.
    """

    def __init__(self, x_dim, y_dim=1, hidden_dim=256, latent_dim=128,
                 n_heads=4, enc_layers=2, dec_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # latent_dim kept in signature for interface compatibility but unused
        self.context_encoder = TemporalEncoder(x_dim + y_dim, hidden_dim, enc_layers, dropout)
        self.context_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.decoder = nn.LSTM(
            x_dim + hidden_dim, hidden_dim // 2,
            num_layers=dec_layers, batch_first=True, bidirectional=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, y_dim),
        )

    def forward(self, context_x, context_y, target_x, target_y=None):
        xy = torch.cat([context_x, context_y], dim=-1)
        r = self.context_proj(self.context_encoder(xy))   # (n_ctx, d)
        r_agg = r.mean(dim=0)                              # (d,)

        n_tgt, seq_len, _ = target_x.shape
        r_exp = r_agg.unsqueeze(0).unsqueeze(1).expand(n_tgt, seq_len, -1)
        dec_input = torch.cat([target_x, r_exp], dim=-1)
        dec_output, _ = self.decoder(dec_input)
        y_pred = self.output_proj(dec_output)
        return y_pred, torch.tensor(0.0, device=context_x.device)


# ---------------------------------------------------------------------------
# NP — Neural Process with latent variable (Garnelo et al. 2018b)
# ---------------------------------------------------------------------------

class NeuralProcess(nn.Module):
    """
    NP: mean aggregation + latent variable, no cross-attention.

    Adds a stochastic latent z to the CNP deterministic path.
    """

    def __init__(self, x_dim, y_dim=1, hidden_dim=256, latent_dim=128,
                 n_heads=4, enc_layers=2, dec_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.context_encoder = TemporalEncoder(x_dim + y_dim, hidden_dim, enc_layers, dropout)
        self.context_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # Latent path
        self.latent_enc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder conditioned on [det_repr, z]
        self.decoder = nn.LSTM(
            x_dim + hidden_dim + latent_dim, hidden_dim // 2,
            num_layers=dec_layers, batch_first=True, bidirectional=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, y_dim),
        )

    def _get_latent(self, r):
        r_agg = r.mean(dim=0)
        h = self.latent_enc(r_agg)
        return self.latent_mu(h), self.latent_logvar(h)

    def forward(self, context_x, context_y, target_x, target_y=None):
        xy = torch.cat([context_x, context_y], dim=-1)
        r = self.context_proj(self.context_encoder(xy))
        r_agg = r.mean(dim=0)  # deterministic path

        prior_mu, prior_logvar = self._get_latent(r)
        if target_y is not None:
            r_tgt = self.context_proj(self.context_encoder(
                torch.cat([target_x, target_y], dim=-1)))
            r_all = torch.cat([r, r_tgt], dim=0)
            post_mu, post_logvar = self._get_latent(r_all)
            z = post_mu + torch.exp(0.5 * post_logvar) * torch.randn_like(post_mu)
            kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
        else:
            z = prior_mu
            kl = torch.tensor(0.0, device=context_x.device)

        n_tgt, seq_len, _ = target_x.shape
        r_exp = r_agg.unsqueeze(0).unsqueeze(1).expand(n_tgt, seq_len, -1)
        z_exp = z.unsqueeze(0).unsqueeze(1).expand(n_tgt, seq_len, -1)
        dec_input = torch.cat([target_x, r_exp, z_exp], dim=-1)
        dec_output, _ = self.decoder(dec_input)
        y_pred = self.output_proj(dec_output)
        return y_pred, kl


# ---------------------------------------------------------------------------
# TNP-D — Transformer Neural Process (Nguyen & Grover, ICLR 2023)
# ---------------------------------------------------------------------------

class TNPDSelfAttentionBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
                                nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TransformerNeuralProcess(nn.Module):
    """
    TNP-D (deterministic): all location tokens (context + target) are fed into
    a shared Transformer with self-attention.  Context tokens carry Y information,
    target tokens do not — the Transformer learns to propagate from context to target.

    Ref: Nguyen & Grover, "Transformer Neural Processes", ICLR 2023.
    """

    def __init__(self, x_dim, y_dim=1, hidden_dim=256, latent_dim=128,
                 n_heads=4, enc_layers=2, dec_layers=2, dropout=0.1,
                 tfm_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Temporal encoders (shared backbone)
        self.ctx_temporal = TemporalEncoder(x_dim + y_dim, hidden_dim, enc_layers, dropout)
        self.tgt_temporal = TemporalEncoder(x_dim, hidden_dim, enc_layers, dropout)

        # Learnable type tokens: distinguish context vs target
        self.ctx_type = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.tgt_type = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        # Transformer self-attention over all location tokens
        self.tfm_blocks = nn.ModuleList([
            TNPDSelfAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(tfm_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

        # Temporal decoder: produces Y sequence from enriched target repr
        self.decoder = nn.LSTM(
            x_dim + hidden_dim, hidden_dim // 2,
            num_layers=dec_layers, batch_first=True, bidirectional=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, y_dim),
        )

    def forward(self, context_x, context_y, target_x, target_y=None):
        n_ctx = context_x.shape[0]
        n_tgt = target_x.shape[0]

        # Encode each location to a fixed vector
        ctx_tokens = self.ctx_temporal(torch.cat([context_x, context_y], dim=-1))  # (n_ctx, d)
        tgt_tokens = self.tgt_temporal(target_x)                                    # (n_tgt, d)

        # Add type embeddings
        ctx_tokens = ctx_tokens + self.ctx_type
        tgt_tokens = tgt_tokens + self.tgt_type

        # Concatenate all tokens and run Transformer
        tokens = torch.cat([ctx_tokens, tgt_tokens], dim=0).unsqueeze(0)  # (1, n_ctx+n_tgt, d)
        for blk in self.tfm_blocks:
            tokens = blk(tokens)
        tokens = self.out_norm(tokens.squeeze(0))  # (n_ctx+n_tgt, d)

        # Extract enriched target representations
        tgt_repr = tokens[n_ctx:]  # (n_tgt, d)

        # Decode to time series
        seq_len = target_x.shape[1]
        tgt_exp = tgt_repr.unsqueeze(1).expand(-1, seq_len, -1)
        dec_input = torch.cat([target_x, tgt_exp], dim=-1)
        dec_output, _ = self.decoder(dec_input)
        y_pred = self.output_proj(dec_output)

        return y_pred, torch.tensor(0.0, device=context_x.device)


# ---------------------------------------------------------------------------
# ANP — Attentive Neural Process (Kim et al. 2019)
# ---------------------------------------------------------------------------

class AttentiveNeuralProcess(nn.Module):
    """
    ANP for zero-shot time series reconstruction.

    Architecture:
    - Context encoder:  BiLSTM([X, Y]) -> r_k   (per observed location)
    - Target encoder:   BiLSTM(X)      -> q_j   (per unobserved location)
    - Deterministic:    CrossAttention(q=q_j, kv=r_k) -> d_j
    - Latent:           mean(r_k) -> MLP -> (mu, logvar) -> z
    - Decoder:          BiLSTM(X_j, cond=[d_j, z]) -> Y_j
    """

    def __init__(self, x_dim, y_dim=1, hidden_dim=256, latent_dim=128,
                 n_heads=4, enc_layers=2, dec_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # --- Encoders ---
        self.context_encoder = TemporalEncoder(x_dim + y_dim, hidden_dim, enc_layers, dropout)
        self.target_encoder = TemporalEncoder(x_dim, hidden_dim, enc_layers, dropout)
        self.context_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.target_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # --- Deterministic path: cross-attention ---
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # --- Latent path ---
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder ---
        self.decoder = nn.LSTM(
            x_dim + hidden_dim + latent_dim, hidden_dim // 2,
            num_layers=dec_layers, batch_first=True, bidirectional=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, y_dim),
        )

    def _encode_context(self, context_x, context_y):
        xy = torch.cat([context_x, context_y], dim=-1)
        r = self.context_encoder(xy)
        return self.context_proj(r)  # (n_ctx, hidden_dim)

    def _encode_target(self, target_x):
        s = self.target_encoder(target_x)
        return self.target_proj(s)  # (n_tgt, hidden_dim)

    def _get_latent(self, r):
        r_agg = r.mean(dim=0, keepdim=True)  # (1, d)
        h = self.latent_encoder(r_agg).squeeze(0)  # (d,)
        return self.latent_mu(h), self.latent_logvar(h)

    def _decode(self, target_x, det_repr, z):
        n_tgt, seq_len, _ = target_x.shape
        det_exp = det_repr.unsqueeze(1).expand(-1, seq_len, -1)
        z_exp = z.unsqueeze(0).unsqueeze(1).expand(n_tgt, seq_len, -1)
        dec_input = torch.cat([target_x, det_exp, z_exp], dim=-1)
        dec_output, _ = self.decoder(dec_input)
        return self.output_proj(dec_output)  # (n_tgt, seq_len, y_dim)

    def forward(self, context_x, context_y, target_x, target_y=None):
        """
        Args:
            context_x: (n_ctx, seq_len, x_dim)
            context_y: (n_ctx, seq_len, 1)
            target_x:  (n_tgt, seq_len, x_dim)
            target_y:  (n_tgt, seq_len, 1) or None (inference)
        Returns:
            y_pred: (n_tgt, seq_len, 1)
            kl:     scalar
        """
        r = self._encode_context(context_x, context_y)  # (n_ctx, d)
        s = self._encode_target(target_x)                # (n_tgt, d)

        # Deterministic: cross-attention (target queries context)
        det, _ = self.cross_attn(s.unsqueeze(0), r.unsqueeze(0), r.unsqueeze(0))
        det = self.attn_norm(det.squeeze(0) + s)  # residual + norm

        # Latent: prior from context
        prior_mu, prior_logvar = self._get_latent(r)

        if target_y is not None:
            # Posterior from context + target
            r_tgt = self._encode_context(target_x, target_y)
            r_all = torch.cat([r, r_tgt], dim=0)
            post_mu, post_logvar = self._get_latent(r_all)
            z = post_mu + torch.exp(0.5 * post_logvar) * torch.randn_like(post_mu)
            kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
        else:
            z = prior_mu
            kl = torch.tensor(0.0, device=context_x.device)

        y_pred = self._decode(target_x, det, z)
        return y_pred, kl


# ---------------------------------------------------------------------------
# Graph Neural Process
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Simple graph convolution with residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        """
        x:   (n_nodes, dim)
        adj: (n_nodes, n_nodes) row-normalised adjacency
        """
        h = self.linear(adj @ x)
        return self.norm(F.relu(h) + x)


class GraphNeuralProcess(AttentiveNeuralProcess):
    """
    GNP: ANP augmented with GNN message-passing over a spatial graph.

    After the initial temporal encoding, a small GNN propagates information
    between spatially nearby locations before the NP aggregation step.
    """

    def __init__(self, x_dim, y_dim=1, hidden_dim=256, latent_dim=128,
                 n_heads=4, enc_layers=2, dec_layers=2, dropout=0.1,
                 gnn_layers=2, k_neighbors=10):
        super().__init__(x_dim, y_dim, hidden_dim, latent_dim,
                         n_heads, enc_layers, dec_layers, dropout)
        self.k_neighbors = k_neighbors
        self.gnn = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(gnn_layers)])
        # Separate projection for target nodes that participate in graph
        self.graph_target_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    @staticmethod
    def _build_adj(dist_sub, k):
        """Build row-normalised k-NN adjacency from a distance sub-matrix."""
        n = dist_sub.shape[0]
        k = min(k, n - 1)
        _, topk = dist_sub.topk(k + 1, dim=1, largest=False)  # include self
        adj = torch.zeros_like(dist_sub)
        adj.scatter_(1, topk, 1.0)
        adj = (adj + adj.T).clamp(max=1.0)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        return adj / deg

    def forward(self, context_x, context_y, target_x, target_y=None,
                dist_matrix=None, ctx_indices=None, tgt_indices=None):
        """
        Extra GNP args (optional — falls back to ANP if absent):
            dist_matrix: (N, N) pairwise distance, N = total basins
            ctx_indices: LongTensor of context basin indices
            tgt_indices: LongTensor of target basin indices
        """
        r = self._encode_context(context_x, context_y)
        s_raw = self.target_encoder(target_x)
        s = self.graph_target_proj(s_raw)

        # --- GNN pass (if graph is available) ---
        if dist_matrix is not None and ctx_indices is not None and tgt_indices is not None:
            all_idx = torch.cat([ctx_indices, tgt_indices])
            dist_sub = dist_matrix[all_idx][:, all_idx]
            adj = self._build_adj(dist_sub, self.k_neighbors)
            h = torch.cat([r, s], dim=0)
            for layer in self.gnn:
                h = layer(h, adj)
            n_ctx = r.shape[0]
            r, s = h[:n_ctx], h[n_ctx:]

        # Deterministic cross-attention
        det, _ = self.cross_attn(s.unsqueeze(0), r.unsqueeze(0), r.unsqueeze(0))
        det = self.attn_norm(det.squeeze(0) + s)

        # Latent
        prior_mu, prior_logvar = self._get_latent(r)
        if target_y is not None:
            r_tgt = self._encode_context(target_x, target_y)
            r_all = torch.cat([r, r_tgt], dim=0)
            post_mu, post_logvar = self._get_latent(r_all)
            z = post_mu + torch.exp(0.5 * post_logvar) * torch.randn_like(post_mu)
            kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
        else:
            z = prior_mu
            kl = torch.tensor(0.0, device=context_x.device)

        y_pred = self._decode(target_x, det, z)
        return y_pred, kl
