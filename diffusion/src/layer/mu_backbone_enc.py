"""
Pure Encoder backbone for calibration task.

Key differences from mu_backbone.py:
- No Decoder: directly project encoder output to prediction
- Fully bidirectional: every position can attend to all other positions
- Simpler architecture, fewer parameters

Rationale:
- Calibration task doesn't require autoregressive generation
- We have the entire sequence available, so bidirectional attention is natural
- Cross-attention in Encoder-Decoder is redundant when input/output are aligned
"""
import torch
import torch.nn as nn
from torch_timeseries.nn.Transformer_EncDec import Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Pure Encoder Model for Calibration Task

    Architecture:
    - Encoder with bidirectional self-attention
    - Direct projection to output dimension
    - No Decoder, no cross-attention

    Benefits for calibration:
    - Every time step can see all other time steps (bidirectional)
    - Simpler architecture, fewer parameters
    - No causal mask limitation
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # Embedding
        time_embed = getattr(configs, 'time_embed', True)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, time_embed=time_embed)

        # Encoder with bidirectional self-attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Direct projection to output dimension (no Decoder)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # De-stationary factors
        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass for pure encoder model.

        Note: x_dec and x_mark_dec are accepted but ignored for API compatibility
        with the original mu_backbone.Model interface.
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # De-stationary factors
        tau = self.tau_learner(x_raw, std_enc).exp()  # B x 1
        delta = self.delta_learner(x_raw, mean_enc)   # B x S

        # Encoder (bidirectional self-attention)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        # Direct projection to output (no Decoder)
        output = self.projection(enc_out)  # (B, seq_len, c_out)

        # De-normalization
        if self.c_out == self.enc_in:
            output = output * std_enc + mean_enc
        else:
            # MS mode: use last c_out features' statistics
            std_out = std_enc[:, :, -self.c_out:]
            mean_out = mean_enc[:, :, -self.c_out:]
            output = output * std_out + mean_out

        if self.output_attention:
            return output[:, -self.pred_len:, :], attns
        else:
            return output[:, -self.pred_len:, :], output
