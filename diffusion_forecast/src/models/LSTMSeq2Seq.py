import torch
import torch.nn as nn


class LSTMSeq2Seq(nn.Module):
    """Sequence-to-sequence LSTM for forecasting.

    Encoder processes history (B, T, enc_in).
    Decoder processes future features (B, O, enc_in) with encoder hidden state.
    Output projection maps to c_out dimensions.
    """

    def __init__(self, enc_in, c_out, pred_len, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out

        self.encoder = nn.LSTM(
            enc_in, d_model, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            enc_in, d_model, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_dec):
        """
        Args:
            x_enc: (B, T, enc_in) history sequence
            x_dec: (B, O, enc_in) future features (known X + zeros for Y)
        Returns:
            (B, O, c_out) predictions
        """
        _, (h, c) = self.encoder(x_enc)
        dec_out, _ = self.decoder(x_dec, (h, c))
        dec_out = self.dropout(dec_out)
        return self.fc(dec_out)
