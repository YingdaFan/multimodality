"""
TiDE — Time-series Dense Encoder (Das et al., Google 2023, arXiv:2304.08424).

Standalone PyTorch reimplementation adapted to ForecastLoader / (B, T, C) convention.
基于 Nixtla NeuralForecast 的 reference (https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/tide.py)
做最小复刻——去掉 BaseWindows / scaler / Lightning 等业务层，留下纯 nn.Module。

设计要点
  - 全 MLP（无 attention，无 RNN），编码-解码风格
  - 显式区分三类协变量：
        hist_exog : 仅过去时段有
        futr_exog : 过去和未来时段都已知（如气象预报）
        stat_exog : 时间不变
  - 论文报告比 PatchTST / TFT 同水平甚至更强，速度快 5-10×

输入 / 输出（forecast 用法）
    forward(y_hist, hist_exog=None, futr_exog=None, stat_exog=None)
        y_hist    : (B, L, 1)              endogenous 历史（这里 L = input_size = window）
        hist_exog : (B, L, F_hist) 可选     past-only 协变量
        futr_exog : (B, L+h, F_futr) 可选   past+future 都已知的协变量
        stat_exog : (B, F_stat) 可选        静态特征
        return    : (B, h, n_outputs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPResidual(nn.Module):
    """两层 MLP + 残差 + 可选 LayerNorm。TiDE 的基础积木。"""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int,
                 dropout: float, layernorm: bool):
        super().__init__()
        self.layernorm = layernorm
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.drop = nn.Dropout(dropout)
        if layernorm:
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = F.relu(self.lin1(x))
        h = self.lin2(h)
        h = self.drop(h)
        h = h + self.skip(x)
        if self.layernorm:
            h = self.norm(h)
        return h


class TiDE(nn.Module):
    """
    Time-series Dense Encoder.

    构造参数
        input_size  : history length L
        h           : forecast horizon (pred_len)
        hist_exog_size / futr_exog_size / stat_exog_size : 各类协变量的通道数
        hidden_size           : MLP 隐藏维度
        decoder_output_dim    : decoder 输出在 reshape 前的每步通道数
        temporal_decoder_dim  : 末端 per-step decoder 的隐藏维度
        dropout / layernorm   : 正则
        num_encoder_layers / num_decoder_layers : 堆叠层数
        temporal_width        : 协变量投影后的紧凑维度
        n_outputs             : 输出通道数（确定性 = 1；高斯 = 2；分位数 = K）
    """

    def __init__(self,
                 input_size: int,
                 h: int,
                 hist_exog_size: int = 0,
                 futr_exog_size: int = 0,
                 stat_exog_size: int = 0,
                 hidden_size: int = 512,
                 decoder_output_dim: int = 32,
                 temporal_decoder_dim: int = 128,
                 dropout: float = 0.3,
                 layernorm: bool = True,
                 num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1,
                 temporal_width: int = 4,
                 n_outputs: int = 1):
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.hist_exog_size = hist_exog_size
        self.futr_exog_size = futr_exog_size
        self.stat_exog_size = stat_exog_size
        self.temporal_width = temporal_width
        self.decoder_output_dim = decoder_output_dim
        self.n_outputs = n_outputs

        # ── 协变量投影（把 F → temporal_width，per timestep） ──
        if hist_exog_size > 0:
            self.hist_exog_projection = MLPResidual(
                input_dim=hist_exog_size, hidden_size=hidden_size,
                output_dim=temporal_width, dropout=dropout, layernorm=layernorm,
            )
        if futr_exog_size > 0:
            self.futr_exog_projection = MLPResidual(
                input_dim=futr_exog_size, hidden_size=hidden_size,
                output_dim=temporal_width, dropout=dropout, layernorm=layernorm,
            )

        # ── Dense encoder：把所有信号 flat 后过 MLP 栈 ──
        enc_in = (
            input_size
            + input_size * (hist_exog_size > 0) * temporal_width
            + (input_size + h) * (futr_exog_size > 0) * temporal_width
            + (stat_exog_size > 0) * stat_exog_size
        )
        enc_layers = [
            MLPResidual(
                input_dim=enc_in if i == 0 else hidden_size,
                hidden_size=hidden_size,
                output_dim=hidden_size,
                dropout=dropout, layernorm=layernorm,
            ) for i in range(num_encoder_layers)
        ]
        self.dense_encoder = nn.Sequential(*enc_layers)

        # ── Dense decoder：投到 h × decoder_output_dim ──
        dec_out_size = decoder_output_dim * h
        dec_layers = [
            MLPResidual(
                input_dim=hidden_size,
                hidden_size=hidden_size,
                output_dim=(dec_out_size if i == num_decoder_layers - 1 else hidden_size),
                dropout=dropout, layernorm=layernorm,
            ) for i in range(num_decoder_layers)
        ]
        self.dense_decoder = nn.Sequential(*dec_layers)

        # ── Temporal decoder：per-step 投到 n_outputs。
        #    若有 futr_exog，每一步还会叠上对应的 future-portion 投影 ──
        self.temporal_decoder = MLPResidual(
            input_dim=decoder_output_dim + (futr_exog_size > 0) * temporal_width,
            hidden_size=temporal_decoder_dim,
            output_dim=n_outputs,
            dropout=dropout, layernorm=layernorm,
        )

        # ── 全局 skip：y_hist 直通到 forecast 段 ──
        self.global_skip = nn.Linear(input_size, h * n_outputs)

    def forward(self,
                y_hist: torch.Tensor,
                hist_exog: torch.Tensor = None,
                futr_exog: torch.Tensor = None,
                stat_exog: torch.Tensor = None):
        """
        y_hist    : (B, L, 1)
        hist_exog : (B, L, F_hist)  optional
        futr_exog : (B, L+h, F_futr) optional
        stat_exog : (B, F_stat)      optional
        return    : (B, h, n_outputs)
        """
        B, L, _ = y_hist.shape

        # 1) flatten y 并算 global skip
        y_flat = y_hist.reshape(B, L)                                    # (B, L)
        skip = self.global_skip(y_flat).reshape(B, self.h, self.n_outputs)
        # skip: (B, h, n_outputs)

        # 2) 拼接：[y_flat | hist_exog_flat | futr_exog_flat | stat_exog]
        x = y_flat
        if self.hist_exog_size > 0 and hist_exog is not None:
            h_proj = self.hist_exog_projection(hist_exog)                 # (B, L, temporal_width)
            x = torch.cat([x, h_proj.reshape(B, -1)], dim=1)
        if self.futr_exog_size > 0 and futr_exog is not None:
            f_proj = self.futr_exog_projection(futr_exog)                 # (B, L+h, temporal_width)
            x = torch.cat([x, f_proj.reshape(B, -1)], dim=1)
        if self.stat_exog_size > 0 and stat_exog is not None:
            x = torch.cat([x, stat_exog], dim=1)

        # 3) Dense encoder → Dense decoder
        x = self.dense_encoder(x)                                          # (B, hidden_size)
        x = self.dense_decoder(x).reshape(B, self.h, -1)                   # (B, h, decoder_output_dim)

        # 4) 拼上 futr_exog 的 future portion，做 temporal decoder
        if self.futr_exog_size > 0 and futr_exog is not None:
            f_proj_future = f_proj[:, L:]                                   # (B, h, temporal_width)
            x = torch.cat([x, f_proj_future], dim=2)
        x = self.temporal_decoder(x)                                        # (B, h, n_outputs)

        return x + skip
