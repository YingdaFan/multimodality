
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RGCN_v1(nn.Module):
    """
    改进的RGCN模型，针对完全缺失流域的预测问题
    主要改进：
    1. 可学习的邻接矩阵与原始邻接矩阵融合
    2. 伪标签生成机制用于缺失流域
    3. 域适应模块处理域偏移问题
    4. 空间注意力机制增强流域间关系学习
    """
    def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout=0, dropout=0,
                 return_states=False, device='cpu', seed=None,
                 use_learnable_adj=True, use_pseudo_labels=True,
                 use_domain_adapt=True, use_spatial_attention=True):

        """
        @param input_dim: [int] number input feature
        @param hidden_dim: [int] hidden size
        @param adj_matrix: Distance matrix for graph convolution
        @param recur_dropout: [float] fraction of the units to drop from the cell update vector
        @param dropout: [float] fraction of the units to drop from the input
        @param return_states: [bool] If true, returns h and c states as well as predictions
        @param use_learnable_adj: [bool] 是否使用可学习的邻接矩阵
        @param use_pseudo_labels: [bool] 是否使用伪标签机制
        @param use_domain_adapt: [bool] 是否使用域适应
        @param use_spatial_attention: [bool] 是否使用空间注意力
        """
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        super().__init__()

        # 原始邻接矩阵
        self.A = torch.from_numpy(adj_matrix).float().to(device)
        num_basins = adj_matrix.shape[0]

        # 可学习的邻接矩阵组件
        self.use_learnable_adj = use_learnable_adj
        if use_learnable_adj:
            self.learnable_A = nn.Parameter(torch.randn(num_basins, num_basins).float().to(device) * 0.01)
            self.adj_fusion_weight = nn.Parameter(torch.tensor(0.5))

        # GCN权重
        self.weight_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_q = nn.Parameter(torch.Tensor(hidden_dim))

        self.input_dim = input_dim
        self.hidden_size = hidden_dim

        # LSTM权重
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))

        # 伪标签生成器
        self.use_pseudo_labels = use_pseudo_labels
        if use_pseudo_labels:
            self.pseudo_generator = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.confidence_estimator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        # 域适应模块
        self.use_domain_adapt = use_domain_adapt
        if use_domain_adapt:
            self.domain_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        # 空间注意力机制
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)

        self.dense = nn.Linear(hidden_dim, 1)
        self.return_states = return_states

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def get_adaptive_adjacency(self):
        """计算自适应的邻接矩阵"""
        if self.use_learnable_adj:
            # 使用sigmoid确保权重在0-1之间，然后与原始矩阵融合
            learned_A = torch.sigmoid(self.learnable_A)
            alpha = torch.sigmoid(self.adj_fusion_weight)
            adaptive_A = alpha * self.A + (1 - alpha) * learned_A
            # 归一化
            row_sums = adaptive_A.sum(dim=1, keepdim=True)
            adaptive_A = adaptive_A / (row_sums + 1e-8)
            return adaptive_A
        else:
            return self.A

    def forward(self, x, init_states=None, basin_mask=None, return_extras=False):
        """
        前向传播
        @param x: 输入数据 (batch, sequence, feature)
        @param init_states: 初始状态
        @param basin_mask: 标记哪些是缺失流域 (batch,)，1表示缺失，0表示有观测
        @param return_extras: 是否返回额外信息（伪标签、置信度等）
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        pseudo_labels = []
        confidences = []
        domain_features = []

        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        x = self.dropout(x)
        HS = self.hidden_size

        # 获取自适应邻接矩阵
        adaptive_A = self.get_adaptive_adjacency()

        for t in range(seq_sz):
            x_t = x[:, t, :]

            # LSTM门控计算
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            # GCN组件 - 使用自适应邻接矩阵
            q_t = torch.tanh(h_t @ self.weight_q + self.bias_q)
            spatial_context = adaptive_A @ q_t

            # 应用空间注意力（每隔几个时间步）
            if self.use_spatial_attention and t % 5 == 0:
                h_t_reshaped = h_t.unsqueeze(1)  # (batch, 1, hidden_dim)
                attn_output, _ = self.spatial_attention(
                    h_t_reshaped, h_t_reshaped, h_t_reshaped
                )
                h_t_attn = self.attention_norm(h_t + attn_output.squeeze(1))
            else:
                h_t_attn = h_t

            # 更新cell state和hidden state
            c_t = f_t * (c_t + spatial_context) + i_t * self.recur_dropout(g_t)
            h_t = o_t * torch.tanh(c_t)

            # 融合注意力特征
            if self.use_spatial_attention and t % 5 == 0:
                h_t = 0.9 * h_t + 0.1 * h_t_attn

            # 域适应处理
            if self.use_domain_adapt and basin_mask is not None:
                domain_feat = self.domain_encoder(h_t)
                domain_features.append(domain_feat)

            # 伪标签生成（用于缺失流域）
            if self.use_pseudo_labels and basin_mask is not None:
                # 结合隐藏状态和空间上下文生成伪标签
                combined_feat = torch.cat([h_t, spatial_context], dim=-1)
                pseudo_pred = self.pseudo_generator(combined_feat)
                conf = self.confidence_estimator(h_t)

                # 只对缺失流域生成伪标签
                if basin_mask is not None:
                    pseudo_pred = pseudo_pred * basin_mask.unsqueeze(-1).unsqueeze(-1).float()
                    conf = conf * basin_mask.unsqueeze(-1).unsqueeze(-1).float()

                pseudo_labels.append(pseudo_pred)
                confidences.append(conf)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)
        out = self.dense(hidden_seq)

        if return_extras:
            extras = {
                'pseudo_labels': torch.stack(pseudo_labels, dim=1) if pseudo_labels else None,
                'confidences': torch.stack(confidences, dim=1) if confidences else None,
                'domain_features': torch.stack(domain_features, dim=1) if domain_features else None,
                'adaptive_adjacency': adaptive_A
            }
            if self.return_states:
                return out, (h_t, c_t), extras
            else:
                return out, extras
        else:
            if self.return_states:
                return out, (h_t, c_t)
            else:
                return out
        







