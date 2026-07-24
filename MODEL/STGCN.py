import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x shape: [nodes, time, features]
        nodes, time_steps, _ = x.size()

        # Reshape to handle each time step separately
        x = x.transpose(0, 1)  # [time, nodes, features]

        # Apply graph convolution to each time step
        support = torch.matmul(x, self.weights)  # [time, nodes, out_features]
        output = torch.matmul(adj, support)  # [time, nodes, out_features]

        # Reshape back and add bias
        output = output.transpose(0, 1)  # [nodes, time, out_features]
        return output + self.bias


class STConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3):
        super(STConvBlock, self).__init__()
        self.temporal1 = TemporalConvLayer(in_channels, hidden_channels, kernel_size)
        self.graph = GraphConvLayer(hidden_channels, hidden_channels)
        self.temporal2 = TemporalConvLayer(hidden_channels, out_channels, kernel_size)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj):
        # First temporal conv
        x = x.permute(0, 2, 1)  # [nodes, features, time]
        x = self.temporal1(x)
        x = x.permute(0, 2, 1)  # [nodes, time, features]

        # Graph conv
        x = self.graph(x, adj)

        # Second temporal conv
        x = x.permute(0, 2, 1)
        x = self.temporal2(x)
        x = x.permute(0, 2, 1)

        x = self.norm(x)
        return self.dropout(x)


class STGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_matrix, remap_matrix=None, recur_dropout=0, dropout=0.1, return_states=False,
                 device='cuda', seed=None):
        super(STGCN, self).__init__()
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.A = torch.from_numpy(adj_matrix).float().to(device)
        self.dim = hidden_dim
        self.return_states = return_states

        # Two ST-Conv blocks
        self.st_block1 = STConvBlock(input_dim, hidden_dim, hidden_dim)
        self.st_block2 = STConvBlock(hidden_dim, hidden_dim, hidden_dim)

        # Output layer
        self.temporal_output = TemporalConvLayer(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        if remap_matrix is not None:
            self.remap_matrix = torch.from_numpy(remap_matrix).float().to(device)
    def forward(self, x, init_states=None):
        # x shape: [nodes, time, features]

        # Process through ST-Conv blocks
        x = self.st_block1(x, self.A)
        x = self.st_block2(x, self.A)

        # Output processing
        x = x.permute(0, 2, 1)
        x = self.temporal_output(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)

        # Final fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)

        if self.return_states:
            h_t = x[:, -1, :]
            c_t = x[:, -1, :]
            return out, (h_t, c_t)
        return out