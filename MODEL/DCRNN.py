import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

def calculate_scaled_laplacian(adj_mx, lambda_max=2.0):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj_mx.shape[0]) - adj_mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian.toarray()

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_matrix, max_diffusion_step=2, 
                 num_nodes=None, filter_type="dual_random_walk", device=None):
        super(DCGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes or adj_matrix.shape[0]
        
        self.supports = []
        if filter_type == "laplacian":
            self.supports.append(torch.FloatTensor(calculate_scaled_laplacian(adj_matrix)).to(device))
        elif filter_type == "random_walk":
            self.supports.append(torch.FloatTensor(calculate_random_walk_matrix(adj_matrix).T).to(device))
        elif filter_type == "dual_random_walk":
            self.supports.append(torch.FloatTensor(calculate_random_walk_matrix(adj_matrix).T).to(device))
            self.supports.append(torch.FloatTensor(calculate_random_walk_matrix(adj_matrix.T).T).to(device))
        
        self.num_matrices = len(self.supports) * self.max_diffusion_step + 1
        
        input_size = input_dim + hidden_dim
        self.weight_r = nn.Parameter(torch.zeros(size=(input_size * self.num_matrices, hidden_dim))).to(device)
        self.weight_u = nn.Parameter(torch.zeros(size=(input_size * self.num_matrices, hidden_dim))).to(device)
        self.weight_c = nn.Parameter(torch.zeros(size=(input_size * self.num_matrices, hidden_dim))).to(device)
        self.bias_r = nn.Parameter(torch.zeros(hidden_dim)).to(device)
        self.bias_u = nn.Parameter(torch.zeros(hidden_dim)).to(device)
        self.bias_c = nn.Parameter(torch.zeros(hidden_dim)).to(device)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_u)
        nn.init.xavier_uniform_(self.weight_c)
        
    def forward(self, inputs, hx):
        """
        :param inputs: (num_nodes, input_dim)
        :param hx: (num_nodes, hidden_dim)
        :return: (num_nodes, hidden_dim)
        """
        concat_input = torch.cat([inputs, hx], dim=-1)
        
        support_set = [concat_input]
        for support in self.supports:
            support_set.extend(self._calculate_chebyshev_polynomials(support, concat_input))
            
        supports_concat = torch.cat(support_set, dim=-1)
        
        r = torch.sigmoid(torch.matmul(supports_concat, self.weight_r) + self.bias_r)
        u = torch.sigmoid(torch.matmul(supports_concat, self.weight_u) + self.bias_u)
        
        concat_reset = torch.cat([inputs, r * hx], dim=-1)
        support_set_reset = [concat_reset]
        for support in self.supports:
            support_set_reset.extend(self._calculate_chebyshev_polynomials(support, concat_reset))
        supports_concat_reset = torch.cat(support_set_reset, dim=-1)
        
        c = torch.tanh(torch.matmul(supports_concat_reset, self.weight_c) + self.bias_c)
        
        new_state = u * hx + (1.0 - u) * c
        
        return new_state
        
    def _calculate_chebyshev_polynomials(self, support, x, K=None):

        if K is None:
            K = self.max_diffusion_step
            
        supports = []
        
        if K <= 0:
            return supports
            
        x0 = x
        x1 = torch.matmul(support, x0)
        supports.append(x1)
        
        for k in range(2, K + 1):
            x2 = 2 * torch.matmul(support, x1) - x0
            supports.append(x2)
            x1, x0 = x2, x1
            
        return supports

class DCRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, adj_matrix, remap_matrix=None, device='cuda', max_diffusion_step=2,
                 filter_type="dual_random_walk", seed=None):
        super(DCRNN, self).__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.hidden_size = hidden_dim
        self.device = device
        self.num_nodes = adj_matrix.shape[0]
        
        self.dcrnn_cell = DCGRUCell(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            adj_matrix=adj_matrix,
            max_diffusion_step=max_diffusion_step,
            filter_type=filter_type,
            device=device
        )
        if remap_matrix is not None:
            self.remap_matrix = torch.from_numpy(remap_matrix).float().to(device)
        self.output_layer = nn.Linear(hidden_dim, 1).to(device)
        self.to(device)


    def forward(self, x):
        """
        :param x: (num_nodes, seq_len, in_dim)
        :return: (num_nodes, seq_len, output_dim)
        """
        num_nodes, seq_len, _ = x.size()
        
        h = torch.zeros(num_nodes, self.hidden_size).to(self.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (num_nodes, in_dim)
            h = self.dcrnn_cell(x_t, h)  # h: (num_nodes, hidden_size)
            output = self.output_layer(h)  # output: (num_nodes, output_dim)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs

