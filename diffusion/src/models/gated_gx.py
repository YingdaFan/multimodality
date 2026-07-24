import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTemporalGxEstimator(nn.Module):
    def __init__(self, base_gx=0.1, kernel_size=15):
        super().__init__()
        self.base_gx = base_gx

        self.conv1 = nn.Conv1d(3, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, padding=kernel_size//2)

        self.gate_fc = nn.Linear(32, 1)
        self.scale_fc = nn.Linear(32, 1)

        nn.init.zeros_(self.gate_fc.weight)
        nn.init.constant_(self.gate_fc.bias, -3.0)
        nn.init.zeros_(self.scale_fc.weight)
        nn.init.zeros_(self.scale_fc.bias)

    def forward(self, batch_x):
        y_obs = batch_x[:, :, -1:]
        y_mean_vae = batch_x[:, :, 42:43]
        y_std_vae = batch_x[:, :, 43:44]

        features = torch.cat([y_obs, y_mean_vae, y_std_vae], dim=-1)
        features = features.permute(0, 2, 1)

        h = F.relu(self.conv1(features))
        h = F.relu(self.conv2(h))
        h = h.permute(0, 2, 1)

        gate = torch.sigmoid(self.gate_fc(h))
        scale = 0.5 + 1.5 * torch.sigmoid(self.scale_fc(h))
        gx = gate * self.base_gx * scale

        return gx, gate, scale
