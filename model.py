import torch
import torch.nn as nn

class InformerGaussian(nn.Module):
    def __init__(self, input_dim=1, d_model=32, num_heads=4):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mu_head = nn.Linear(d_model, 1)
        self.log_sigma_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        encoded = self.encoder(x)
        last = encoded[-1]
        mu = self.mu_head(last)
        log_sigma = self.log_sigma_head(last)
        return mu.squeeze(), log_sigma.squeeze()