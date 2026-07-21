import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, feature_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        self.mean = nn.Linear(feature_dim, latent_dim)
        self.log_var = nn.Linear(feature_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean(h)
        log_var = self.log_var(h)
        z = self.reparameterize(mean, log_var)
        return mean, log_var, z
    
    def decode(self, z):
        return self.decoder(z)

        
class MDN_LSTM(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()        
        self.W_forget = nn.Linear(
            state_dim + hidden_dim,
            hidden_dim
        )

        self.W_input = nn.Linear(
            state_dim + hidden_dim,
            hidden_dim
        )

        self.W_memory = nn.Linear(
            state_dim + hidden_dim,
            hidden_dim
        )

        self.W_output = nn.Linear(
            state_dim + hidden_dim,
            hidden_dim
        )
        
    def lstm_step(self, state, short_m, long_m):
        combined = torch.cat([state, short_m], dim=-1)
        
        f = torch.sigmoid(self.W_forget(combined))
        i = torch.sigmoid(self.W_input(combined))
        candidate_m = torch.tanh(self.W_memory(combined))
        o = torch.sigmoid(self.W_output(combined))
        
        long_m = f * long_m + i * candidate_m
        short_m = o * torch.tanh(long_m)
        return short_m, long_m
