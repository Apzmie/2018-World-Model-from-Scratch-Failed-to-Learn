import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, action_dim, latent_dim=16, hidden_dim=256, num_distribution=5):
        super().__init__()        
        self.W_forget = nn.Linear(action_dim + latent_dim + hidden_dim, hidden_dim)
        self.W_input = nn.Linear(action_dim + latent_dim + hidden_dim, hidden_dim)
        self.W_memory_update = nn.Linear(action_dim + latent_dim + hidden_dim, hidden_dim)
        self.W_output = nn.Linear(action_dim + latent_dim + hidden_dim, hidden_dim)
        
        self.probability = nn.Linear(hidden_dim, num_distribution)
        self.mean = nn.Linear(hidden_dim, num_distribution * latent_dim)
        self.log_std = nn.Linear(hidden_dim, num_distribution * latent_dim)
        self.num_distribution = num_distribution
        self.latent_dim = latent_dim
        
    def lstm(self, input, summary, memory):
        combined = torch.cat([input, summary], dim=-1)
        
        forget_gate = torch.sigmoid(self.W_forget(combined))
        input_gate = torch.sigmoid(self.W_input(combined))
        memory_update = torch.tanh(self.W_memory_update(combined))
        output_gate = torch.sigmoid(self.W_output(combined))
        
        new_memory = forget_gate * memory + input_gate * memory_update
        new_summary = output_gate * torch.tanh(new_memory)
        return new_summary, new_memory
        
    def forward(self, action, z, summary, memory):
        input = torch.cat([action, z], dim=-1)
        summary, memory = self.lstm(input, summary, memory)
        
        probability = F.softmax(self.probability(summary), dim=-1)
        
        mean = self.mean(summary)
        mean = mean.view(-1, self.num_distribution, self.latent_dim)
        
        log_std = self.log_std(summary)
        log_std = log_std.view(-1, self.num_distribution, self.latent_dim)
        std = torch.exp(log_std)
              
        return summary, memory, probability, mean, std
        
        
class Controller(nn.Module):
    def __init__(self, action_dim, latent_dim=16, hidden_dim=256):
        super().__init__()      
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)
        
    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        action = torch.tanh(self.fc(x))
        return action

        
class Diagonal_CMA_ES:
    def __init__(self, num_params, candidate_size=24, selected_ratio=0.25, exploration_scale=0.5, exploration_decay=0.99, learning_rate=0.2):
        self.num_params = num_params
        self.mean = np.zeros(num_params)
        self.variance = np.ones(num_params)
        
        self.candidate_size = candidate_size
        self.num_selected = int(candidate_size * selected_ratio)

        self.exploration_scale = exploration_scale
        self.exploration_decay = exploration_decay       
        self.learning_rate = learning_rate

    def sample(self):
        noise = np.random.randn(self.candidate_size, self.num_params)
        candidates = self.mean + self.exploration_scale * noise * np.sqrt(self.variance)
        return candidates

    def update(self, candidates, rewards):
        ranked_indices = np.argsort(rewards)[::-1]
        selected = candidates[ranked_indices[:self.num_selected]]

        old_mean = self.mean.copy()
        self.mean = np.mean(selected, axis=0)

        distances = selected - old_mean
        new_variance = np.mean(distances ** 2, axis=0)

        self.variance = (1 - self.learning_rate) * self.variance + self.learning_rate * new_variance
        self.variance = np.clip(self.variance, 1e-8, None)
        
        self.exploration_scale = self.exploration_scale * self.exploration_decay

    def best_candidate(self):
        return self.mean
    

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.vae = VAE(state_dim)
        self.mdn_lstm = MDN_LSTM(action_dim)
                
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.mdn_optimizer = torch.optim.Adam(self.mdn_lstm.parameters(), lr=lr)
        
    def mdn_loss(self, target_z, mean, std, probability):
        """
        target_z: (batch, latent_dim)
        mean: (batch, num_distribution, latent_dim)
        std: (batch, num_distribution, latent_dim)
        probability: (batch, num_distribution)
        """   
        target_z = target_z.unsqueeze(1)

        normal = torch.distributions.Normal(mean, std)
        component_log_prob = normal.log_prob(target_z)
        vector_log_prob = torch.sum(component_log_prob, dim=-1)

        gaussian_log_prob = torch.log(probability + 1e-8)

        log_likelihood = torch.logsumexp(vector_log_prob + gaussian_log_prob, dim=1)
        loss = -torch.mean(log_likelihood)
        return loss
        
    def update_vae(self, state):
        mean, log_var, z = self.vae.encode(state)
        reconstructed = self.vae.decode(z)
        reconstruction_loss = F.mse_loss(reconstructed, state)
        distribution_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
        loss = reconstruction_loss + distribution_loss
        
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        
        return reconstruction_loss.item(), distribution_loss.item(), loss.item()
    
    def update_mdn(self, action, z, summary, memory, target_z):
        summary, memory, probability, mean, std = self.mdn_lstm(action, z, summary, memory)
        
        
        
        
        
        
    
        self.mdn_optimizer.zero_grad()
        loss.backward()
        self.mdn_optimizer.step()
        
        return loss.item()
    
    def sample_next_z(self, probability, mean, std):
        gaussian_idx = torch.distributions.Categorical(probability).sample()
        batch_idx = torch.arange(probability.shape[0])
        selected_mean = mean[batch_idx, gaussian_idx]
        selected_std = std[batch_idx, gaussian_idx]
        eps = torch.randn_like(selected_std)
        next_z = selected_mean + eps * selected_std
        return next_z
    








        
