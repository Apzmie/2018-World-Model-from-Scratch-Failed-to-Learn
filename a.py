from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter       

BASE_DIR = ""


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
            

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.vae = VAE(state_dim)
        self.mdn_lstm = MDN_LSTM(action_dim)
                
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.mdn_optimizer = torch.optim.Adam(self.mdn_lstm.parameters(), lr=lr)
        
    def update_vae(self, state):
        mean, log_var, z = self.vae.encode(state)
        reconstructed = self.vae.decode(z)
        reconstruction_loss = F.mse_loss(reconstructed, state)
        distribution_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
        loss = reconstruction_loss + distribution_loss
        
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        
        return {
            "reconstruction_loss": reconstruction_loss.item(),
            "distribution_loss": distribution_loss.item(),
            "vae_loss": loss.item(),
        }
    
    def update_mdn(self, action, z, summary, memory, target_z):
        """
        target_z: (batch, latent_dim)
        mean: (batch, num_distribution, latent_dim)
        std: (batch, num_distribution, latent_dim)
        probability: (batch, num_distribution)
        """   
        summary, memory, probability, mean, std = self.mdn_lstm(action, z, summary, memory)
        
        target_z = target_z.unsqueeze(1)

        normal = torch.distributions.Normal(mean, std)
        element_log_prob = normal.log_prob(target_z)
        sum_log_prob = torch.sum(element_log_prob, dim=-1)

        distribution_log_prob = torch.log(probability + 1e-8)

        log_likelihood = torch.logsumexp(sum_log_prob + distribution_log_prob, dim=1)
        loss = -torch.mean(log_likelihood)
            
        self.mdn_optimizer.zero_grad()
        loss.backward()
        self.mdn_optimizer.step()
        
        return loss.item()
    
    def sample_next_z(self, probability, mean, std):
        distribution_idx = torch.distributions.Categorical(probability).sample()
        batch_idx = torch.arange(probability.shape[0])
        selected_mean = mean[batch_idx, distribution_idx]
        selected_std = std[batch_idx, distribution_idx]
        eps = torch.randn_like(selected_std)
        next_z = selected_mean + eps * selected_std
        return next_z
        
        
class Controller(nn.Module):
    def __init__(self, action_dim, latent_dim=16, hidden_dim=256):
        super().__init__()      
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)
        
    def forward(self, z, summary):
        x = torch.cat([z, summary], dim=-1)
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
        
            
class Agent:
    def __init__(self, action_dim):
        self.controller = Controller(action_dim)                
        num_params = sum(p.numel() for p in self.controller.parameters())        
        self.es = Diagonal_CMA_ES(num_params)
        
    def get_parameters(self):
        params = []
        for p in self.controller.parameters():
            params.append(
                p.detach().cpu().numpy().flatten()
            )
        return np.concatenate(params)
        
    def set_parameters(self, params):
        index = 0
        for p in self.controller.parameters():
            size = p.numel()
            new_values = params[index:index + size]
            new_values = torch.tensor(new_values, dtype=p.dtype).view(p.shape)
            p.data.copy_(new_values)

            index += size
            
    def sample_candidates(self):
        candidates = self.es.sample()
        return candidates
        
    def get_action(self, z, summary):
        with torch.no_grad():
            action = self.controller(z, summary)
        return action


if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    world_model = WorldModel(state_dim, action_dim)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    epoch_count = 0
    best_vae_loss = float("inf")
    states = []
    
    while len(states) < 100000:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            actions = np.random.uniform(-1, 1, (len(agent_ids), action_dim)).astype(np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            state = states_tensor[i].cpu().numpy()                
            states.append(state.copy())
            
    while True:
        indices = np.random.permutation(len(states))
        
        reconstruction_losses = []
        distribution_losses = []
        vae_losses = []
        
        for start in range(0, len(states), batch_size):
            batch_idx = indices[start:start + batch_size]
            states_batch = torch.from_numpy(states[batch_idx])
            metrics = world_model.update_vae(states_batch)
            reconstruction_losses.append(metrics["reconstruction_loss"])
            distribution_losses.append(metrics["distribution_loss"])
            losses.append(metrics["vae_loss"])
            
        epoch_count += 1
        mean_reconstruction_loss = np.mean(reconstruction_losses)
        mean_distribution_loss = np.mean(distribution_losses)
        mean_vae_loss = np.mean(vae_losses)
        writer.add_scalar("Train/reconstruction_loss", mean_reconstruction_loss, epoch_count)
        writer.add_scalar("Train/distribution_loss", mean_distribution_loss, epoch_count)
        writer.add_scalar("Train/vae_loss", mean_vae_loss, epoch_count)
            
        if mean_vae_loss < best_vae_loss:
            best_vae_loss = mean_vae_loss                 
            torch.save(world_model.vae.state_dict(), f"{BASE_DIR}/vae.pth")
            print(f"[Train] Model saved at new best loss {best_vae_loss:.4f}")        
        
