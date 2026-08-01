from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from world_model_2018 import WorldModel       

BASE_DIR = ""


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
    
    batch_size=256
    
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
        
        for i, agent_id in enumerate(agent_ids):
            state = states_tensor[i].cpu().numpy()                
            states.append(state.copy())
        states = np.array(states, dtype=np.float32)
            
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
            vae_losses.append(metrics["vae_loss"])
            
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
        
