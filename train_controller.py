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
    world_model.vae.load_state_dict(torch.load(f"{BASE_DIR}/vae.pth"))
    world_model.mdn_lstm.load_state_dict(torch.load(f"{BASE_DIR}/mdn_lstm.pth"))
    agent = Agent(action_dim)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    best_reward = -float("inf")
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            with torch.no_grad():
                _, _, start_z = world_model.vae.encode(states_tensor)
                
            candidates = agent.sample_candidates()
            rewards = []
            
            for params in candidates:
                agent.set_parameters(params)              
                z = start_z.clone()
                num_agents = len(agent_ids)
                
                summary = torch.zeros(num_agents, 256)
                memory = torch.zeros(num_agents, 256)                
                total_reward = torch.zeros(num_agents)
                
                for t in range(1000):
                    with torch.no_grad():
                        action = agent.controller(z, summary)
                        
                        summary, memory, probability, mean, std, reward = world_model.mdn_lstm(action, z, summary, memory)

                        z = world_model.sample_next_z(probability, mean, std)
                        
                        total_reward += reward.squeeze(-1)
                        
                rewards.append(total_reward.mean().item())
            
            rewards = np.array(rewards)
            current_best = rewards.max()
            
            agent.es.update(candidates, rewards) 
            
            if current_best > best_reward:
                best_reward = current_best
                
                agent.set_parameters(agent.es.best_candidate())
                torch.save(agent.controller.state_dict(), f"{BASE_DIR}/controller.pth")
                print(f"[Train] Model saved at new best reward {best_reward:.4f}") 
            
            else:
                agent.set_parameters(agent.es.best_candidate())
                               
        env.step()
       
