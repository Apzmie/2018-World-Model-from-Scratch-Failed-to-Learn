from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from world_model_2018 import WorldModel, Agent 

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
    
    generation = 0
    best_reward = -float("inf")
    
    while True:
        candidates = agent.sample_candidates()
        rewards = []
        
        for candidate in candidates:
            agent.set_parameters(candidate)
            env.reset()
            
            steps = 0
            total_reward = 0
            
            summary = None
            memory = None
            prev_action = None

            done = False            
            while steps < 1000:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                agent_ids = decision_steps.agent_id
                if len(agent_ids) > 0:
                    num_agents = len(agent_ids)
                    if summary is None:
                        summary = torch.zeros(num_agents, 256)
                        memory = torch.zeros(num_agents, 256)
                        prev_action = torch.zeros(num_agents, action_dim)
                    states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
                    with torch.no_grad():
                        mean, _, _ = world_model.vae.encode(states_tensor)
                        
                        summary, memory, _, _, _ = world_model.mdn_lstm(prev_action, mean, summary, memory)
                        action = agent.get_action(mean, summary)
                        
                    action_np = action.cpu().numpy()
                    env.set_actions(behavior_name, ActionTuple(continuous=action_np))
                    prev_action = action.detach()
                    
                env.step()
                steps += 1
                next_decision_steps, terminal_steps = env.get_steps(behavior_name)
                
                for i, agent_id in enumerate(agent_ids):
                    if agent_id in terminal_steps:
                        reward = terminal_steps[agent_id].reward
                        done = True
                        break
                    elif agent_id in next_decision_steps:
                        reward = next_decision_steps[agent_id].reward
                    else:
                        continue
                        
                    total_reward += reward
                    
            rewards.append(total_reward)
            
        rewards = np.array(rewards)       
        agent.es.update(candidates, rewards)
        agent.set_parameters(agent.es.best_candidate())        
        generation += 1
        
        best = np.max(rewards)
        mean = np.mean(rewards)
        
        writer.add_scalar("Train/best_reward", best, generation)
        writer.add_scalar("Train/mean_reward", mean, generation)
        
        if best > best_reward:
            best_reward = best
            torch.save(agent.controller.state_dict(), f"{BASE_DIR}/controller.pth")
            print(f"[Train] Model saved at new best reward {best_reward:.4f}")                
                
