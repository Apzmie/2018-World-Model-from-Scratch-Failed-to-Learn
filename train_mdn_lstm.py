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
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    total_steps = 0
    trajectories = {}
    best_loss = float("inf")

    while total_steps < 100000:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            state_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            random_action = np.random.uniform(-1, 1, (len(agent_ids), action_dim)).astype(np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=random_action))
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        terminal_ids = set(terminal_steps.agent_id)
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in trajectories:
                trajectories[agent_id] = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                    "dones": []
                }
                       
            if agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
            else:
                continue
                
            trajectories[agent_id]["states"].append(state_tensor[i].numpy().copy())
            trajectories[agent_id]["actions"].append(random_action[i].copy())
            trajectories[agent_id]["rewards"].append(reward.copy())
            trajectories[agent_id]["dones"].append(agent_id in terminal_ids)
        
        total_steps += len(agent_ids)
            
    
    while True:
        total_loss = 0
        count = 0
        
        for agent_id, traj in trajectories.items():
            states = torch.tensor(np.array(traj["states"]), dtype=torch.float32)
            actions = torch.tensor(np.array(traj["actions"]), dtype=torch.float32)
            dones = traj["dones"]

            with torch.no_grad():
                _, _, z = world_model.vae.encode(states)

            summary = torch.zeros(1,256)
            memory = torch.zeros(1,256)

            for t in range(len(actions)-1):
                loss, summary, memory = world_model.update_mdn(actions[t:t+1], z[t:t+1], summary, memory, z[t+1:t+2], rewards[t:t+1])
                    
                total_loss = total_loss + loss

                count += 1
                if dones[t]:
                    summary.zero_()
                    memory.zero_()
        
        total_loss = total_loss / count
        
        world_model.mdn_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.mdn_lstm.parameters(), 1.0)
        world_model.mdn_optimizer.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()                 
            torch.save(world_model.mdn_lstm.state_dict(), f"{BASE_DIR}/mdn_lstm.pth")
            print(f"[Train] Model saved at new best loss {best_loss:.4f}") 
        
