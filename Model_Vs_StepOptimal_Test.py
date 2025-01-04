import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Continuum_Cellspace2 import *

numExits = 1
numObs = 0
state_size = 4 + 2*numExits + 2*numObs

#############################################
# DQN Class (same as provided)
#############################################
class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # He (Kaiming) normal initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='linear')
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

#############################################
# Load Models
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_saved_path_up = './model/Continuum_1ExitUp_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
model_saved_path_down = './model/Continuum_1ExitDown_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
model_saved_path_2exits = './model/Continuum_2Exit_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
model_saved_path_4exits = './model/Continuum_4Exit_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
#model_saved_path_3exits_ob = './model/Continuum_3Exits_Ob_DQN_Fully.pth'
model_saved_path_ob_center = './model/Continuum_1Exit_Ob_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
#model_saved_path_ob_up = './model/Continuum_Ob_Up_DQN_Fully.pth'
model_saved_1Exit_rand = './model/Continuum_1ExitRandom_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep20000.pth'
model_saved_2Exit_rand = './model/Continuum_2ExitRandom_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep20000.pth'

# Load the models
checkpoint_1_up = torch.load(model_saved_path_up, map_location=device)
mainQN_Up = DQN().to(device)
mainQN_Up.load_state_dict(checkpoint_1_up['mainQN_state_dict'])
mainQN_Up.eval()

checkpoint_1_down = torch.load(model_saved_path_down, map_location=device)
mainQN_down = DQN().to(device)
mainQN_down.load_state_dict(checkpoint_1_down['mainQN_state_dict'])
mainQN_down.eval()

checkpoint_2 = torch.load(model_saved_path_2exits, map_location=device)
mainQN_2Exits = DQN().to(device)
mainQN_2Exits.load_state_dict(checkpoint_2['mainQN_state_dict'])
mainQN_2Exits.eval()

checkpoint_4 = torch.load(model_saved_path_4exits, map_location=device)
mainQN_4Exits = DQN().to(device)
mainQN_4Exits.load_state_dict(checkpoint_4['mainQN_state_dict'])
mainQN_4Exits.eval()

#mainQN_3Exits_Ob = DQN().to(device)
#mainQN_3Exits_Ob.load_state_dict(torch.load(model_saved_path_3exits_ob, map_location=device))

checkpoint_1_ob_center = torch.load(model_saved_path_ob_center, map_location=device)
mainQN_Ob_center = DQN().to(device)
mainQN_Ob_center.load_state_dict(checkpoint_1_ob_center['mainQN_state_dict'])
mainQN_Ob_center.eval()

checkpoint_1_random = torch.load(model_saved_1Exit_rand, map_location=device)
mainQN_1RAND = DQN(state_size=state_size).to(device)
mainQN_1RAND.load_state_dict(checkpoint_1_random['mainQN_state_dict'])
mainQN_1RAND.eval()

# checkpoint_2_random = torch.load(model_saved_2Exit_rand, map_location=device)
# mainQN_2RAND = DQN(state_size=state_size).to(device)
# mainQN_2RAND.load_state_dict(checkpoint_2_random['mainQN_state_dict'])
# mainQN_2RAND.eval()

#############################################
# Utility Functions
#############################################

def run_episode_with_model(env, model, start_pos, max_steps=1500):
    """
    Run an episode from a given start position using the trained DQN model.
    Returns the cumulative reward and the number of steps until done.
    """
    # Reset environment and set the agent to the desired start_pos
    state = env.reset()   
    # Manually set agent position and velocity
    # Assuming Z dimension is fixed as in the environment initialization
    env.agent.position[0] = start_pos[0]
    env.agent.position[1] = start_pos[1]
    env.agent.velocity[:] = 0.0

    cumulative_reward = 0.0
    done = False
    steps = 0

    for _ in range(max_steps):
        steps += 1
        # Prepare state for model
        # State: (x, y, vx, vy)
        current_state = np.array([env.agent.position[0], env.agent.position[1],
                                  env.agent.velocity[0], env.agent.velocity[1]])
        # Normalize position features
        current_state[:2] = env.Normalization_XY(current_state[:2])
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            Qs = model(state_tensor).cpu().numpy()[0]
        # Choose the best action
        action = np.random.choice(np.where(Qs == Qs.max())[0])
        
        next_state, reward, done = env.step(action)
        cumulative_reward += reward

        if done:
            break

    return cumulative_reward, steps



def run_episode_with_optimal(env, start_pos, max_steps=1500):
    """
    Run an episode from a given start position using the environment's step_optimal function.
    Returns the cumulative reward and the number of steps until done.
    """
    state = env.reset()
    # Manually set agent position and velocity
    env.agent.position[0] = start_pos[0]
    env.agent.position[1] = start_pos[1]
    env.agent.velocity[:] = 0.0

    cumulative_reward = 0.0
    done = False
    steps = 0

    for _ in range(max_steps):
        steps += 1
        # The step_optimal function moves all particles optimally towards the exit
        # But we still need to track the agent (ID=0) reward each timestep.
        # The environment in your code does not return reward for step_optimal.
        # However, we know from the code that the "reward" at each step is the same as in step(action).
        # For fairness, we assume the same step cost and end reward as defined in the environment.
        # So we manually replicate what "reward" would have been:
        # The environment code sets reward = self.reward (which is -0.1 per step), and end_reward = 0 when done.
        
        done = env.step_optimal()  # This doesn't return next_state or reward, so we deduce reward ourselves
        step_reward = env.reward if not done else env.end_reward
        cumulative_reward += step_reward

        if done:
            # Agent reached the exit, final reward is end_reward = 0 added on top, but 0 won't change cumulative.
            break

    return cumulative_reward, steps

#############################################
# Testing multiple start positions
#############################################

# We will test across a grid in the environment:
# Let's say environment is from [0,10]x[0,10]. We'll pick a grid of start positions.
x_positions = np.linspace(1, 9, 5)  # adjust the number of test points as needed
y_positions = np.linspace(1, 9, 5)

#Exit.clear()
#Exit.append( np.array([0.5, 1.0, 0.5]) )   ##Add Up exit
#Exit.append( np.array([0.5, 0, 0.5]) )     ##Add Down Exit
#Exit.append( np.array([0, 0.5, 0.5]) )     ##Add Left exit
#Exit.append( np.array([1.0, 0.5, 0.5]) )   ##Add Right Exit

#Ob.clear()
#Ob_size.clear()
# Ob1 = []
# Ob1.append(np.array([0.5, 0.7, 0.5]))
# Ob.append(Ob1)
# Ob_size.append(2.0)

env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=1)
# env.Exit.append( np.array([0.5, 1.0, 0.5]) )   ##Add Up exit

model_rewards = np.zeros((len(x_positions), len(y_positions)))
optimal_rewards = np.zeros((len(x_positions), len(y_positions)))

for i, x in enumerate(x_positions):
    for j, y in enumerate(y_positions):
        start_pos = (x, y)

        # Run episode with model
        m_reward, m_steps = run_episode_with_model(env, mainQN_1RAND, start_pos)
        
        # Run episode with optimal policy
        o_reward, o_steps = run_episode_with_optimal(env, start_pos)

        model_rewards[i, j] = m_reward
        optimal_rewards[i, j] = o_reward

        print(f"Start Pos: ({x:.2f}, {y:.2f}) | Model Reward: {m_reward:.2f}, Optimal Reward: {o_reward:.2f}")

#############################################
# Print and plot results
#############################################

print("\nModel Rewards:\n", model_rewards)
print("\nOptimal Rewards:\n", optimal_rewards)

# Compute difference (model - optimal)
diff_rewards = model_rewards - optimal_rewards

# Plotting as a heatmap:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im1 = axs[0].imshow(model_rewards, origin='lower', extent=(y_positions.min(), y_positions.max(), x_positions.min(), x_positions.max()), aspect='auto')
axs[0].set_title('Model Rewards')
axs[0].set_xlabel('Y Position')
axs[0].set_ylabel('X Position')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(optimal_rewards, origin='lower', extent=(y_positions.min(), y_positions.max(), x_positions.min(), x_positions.max()), aspect='auto')
axs[1].set_title('Optimal Rewards')
axs[1].set_xlabel('Y Position')
axs[1].set_ylabel('X Position')
fig.colorbar(im2, ax=axs[1])

im3 = axs[2].imshow(diff_rewards, origin='lower', extent=(y_positions.min(), y_positions.max(), x_positions.min(), x_positions.max()), aspect='auto')
axs[2].set_title('Difference (Model - Optimal)')
axs[2].set_xlabel('Y Position')
axs[2].set_ylabel('X Position')
fig.colorbar(im3, ax=axs[2])

plt.tight_layout()
plt.show()
