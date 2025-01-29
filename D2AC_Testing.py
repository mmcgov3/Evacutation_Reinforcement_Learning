"""
ActorCritic_test.py

Script to test an Actor-Critic model on a 10x10 Cell_Space environment.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

from Continuum_Cellspace import Cell_Space, Ob, Ob_size, Exit, agent_size

#########################
# Hyperparameters / Config
#########################
TESTING_DIR = "./testing/ac_test"   # <--- You can change this path as needed
MODEL_PATH = "./model/ActorCritic_example.pth"  # <--- Path to your Actor-Critic checkpoint

MAX_STEPS = 2000
GRID_SIZE = 10    # we create a 10x10 grid
STEP_OFFSET = 0.5 # positions = {0.5, 1.5, ..., 9.5}
#########################

# Build Actor-Critic to match your training code
class ActorCritic(nn.Module):
    def __init__(self, state_size=4, action_size=8):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

        # Kaiming normal as an example
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (state_size, ) or (batch, state_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Actor
        actor_logits = self.actor(x)
        # Critic
        value = self.critic(x)
        return actor_logits, value

def load_actor_critic_model(model_path):
    """
    Loads the ActorCritic network from a checkpoint.
    """
    net = ActorCritic()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it has a different naming, adapt accordingly
        net.load_state_dict(checkpoint)
    net.eval()
    return net

def agent_ac_step(env, net):
    """
    1-step using the ActorCritic model:
      - Build state [x, y, vx, vy]
      - Possibly do the same normalization for x,y that was used in training
      - forward -> get actor_logits
      - sample from torch.distributions.Categorical(logits=actor_logits)
      - env.step(action)
    Returns: next_state, reward, done
    """
    state = np.array([env.agent.position[0],
                      env.agent.position[1],
                      env.agent.velocity[0],
                      env.agent.velocity[1]], dtype=np.float32)
    # If your AC training also normalized XY, do that now:
    state[:2] = env.Normalization_XY(state[:2])

    inp = torch.FloatTensor(state)
    with torch.no_grad():
        actor_logits, value = net(inp)
        # Sample from the categorical
        dist = torch.distributions.Categorical(logits=actor_logits)
        action = dist.sample()

    next_state, reward, done = env.step(action.item())
    return next_state, reward, done

def obstacle_overlap(x, y):
    # Same logic as the DQN script
    if len(Ob) == 0:
        return False
    for idx, obslist in enumerate(Ob):
        for obs in obslist:
            obs_x = obs[0] * 10
            obs_y = obs[1] * 10
            size_obs = Ob_size[idx]
            dis = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dis < (agent_size + size_obs)/2:
                return True
    return False

def distance_from_exit(x, y):
    # If exit = [5,10]
    return np.sqrt((x - 5.0)**2 + (y - 10.0)**2)

def run_ac_test():
    # Setup testing directories
    os.makedirs(TESTING_DIR, exist_ok=True)

    # Initialize environment
    # Domain [0..10], single agent
    from Continuum_Cellspace import Exit, Ob, Ob_size
    # Clear out old obstacles or exits if needed
    Exit.clear()
    Ob.clear()
    Ob_size.clear()

    # Add top-center exit 
    Exit.append(np.array([0.5,1.0,0.5]))

    from Continuum_Cellspace import Cell_Space, delta_t
    env = Cell_Space(0,10,0,10,0,2,rcut=1.5,dt=delta_t,Number=1)

    # Load AC model
    net = load_actor_critic_model(MODEL_PATH)

    results = []
    difference_map = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)

    case_i = 0
    x_vals = [STEP_OFFSET + i for i in range(GRID_SIZE)]  # 0.5..9.5
    y_vals = [STEP_OFFSET + j for j in range(GRID_SIZE)]

    model_steps_list = []
    optimal_steps_list = []
    distances_list = []

    for ix, x_ in enumerate(x_vals):
        for iy, y_ in enumerate(y_vals):
            if obstacle_overlap(x_, y_):
                continue

            dist = distance_from_exit(x_, y_)

            # set start location
            env.reset()
            env.agent.position[0] = x_
            env.agent.position[1] = y_
            env.agent.position[2] = 1.0
            env.agent.velocity[:] = 0.0

            # ============== AC-based steps ==============
            steps_model = 0
            done = False
            while not done and steps_model < MAX_STEPS:
                _, _, done = agent_ac_step(env, net)
                steps_model += 1

            if not done:
                steps_model = MAX_STEPS

            # ============== Optimal steps ==============
            env.reset()
            env.agent.position[0] = x_
            env.agent.position[1] = y_
            env.agent.position[2] = 1.0
            env.agent.velocity[:] = 0.0

            steps_opt = 0
            done_opt = False
            while not done_opt and steps_opt < MAX_STEPS:
                _, _, done_opt = env.step_optimal_single_particle()
                steps_opt += 1

            if not done_opt:
                steps_opt = MAX_STEPS

            results.append((case_i, dist, steps_model, steps_opt))
            difference_map[ix, iy] = steps_model - steps_opt

            distances_list.append(dist)
            model_steps_list.append(steps_model)
            optimal_steps_list.append(steps_opt)

            case_i += 1

    # Write CSV
    csv_path = os.path.join(TESTING_DIR, "test_results_ac.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["case_i", "distance", "model_steps", "optimal_steps"])
        for row in results:
            writer.writerow(row)
    print(f"Saved CSV results to {csv_path}")

    # Scatter Plot
    plt.figure()
    plt.scatter(distances_list, model_steps_list, c='b', label='AC Model Steps')
    plt.scatter(distances_list, optimal_steps_list, c='r', label='Optimal Steps')
    plt.xlabel("Starting Distance from Exit")
    plt.ylabel("Steps to Exit")
    plt.title("Actor-Critic vs. Optimal Steps (Scatter)")
    plt.legend()
    scatter_path = os.path.join(TESTING_DIR, "scatter_ac.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    # Heat Map
    plt.figure()
    im = plt.imshow(difference_map.T, origin='lower', cmap='bwr')
    plt.colorbar(im, label='(Model Steps - Optimal Steps)')
    plt.title("Actor-Critic vs. Optimal: Step Difference Heat Map")
    heatmap_path = os.path.join(TESTING_DIR, "heatmap_ac.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heat map to {heatmap_path}")

    print("Actor-Critic test completed successfully.")


if __name__ == "__main__":
    run_ac_test()
