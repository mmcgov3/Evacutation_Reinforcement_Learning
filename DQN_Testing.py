"""
DQN_test.py

Script to test a DQN model on a 10x10 Cell_Space environment.
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
TESTING_DIR = "./testing/dqn_test"   # <--- You can change this path as needed
MODEL_PATH = "./model/DQN_example.pth"  # <--- Path to your DQN checkpoint

MAX_STEPS = 2000
GRID_SIZE = 10    # we create a 10x10 grid
STEP_OFFSET = 0.5 # positions = {0.5, 1.5, ..., 9.5}
#########################

# Build DQN architecture to match your training code
class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

        # He init as in your training
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

def load_dqn_model(model_path):
    """
    Loads DQN model from checkpoint. 
    We assume your .pth includes something like 'mainQN_state_dict' for the weights.
    """
    net = DQN()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # If your checkpoint has a key 'mainQN_state_dict', load it:
    if 'mainQN_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['mainQN_state_dict'])
    else:
        # Otherwise, maybe the checkpoint directly has 'model_state_dict'
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

def agent_dqn_step(env, net):
    """
    1-step using a DQN model:
      - Build state [x, y, vx, vy]
      - Normalize XY
      - forward -> pick argmax
      - env.step(action)
    Returns: next_state, reward, done
    """
    state = np.array([env.agent.position[0],
                      env.agent.position[1],
                      env.agent.velocity[0],
                      env.agent.velocity[1]], dtype=np.float32)
    # Normalization like in your DQN code
    state[:2] = env.Normalization_XY(state[:2])
    with torch.no_grad():
        inp = torch.FloatTensor(state).unsqueeze(0)
        Qs = net(inp).numpy()[0]
        action = np.argmax(Qs)
    next_state, reward, done = env.step(action)
    return next_state, reward, done

def obstacle_overlap(x, y):
    """
    Check if (x, y) is overlapping with any obstacle in the global environment data.
    We'll consider each obstacle's position and size.
    """
    # If no obstacles, just return False
    if len(Ob) == 0:
        return False
    # We'll reconstruct the actual absolute obstacle positions
    # the environment normally does that inside Cell_Space
    # but let's do a quick check with the environment logic:
    #   ob_global = L[:,0] + ob_local * (L[:,1] - L[:,0])
    # For testing, we assume a single environment with [0,10], [0,10].
    # If your environment modifies this differently, we might have to replicate that logic.
    # We'll just do a direct approach: 
    #   for each obstacle i in Ob[i], Ob_size[i]
    #   check if distance < (agent_size + Ob_size[i]) / 2
    #   But you haven't shown how you store multiple positions in Ob[i].
    # We'll do a naive approach:
    for idx, obslist in enumerate(Ob):
        for obs in obslist:
            # obs is in normalized coords? or direct? 
            # Typically it's in normalized [0..1]. For domain [0..10], let's do:
            obs_x = obs[0] * 10
            obs_y = obs[1] * 10
            # size in Ob_size?
            size_obs = Ob_size[idx]
            dis = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dis < (agent_size + size_obs)/2:
                return True
    return False

def distance_from_exit(x, y):
    """
    Compute 2D distance from (x,y) to the exit, which is presumably at [5, 10]
    if we use Exit.append(np.array([0.5, 1.0, 0.5]))
    We'll do a direct Euclidian distance in XY plane ignoring Z.
    """
    exit_x = 5.0
    exit_y = 10.0
    return np.sqrt((x - exit_x)**2 + (y - exit_y)**2)

def run_dqn_test():
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
    # Normalized [0.5, 1.0, 0.5] => actual [5,10,1] in the domain [0..10]
    Exit.append(np.array([0.5,1.0,0.5]))

    # Build environment
    from Continuum_Cellspace import Cell_Space, delta_t
    env = Cell_Space(0,10,0,10,0,2,rcut=1.5,dt=delta_t,Number=1)

    # Load DQN model
    net = load_dqn_model(MODEL_PATH)

    # Prepare data storage
    results = []  # list of (case_i, distance, model_steps, optimal_steps)
    difference_map = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)

    case_i = 0
    # We'll define X coords = [0.5,1.5,...,9.5], same for Y
    x_vals = [STEP_OFFSET + i for i in range(GRID_SIZE)]  # 0.5..9.5
    y_vals = [STEP_OFFSET + j for j in range(GRID_SIZE)]

    # For scatter plot
    model_steps_list = []
    optimal_steps_list = []
    distances_list = []

    for ix, x_ in enumerate(x_vals):
        for iy, y_ in enumerate(y_vals):
            # Check obstacle overlap
            if obstacle_overlap(x_, y_):
                # skip
                continue

            # Distance
            dist = distance_from_exit(x_, y_)

            # Place agent at (x_, y_)
            env.reset()  # Then override position
            env.agent.position[0] = x_
            env.agent.position[1] = y_
            env.agent.position[2] = 1.0  # if you want z=1, you can set it or 0.5, etc.
            env.agent.velocity[:] = 0.0  # start with zero velocity

            # ============== DQN-based steps ==============
            steps_model = 0
            done = False
            while not done and steps_model < MAX_STEPS:
                _, _, done = agent_dqn_step(env, net)
                steps_model += 1

            if not done:
                # We might say it fails or just record steps_model=MAX_STEPS
                # We'll do that
                steps_model = MAX_STEPS

            # ============== Optimal steps ==============
            # reset to same position
            env.reset()
            env.agent.position[0] = x_
            env.agent.position[1] = y_
            env.agent.position[2] = 1.0
            env.agent.velocity[:] = 0.0

            steps_opt = 0
            done_opt = False
            while not done_opt and steps_opt < MAX_STEPS:
                ns, rew, done_opt = env.step_optimal_single_particle()
                steps_opt += 1

            if not done_opt:
                steps_opt = MAX_STEPS

            results.append((case_i, dist, steps_model, steps_opt))
            difference_map[ix, iy] = steps_model - steps_opt

            distances_list.append(dist)
            model_steps_list.append(steps_model)
            optimal_steps_list.append(steps_opt)

            case_i += 1

    # =========================
    # Write CSV
    # =========================
    csv_path = os.path.join(TESTING_DIR, "test_results_dqn.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["case_i", "distance", "model_steps", "optimal_steps"])
        for row in results:
            writer.writerow(row)
    print(f"Saved CSV results to {csv_path}")

    # =========================
    # Scatter Plot
    # X-axis = distance, Y-axis = steps
    # We'll plot two sets of points: model vs. optimal
    # =========================
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(distances_list, model_steps_list, c='b', label='DQN Model Steps')
    plt.scatter(distances_list, optimal_steps_list, c='r', label='Optimal Steps')
    plt.xlabel("Starting Distance from Exit")
    plt.ylabel("Steps to Exit")
    plt.title("DQN vs. Optimal Steps (Scatter)")
    plt.legend()
    scatter_path = os.path.join(TESTING_DIR, "scatter_dqn.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    # =========================
    # Heat Map
    # difference_map[ix, iy] = (model_steps - optimal_steps)
    # We need to show it as a 2D array
    # We'll define x axis as i, y axis as j. 
    # Some cells are np.nan if obstacle. 
    # =========================
    plt.figure()
    # If you want (0,0) at bottom-left, we might do origin='lower'
    # but by default, imshow might put (0,0) top-left.
    # We'll just do a straightforward approach:
    im = plt.imshow(difference_map.T, origin='lower', cmap='bwr')
    plt.colorbar(im, label='(Model Steps - Optimal Steps)')
    plt.title("DQN vs. Optimal: Step Difference Heat Map")
    heatmap_path = os.path.join(TESTING_DIR, "heatmap_dqn.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heat map to {heatmap_path}")

    print("DQN test completed successfully.")


if __name__ == "__main__":
    run_dqn_test()
