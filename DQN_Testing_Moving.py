"""
DQN_test.py

Script to test a DQN model on a 10x10 Cell_Space environment, saving environment 
snapshots each step in subfolders for the model-based run and the optimal run,
also saving a CSV, scatter plot, heat map, and a contour+arrow plot of 
the Q-network's chosen actions in [0..1] space.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

from Continuum_Cellspace import *  # Imports Cell_Space, Ob, Ob_size, Exit, agent_size, delta_t

#########################
# Obstacle Parameters (same as in training)
#########################
OBSTACLE_INIT = np.array([0.5, 0.7, 0.5])  # normalized initial position
OBSTACLE_X_MIN = 0.2
OBSTACLE_X_MAX = 0.8
OBSTACLE_SPEED = 0.01   # movement step per simulation step

# Global variables to track obstacle's current normalized x and velocity
obstacle_x = OBSTACLE_INIT[0]
obstacle_x_vel = OBSTACLE_SPEED

#########################
# Testing Config
#########################
TESTING_DIR = "./testing/Continuum_1Exit_ObMoving_DQN_Fully_Pytorch"  # Directory to store outputs
MODEL_PATH = "./model/Continuum_1Exit_ObMoving_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth"

MAX_STEPS = 2000
GRID_SIZE = 10
STEP_OFFSET = 0.5

##############
# DQN Network
##############
class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

        # He init
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
    Loads DQN model from a .pth file.
    Assumes either 'mainQN_state_dict' or 'model_state_dict' for the weights.
    """
    net = DQN()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'mainQN_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['mainQN_state_dict'])
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

#########################
# Obstacle Update Helper
#########################
def update_obstacle(env):
    """
    Updates the obstacle's position so that it oscillates horizontally.
    Uses global variables obstacle_x and obstacle_x_vel.
    The obstacle's new absolute position is computed using the environment's scaling.
    """
    global obstacle_x, obstacle_x_vel
    obstacle_x += obstacle_x_vel
    if obstacle_x > OBSTACLE_X_MAX:
        obstacle_x = OBSTACLE_X_MAX
        obstacle_x_vel = -obstacle_x_vel
    elif obstacle_x < OBSTACLE_X_MIN:
        obstacle_x = OBSTACLE_X_MIN
        obstacle_x_vel = -obstacle_x_vel

    # Compute the new absolute position using env.L (lower and upper bounds)
    new_obs = env.L[:, 0] + np.array([obstacle_x, OBSTACLE_INIT[1], OBSTACLE_INIT[2]]) * (env.L[:, 1] - env.L[:, 0])
    env.Ob[0][0] = new_obs

#########################
# Step Functions
#########################
def agent_dqn_step(env, net, step_counter, model_pathdir):
    """
    - Update obstacle position
    - Save environment snapshot
    - Build state [x, y, vx, vy], normalize (x,y)
    - Forward pass through net => argmax Q => env.step(action)
    """
    update_obstacle(env)
    env.save_output(os.path.join(model_pathdir, f's.{step_counter}'))
    state = np.array([
        env.agent.position[0],
        env.agent.position[1],
        env.agent.velocity[0],
        env.agent.velocity[1]
    ], dtype=np.float32)

    # Normalize XY
    state[:2] = env.Normalization_XY(state[:2])
    
    with torch.no_grad():
        inp = torch.FloatTensor(state).unsqueeze(0)
        Qs = net(inp).numpy()[0]
        action = np.argmax(Qs)

    next_state, reward, done = env.step(action)
    return next_state, reward, done

def agent_optimal_step(env, step_counter, opt_pathdir):
    """
    - Update obstacle position
    - Save environment snapshot
    - Call step_optimal_single_particle() to take the optimal step
    """
    update_obstacle(env)
    env.save_output(os.path.join(opt_pathdir, f's.{step_counter}'))
    next_state, reward, done = env.step_optimal_single_particle()
    return next_state, reward, done

#########################
# Helper Functions
#########################
def obstacle_overlap(x, y):
    """ Check if (x,y) overlaps with obstacles. """
    if len(Ob) == 0:
        return False

    # In our updated code, obstacles in env.Ob are stored in absolute coordinates.
    for idx, obslist in enumerate(Ob):
        for obs in obslist:
            obs_x = obs[0]  # absolute x coordinate
            obs_y = obs[1]  # absolute y coordinate
            size_obs = Ob_size[idx]
            dis = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dis < (agent_size + size_obs) / 2:
                return True
    return False

def distance_from_exit(x, y):
    """ 2D distance from (x,y) to (5,10). """
    exit_x = 5.0
    exit_y = 10.0
    return np.sqrt((x - exit_x)**2 + (y - exit_y)**2)

#########################
# Runners
#########################
def run_trained_model(env, net, case_i, model_pathdir, max_steps=2000):
    """
    Calls agent_dqn_step in a loop, saving snapshots each step.
    Resets the obstacle's state and fixes the random seed for reproducibility.
    Returns the step count used.
    """
    global obstacle_x, obstacle_x_vel
    np.random.seed(case_i)  # Set seed to ensure same starting configuration
    env.reset()
    obstacle_x = OBSTACLE_INIT[0]
    obstacle_x_vel = OBSTACLE_SPEED

    steps = 0
    done = False
    while not done and steps < max_steps:
        _, _, done = agent_dqn_step(env, net, steps, model_pathdir)
        steps += 1
    if not done:
        steps = max_steps
    return steps

def run_optimal_policy(env, case_i, opt_pathdir, max_steps=2000):
    """
    Calls agent_optimal_step in a loop, saving snapshots each step.
    Resets the obstacle's state and fixes the random seed for reproducibility.
    Returns the step count used.
    """
    global obstacle_x, obstacle_x_vel
    np.random.seed(case_i)  # Set the same seed so that starting configuration is identical
    env.reset()
    obstacle_x = OBSTACLE_INIT[0]
    obstacle_x_vel = OBSTACLE_SPEED

    steps = 0
    done = False
    while not done and steps < max_steps:
        _, _, done = agent_optimal_step(env, steps, opt_pathdir)
        steps += 1
    if not done:
        steps = max_steps
    return steps

#########################
# Contour + Arrow Plot
#########################
def create_contour_arrow_plot(net, save_path):
    """
    Replicates the old approach: 
      - Create a grid in [0..1] (after subtracting offset) for x and y.
      - Build input states with a fixed velocity (e.g. vy=0.5).
      - Forward pass through net => determine argmax for each state.
      - Plot a contour (with colors corresponding to actions) and overlay arrow directions.
    """
    offset = [0, 0]

    xlin = np.linspace(0, 1, 100) - offset[0]
    ylin = np.linspace(0, 1, 100) - offset[1]
    x, y = np.meshgrid(xlin, ylin)
    xy = np.vstack([x.ravel(), y.ravel()]).T

    # Use a fixed velocity for testing (for instance, vy=0.5)
    vxy = np.zeros_like(xy)
    vxy[:, 1] = 0.5

    # Build the input states: [x, y, vx, vy]
    xtest = np.hstack([xy, vxy])

    net.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(xtest).float()
        qvals = net(inputs).numpy()
    action_pred = np.argmax(qvals, axis=1)

    action_grid = action_pred.reshape(x.shape)

    # Create a coarser grid for arrow plotting
    xlin_arrow = np.linspace(0.05, 0.95, 15) - offset[0]
    ylin_arrow = np.linspace(0.05, 0.95, 15) - offset[1]
    x_arrow, y_arrow = np.meshgrid(xlin_arrow, ylin_arrow)
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T

    vxy_arrow = np.zeros_like(xy_arrow)
    vxy_arrow[:, 1] = 0.5
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

    with torch.no_grad():
        arrow_qvals = net(torch.from_numpy(x_arrow_test).float()).numpy()
    action_arrow_pred = np.argmax(arrow_qvals, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    contour = ax.contourf(x, y, action_grid + 0.1,
                          cmap=plt.cm.rainbow, alpha=0.8)

    arrow_len = 0.07
    angle = np.sqrt(2) / 2
    arrow_map = {
        0: [0, arrow_len],                   # up
        1: [-angle * arrow_len, angle * arrow_len],
        2: [-arrow_len, 0],
        3: [-angle * arrow_len, -angle * arrow_len],
        4: [0, -arrow_len],
        5: [angle * arrow_len, -angle * arrow_len],
        6: [arrow_len, 0],
        7: [angle * arrow_len, angle * arrow_len],
    }

    for idx, p in enumerate(xy_arrow):
        dx, dy = arrow_map[action_arrow_pred[idx]]
        ax.annotate('', xy=p, xytext=p + np.array([dx, dy]),
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))

    ax.set_title("Contour+Arrows for DQN in [0..1] domain")
    ax.tick_params(labelsize='large')

    plt.savefig(save_path)
    plt.close()
    print(f"Saved contour+arrow plot to {save_path}")

#########################
# Main
#########################
def main():
    # 1) Setup testing directory
    os.makedirs(TESTING_DIR, exist_ok=True)

    # 2) Clear old obstacles/exits, add top-center exit and obstacle
    Exit.clear()
    Ob.clear()
    Ob_size.clear()
    Exit.append(np.array([0.5, 1.0, 0.5]))  # => top-center exit (after scaling, (5,10,1))
    Ob1 = []
    Ob1.append(np.array([0.5, 0.7, 0.5]))  # initial normalized obstacle position
    Ob.append(Ob1)
    Ob_size.append(2.0)

    # Create the environment (10x10)
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=1)

    # 3) Load DQN model
    net = load_dqn_model(MODEL_PATH)

    # 4) Prepare data storage
    results = []
    difference_map = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)

    model_steps_list = []
    optimal_steps_list = []
    distances_list = []

    case_i = 0
    x_vals = [STEP_OFFSET + i for i in range(GRID_SIZE)]
    y_vals = [STEP_OFFSET + j for j in range(GRID_SIZE)]

    # 5) Loop over grid positions
    for ix, x_ in enumerate(x_vals):
        for iy, y_ in enumerate(y_vals):
            if obstacle_overlap(x_, y_):
                continue

            dist = distance_from_exit(x_, y_)
            case_path = os.path.join(TESTING_DIR, f"case_{case_i}")
            os.makedirs(case_path, exist_ok=True)

            model_folder = os.path.join(case_path, "model")
            opt_folder   = os.path.join(case_path, "optimal")
            os.makedirs(model_folder, exist_ok=True)
            os.makedirs(opt_folder, exist_ok=True)

            # Model run
            np.random.seed(case_i)  # Set seed so that both runs share the same starting configuration
            env.reset()
            env.agent.position[:] = (x_, y_, 1.0)
            env.agent.velocity[:] = 0.0
            steps_model = run_trained_model(env, net, case_i, model_folder, max_steps=MAX_STEPS)

            # Optimal run
            np.random.seed(case_i)  # Reset seed to ensure identical starting conditions
            env.reset()
            env.agent.position[:] = (x_, y_, 1.0)
            env.agent.velocity[:] = 0.0
            steps_opt = run_optimal_policy(env, case_i, opt_folder, max_steps=MAX_STEPS)

            results.append((case_i, dist, steps_model, steps_opt, steps_model - steps_opt))
            difference_map[ix, iy] = steps_model - steps_opt

            distances_list.append(dist)
            model_steps_list.append(steps_model)
            optimal_steps_list.append(steps_opt)

            case_i += 1

    # 6) Save CSV results
    csv_path = os.path.join(TESTING_DIR, "test_results_dqn.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["case_i", "distance", "model_steps", "optimal_steps", "Difference"])
        for row in results:
            writer.writerow(row)
    print(f"Saved CSV results to {csv_path}")

    # 7) Scatter plot
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

    # 8) Heat Map
    plt.figure()
    valid_diff = difference_map[~np.isnan(difference_map)]
    max_val = np.abs(valid_diff).max() if valid_diff.size > 0 else 0
    im = plt.imshow(difference_map.T, origin='lower', cmap='bwr', vmin=-max_val, vmax=max_val)
    plt.colorbar(im, label='(Model Steps - Optimal Steps)')
    plt.title("DQN vs. Optimal: Step Difference Heat Map")
    heatmap_path = os.path.join(TESTING_DIR, "heatmap_dqn.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heat map to {heatmap_path}")

    # 9) Contour+Arrow Plot in [0..1]
    contour_path = os.path.join(TESTING_DIR, "contour_arrows.png")
    create_contour_arrow_plot(net, contour_path)

    # Final print of average steps
    model_avg = np.nanmean(model_steps_list)
    opt_avg   = np.nanmean(optimal_steps_list)
    print("\nDQN test completed successfully.")
    print(f"Average DQN steps: {model_avg:.2f} | Average Optimal steps: {opt_avg:.2f}")

if __name__ == "__main__":
    main()
