"""
DQN_test.py

Script to test a DQN model on a 10x10 Cell_Space environment, saving environment 
snapshots each step in subfolders for the model-based run and the optimal run,
also saving a CSV, scatter plot, heat map, and now a contour+arrow plot of 
the Q-network's chosen actions in [0..1] space (like the old approach).
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
TESTING_DIR = "./testing/DQN_1Exit_Ob_CornerSampling"  # Directory to store outputs
MODEL_PATH = "./model/Continuum_1Exit_Ob_DQN_CornerSampling_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth"

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


##################
# Step Functions
##################
def agent_dqn_step(env, net, step_counter, model_pathdir):
    """
    - Save environment snapshot
    - Build state [x, y, vx, vy], normalize (x,y)
    - Argmax Q => env.step(action)
    """
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
    - Save environment snapshot
    - step_optimal_single_particle()
    """
    env.save_output(os.path.join(opt_pathdir, f's.{step_counter}'))
    next_state, reward, done = env.step_optimal_single_particle()
    return next_state, reward, done


##################
# Helper Functions
##################
def obstacle_overlap(x, y):
    """ Check if (x,y) overlaps with obstacles. """
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
    """ 2D distance from (x,y) to (5,10). """
    exit_x = 5.0
    exit_y = 10.0
    return np.sqrt((x - exit_x)**2 + (y - exit_y)**2)


##################
# Runners
##################
def run_trained_model(env, net, case_i, model_pathdir, max_steps=2000):
    """
    Calls agent_dqn_step in a loop, saving snapshots each step.
    Returns step count used.
    """
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
    Calls step_optimal_single_particle in a loop, saving snapshots each step.
    Returns step count used.
    """
    steps = 0
    done = False
    while not done and steps < max_steps:
        _, _, done = agent_optimal_step(env, steps, opt_pathdir)
        steps += 1
    if not done:
        steps = max_steps
    return steps


##################
# Contour + Arrow Plot
##################
def create_contour_arrow_plot(net, save_path):
    """
    Replicates the old approach: 
      - Make a grid [0..1] (with offset=[0,0]) => x, y
      - Assign some velocity (0.5 in y or 0 in x, etc.)
      - Forward pass => argmax => color contour
      - Overlay arrow directions for each action
      - Save as 'contour_arrows.png'
    """
    offset = [0,0]

    # We'll create a grid from [0..1] in x and y, 100 points each
    # Then subtract offset (which is zero), so effectively [0..1].
    # In the old script, it then plots from -0.5..0.5 in each axis, 
    # but let's keep it consistent that we show 0..1 or -0.5..0.5?
    # The old code: fig, ax => xlim=(-0.5,0.5), ylim=(-0.5,0.5).
    # We'll replicate that exactly.
    xlin = np.linspace(0, 1, 100) - offset[0]  # effectively [0..1]
    ylin = np.linspace(0, 1, 100) - offset[1]  # effectively [0..1]
    x, y = np.meshgrid(xlin, ylin)
    xy = np.vstack([x.ravel(), y.ravel()]).T

    # We'll do constant velocity: (vx=0, vy=0.5), for example
    # The old script sets vxy[:,1] = 0.5, 
    vxy = np.zeros_like(xy)
    vxy[:,1] = 0.5

    # Build the input states = [x, y, vx, vy]
    xtest = np.hstack([xy, vxy])

    # Forward pass
    net.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(xtest).float()
        qvals = net(inputs).numpy()
    action_pred = np.argmax(qvals, axis=1)

    # Reshape for contour
    action_grid = action_pred.reshape(x.shape)

    # We'll also create a coarser grid for arrow plots
    xlin_arrow = np.linspace(0.05, 0.95, 15) - offset[0]
    ylin_arrow = np.linspace(0.05, 0.95, 15) - offset[1]
    x_arrow, y_arrow = np.meshgrid(xlin_arrow, ylin_arrow)
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T

    # Same velocity approach
    vxy_arrow = np.zeros_like(xy_arrow)
    vxy_arrow[:,1] = 0.5
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

    with torch.no_grad():
        arrow_qvals = net(torch.from_numpy(x_arrow_test).float()).numpy()
    action_arrow_pred = np.argmax(arrow_qvals, axis=1)

    # Create figure
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)


    # Contour of actions
    contour = ax.contourf(x, y, action_grid+0.1,
                          cmap=plt.cm.rainbow, alpha=0.8)

    # Action -> arrow direction map
    arrow_len = 0.07
    angle = np.sqrt(2)/2
    arrow_map = {
        0: [0, arrow_len],                  # up
        1: [-angle*arrow_len, angle*arrow_len],
        2: [-arrow_len, 0],
        3: [-angle*arrow_len, -angle*arrow_len],
        4: [0, -arrow_len],
        5: [angle*arrow_len, -angle*arrow_len],
        6: [arrow_len, 0],
        7: [angle*arrow_len, angle*arrow_len],
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


##################
# Main
##################
def main():
    # 1) Setup testing directory
    os.makedirs(TESTING_DIR, exist_ok=True)

    # 2) Clear old obstacles/exits, add top-center exit
    Exit.clear()
    Ob.clear()
    Ob_size.clear()
    Exit.append(np.array([0.5,1.0,0.5]))  # => (5,10,1) in domain
    Ob1 = []
    Ob1.append(np.array([0.5, 0.7, 0.5]))
    Ob.append(Ob1)
    Ob_size.append(2.0)

    from Continuum_Cellspace import Cell_Space, delta_t
    env = Cell_Space(0,10,0,10,0,2,rcut=1.5,dt=delta_t,Number=1)

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

    # 5) Loop over grid
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
            env.reset()
            env.agent.position[:] = (x_, y_, 1.0)
            env.agent.velocity[:] = 0.
            steps_model = run_trained_model(env, net, case_i, model_folder, max_steps=MAX_STEPS)

            # Optimal run
            env.reset()
            env.agent.position[:] = (x_, y_, 1.0)
            env.agent.velocity[:] = 0.
            steps_opt = run_optimal_policy(env, case_i, opt_folder, max_steps=MAX_STEPS)

            results.append((case_i, dist, steps_model, steps_opt, steps_model-steps_opt))
            difference_map[ix, iy] = steps_model - steps_opt

            distances_list.append(dist)
            model_steps_list.append(steps_model)
            optimal_steps_list.append(steps_opt)

            case_i += 1

    # 6) Save CSV
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

    # Mask out NaN entries
    valid_diff = difference_map[~np.isnan(difference_map)]
    # Compute absolute max among valid entries
    max_val = np.abs(valid_diff).max() if valid_diff.size > 0 else 0

    im = plt.imshow(difference_map.T, origin='lower', cmap='bwr',
                    vmin=-max_val, vmax=max_val)
    plt.colorbar(im, label='(Model Steps - Optimal Steps)')
    plt.title("DQN vs. Optimal: Step Difference Heat Map")
    heatmap_path = os.path.join(TESTING_DIR, "heatmap_dqn.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heat map to {heatmap_path}")


    # 9) Contour+Arrow Plot in [0..1]
    contour_path = os.path.join(TESTING_DIR, "contour_arrows.png")
    create_contour_arrow_plot(net, contour_path)

    # (Optional) final print
    model_avg = np.nanmean(model_steps_list)
    opt_avg   = np.nanmean(optimal_steps_list)
    print("\nDQN test completed successfully.")
    print(f"Average DQN steps: {model_avg:.2f} | Average Optimal steps: {opt_avg:.2f}")


if __name__ == "__main__":
    main()