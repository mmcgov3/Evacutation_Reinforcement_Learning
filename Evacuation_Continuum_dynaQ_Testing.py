import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from Continuum_Cellspace_Gym import ParticleNavEnv

# Import your environment and agent code
# from your_dyna_q_code import ParticleNavEnv, DynaQAgent, QNetwork

#######################################
# Configuration Variables (Adjust as needed)
#######################################
env_size = 50.0  # Must match the environment's settings
num_points = 50   # Number of points per dimension in the grid
arrow_points = 15 # Number of points per dimension for arrow plotting
model_step = 5000 # The timestep at which the QNetwork was saved
qnetwork_save_dir = "./model/1ExitDyna/QNetwork"
output_dir = "./policy_plots"
os.makedirs(output_dir, exist_ok=True)

# Load environment
env = ParticleNavEnv(env_size=env_size, num_exits=1, num_obstacles=1, change_env_interval=200)
# Reset environment to get initial exit positions, etc.
env.reset()

# Load a QNetwork that matches your trained model architecture and hyperparameters
# Adjust hidden_size if you used a different size, and ensure state & action dims match training
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_size = 64

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_size)
        self.elu = nn.ELU()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.output_layer(x)
        return x

qnetwork = QNetwork(state_dim, action_dim, hidden_size=hidden_size)
model_path = os.path.join(qnetwork_save_dir, f"qnetwork_{model_step}.pth")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

qnetwork.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
qnetwork.eval()

# We will produce a grid of points and evaluate the best action at each point.
# We'll fix velocity at zero and attempt to compute relative exit position and obstacle distances.

# Extract the exit position (assuming we have one exit)
# env.Exits is a list of exit positions in absolute coordinates [x,y,z]
if len(env.Exit) > 0:
    exit_pos = env.Exit[0][:2]  # [ex, ey]
else:
    # If no exits found, just assume center top exit
    exit_pos = np.array([env_size/2, env_size]) 

# For simplicity:
# - Set vx, vy = 0
# - Compute rel_exit_x, rel_exit_y as exit_pos - agent_pos
# - Set obstacle distances to some large value if we cannot compute them easily

# Get the total state dimension from the environment (may include obstacle distances)
# We'll guess obstacle distance count from difference (for demonstration)
# If your state is [px, py, vx, vy, rel_exit_x, rel_exit_y, ... 8 obstacle distances ...]
# That would be 6 + 8 = 14 dim. Adjust this logic based on your actual state definition.
# Here, we assume the state includes:
# px, py, vx, vy, rel_ex_x, rel_ex_y, and 8 obstacle distances = total 14 dim
# If your state format differs, please adjust accordingly.

num_obstacle_dirs = state_dim - 6  # if state_dim > 6, rest are obstacle distances
if num_obstacle_dirs < 0:
    num_obstacle_dirs = 0  # In case state_dim <=6

# We'll set obstacle distances to env_size as if no obstacles are near
max_sensor_range = env_size

# Create the grid of positions (px, py)
px_lin = np.linspace(0, env_size, num_points)
py_lin = np.linspace(0, env_size, num_points)
px_grid, py_grid = np.meshgrid(px_lin, py_lin)

# Create a smaller grid for arrows
px_arrow_lin = np.linspace(env_size*0.05, env_size*0.95, arrow_points)
py_arrow_lin = np.linspace(env_size*0.05, env_size*0.95, arrow_points)
px_arrow_grid, py_arrow_grid = np.meshgrid(px_arrow_lin, py_arrow_lin)

# Construct states
def construct_state(px, py):
    vx = 0.0
    vy = 0.0
    rel_exit_x = exit_pos[0] - px
    rel_exit_y = exit_pos[1] - py
    # obstacle distances
    obstacle_distances = [max_sensor_range]*num_obstacle_dirs
    state = np.array([px, py, vx, vy, rel_exit_x, rel_exit_y] + obstacle_distances, dtype=np.float32)
    return state

# Flatten the grids and create a batch of states for vectorized prediction
positions = np.vstack([px_grid.ravel(), py_grid.ravel()]).T
states = np.array([construct_state(p[0], p[1]) for p in positions])

positions_arrow = np.vstack([px_arrow_grid.ravel(), py_arrow_grid.ravel()]).T
states_arrow = np.array([construct_state(p[0], p[1]) for p in positions_arrow])

# Convert to torch and evaluate Q-network
with torch.no_grad():
    q_values = qnetwork(torch.from_numpy(states))
    actions = q_values.argmax(dim=1).cpu().numpy()

    q_values_arrow = qnetwork(torch.from_numpy(states_arrow))
    actions_arrow = q_values_arrow.argmax(dim=1).cpu().numpy()

# Reshape actions back into grid
action_grid = actions.reshape(px_grid.shape)

# Create a mapping from action index to arrow vector (assuming the same 8-direction mapping as before)
# Actions: 0:Up, 1:Up-Left, 2:Left, 3:Down-Left, 4:Down, 5:Down-Right, 6:Right, 7:Up-Right
arrow_len = env_size * 0.02 # scale arrow length based on env_size
angle = np.sqrt(2)/2
arrow_map = {
    0: (0, arrow_len),           # Up
    1: (-angle*arrow_len, angle*arrow_len), # Up-Left
    2: (-arrow_len, 0),          # Left
    3: (-angle*arrow_len, -angle*arrow_len),# Down-Left
    4: (0, -arrow_len),          # Down
    5: (angle*arrow_len, -angle*arrow_len), # Down-Right
    6: (arrow_len, 0),           # Right
    7: (angle*arrow_len, angle*arrow_len)   # Up-Right
}

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_xlim(0, env_size)
ax.set_ylim(0, env_size)
ax.set_title("Policy Visualization (Best Actions)")

# Contour plot of actions
contour = ax.contourf(px_grid, py_grid, action_grid + 0.1, cmap=plt.cm.get_cmap('rainbow'), alpha=0.8)

# Add arrows
for idx, p in enumerate(positions_arrow):
    dx, dy = arrow_map[actions_arrow[idx]]
    ax.annotate('', xy=p, xytext=(p[0]+dx, p[1]+dy),
                arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))

# Mark the exit position
ax.plot(exit_pos[0], exit_pos[1], 'rx', markersize=10, label="Exit")

ax.legend(loc="upper right")
ax.tick_params(labelsize='large')

# Save figure
figure_filename = f"policy_map_{model_step}.png"
figure_filepath = os.path.join(output_dir, figure_filename)
plt.savefig(figure_filepath, dpi=300)
plt.close(fig)

print(f"Policy figure saved at: {figure_filepath}")
