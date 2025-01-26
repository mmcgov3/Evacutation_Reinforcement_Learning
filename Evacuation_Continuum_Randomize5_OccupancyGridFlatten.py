import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Import your environment returning: (occ_grid_2D, (ax, ay, vx, vy))
# ----------------------------------------------------------------------
from Continuum_Cellspace6_OccupancyGrid import Cell_Space

# Set random seeds for reproducibility
np.random.seed(52)
torch.manual_seed(52)

########################
# Customizable parameters
########################
Number_Agent = 1

# Suppose we do a 10×10 domain => 20×20 occupancy grid => 400 cells + 4 agent scalars => 404 input dims
output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'OccupancyGrid_Flattened_DQN_1Exit_0Ob')
model_saved_path = os.path.join(model_saved_path, 'OccupancyGrid_Flattened_DQN_1Exit_0Ob')

# Hyperparameters and config
train_episodes = 50000
max_steps = 2000
gamma = 0.999

explore_start = 1.0
explore_stop = 0.1
decay_percentage = 0.5
decay_rate = 4 / decay_percentage

learning_rate = 1e-4
memory_size = 10000
batch_size = 50
pretrain_length = batch_size

update_target_every = 1
tau = 0.1
save_step = 1000
train_step = 1
Cfg_save_freq = 1000
cfg_save_step = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# 1) DQN that flattens occupancy grid + agent
#############################################
class DQN(nn.Module):
    def __init__(self, grid_rows=20, grid_cols=20, action_size=8):
        """
        We assume domain=10x10 => 20x20 grid => 400 cells + 4 agent scalars => 404 input.
        """
        super(DQN, self).__init__()

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.action_size = action_size

        input_dim = grid_rows * grid_cols + 4  # e.g. 400 + 4 => 404

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # He / Kaiming init
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='linear')
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, batch_states):
        """
        batch_states: list of states, each => (occ_grid_2D, (ax, ay, vx, vy))
        We'll flatten the grid => shape( rows*cols ) and append the 4 agent scalars => total= rows*cols + 4.

        Key fix: we convert the Python list 'flattened_input' to ONE NumPy array,
        then to a PyTorch tensor. This removes the slow/warning scenario.
        """
        if not isinstance(batch_states, list):
            batch_states = [batch_states]

        flattened_input = []
        for state in batch_states:
            occ_grid_2D, (ax, ay, avx, avy) = state
            flat_grid = occ_grid_2D.reshape(-1)  # shape => (grid_rows*grid_cols,)
            agent_arr = np.array([ax, ay, avx, avy], dtype=np.float32)
            combined = np.concatenate([flat_grid, agent_arr], axis=0)  # shape => (404,)
            flattened_input.append(combined)

        # FIX: convert once to a single NumPy array, then to Tensor
        flattened_input_np = np.array(flattened_input, dtype=np.float32)  # shape => [B, 404]
        x_input = torch.from_numpy(flattened_input_np).to(device)

        x = F.elu(self.fc1(x_input))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        out = self.fc4(x)
        return out

#############################################
# 2) Memory Class
#############################################
class Memory:
    def __init__(self, max_size=500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        # experience => ( (occ_grid, (ax,ay,vx,vy)), action, reward, next_state, done )
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

#############################################
# Soft, Hard Update, Checkpoint
#############################################
def soft_update(target_net, main_net, tau):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(
            tau * main_param.data + (1.0 - tau) * target_param.data
        )

def hard_update(target_net, main_net):
    target_net.load_state_dict(main_net.state_dict())

def load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer):
    checkpoint_files = [f for f in os.listdir(model_saved_path) if f.endswith('.pth')]
    if len(checkpoint_files) > 0:
        checkpoint_files.sort()
        latest_ckpt = os.path.join(model_saved_path, checkpoint_files[-1])
        checkpoint = torch.load(latest_ckpt, map_location=device)
        mainQN.load_state_dict(checkpoint['mainQN_state_dict'])
        targetQN.load_state_dict(checkpoint['targetQN_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        print("Successfully loaded:", latest_ckpt)

        # Remove old checkpoint files
        for filename in os.listdir(model_saved_path):
            filepath = os.path.join(model_saved_path, filename)
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        print("Removing checkpoint and files")
        print("Done")
        return start_episode
    else:
        print("Could not find old network weights. Run with initialization")
        return 1

#############################################
# Main Training Loop
#############################################
if __name__ == '__main__':
    # 1) Create the environment that returns (occ_grid_2D, (ax, ay, vx, vy))
    env = Cell_Space(
        xmin=0, xmax=10,
        ymin=0, ymax=10,
        zmin=0, zmax=2,
        dt=0.1,
        Number=Number_Agent,
        numExits=1,  # or however many you want
        numObs=0
    )

    # 2) Build DQN for a 20×20 grid => input_dim=400 +4=404
    mainQN = DQN(grid_rows=20, grid_cols=20, action_size=8).to(device)
    targetQN = DQN(grid_rows=20, grid_cols=20, action_size=8).to(device)
    hard_update(targetQN, mainQN)

    optimizer = torch.optim.Adam(mainQN.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    memory = Memory(max_size=memory_size)

    # Ensure dirs exist
    if not os.path.isdir(model_saved_path):
        os.mkdir(model_saved_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    start_episode = load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer)

    # Lists to store episode stats
    episode_rewards = []
    episode_losses = []

    step = 0
    for ep in range(start_episode, train_episodes + 1):
        state = env.reset()  # => (occ_grid, (ax, ay, vx, vy))
        done = False
        total_reward = 0
        t = 0

        # For computing avg loss in this episode
        episode_loss_accum = 0.0
        episode_loss_count = 0

        # Possibly save environment states
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, f'case_{ep}')
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(os.path.join(pathdir, f's.{t}'))

        while t < max_steps:
            # Epsilon Decay
            epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * ep / train_episodes)

            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.choose_random_action()
            else:
                with torch.no_grad():
                    Qs = mainQN([state]).cpu().numpy()[0]  
                action_candidates = np.where(Qs == np.max(Qs))[0]
                action = np.random.choice(action_candidates)

            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            t += 1

            # Store in replay
            memory.add((state, action, reward, next_state, done))

            if done:
                if ep % Cfg_save_freq == 0:
                    env.save_output(os.path.join(pathdir, f's.{t}'))
                break
            else:
                state = next_state
                if ep % Cfg_save_freq == 0 and t % cfg_save_step == 0:
                    env.save_output(os.path.join(pathdir, f's.{t}'))

            # Replay and train
            if len(memory.buffer) >= batch_size and t % train_step == 0:
                batch = memory.sample(batch_size)
                states_batch  = [b[0] for b in batch]
                actions_batch = [b[1] for b in batch]
                rewards_batch = [b[2] for b in batch]
                next_states_batch = [b[3] for b in batch]
                dones_batch   = [b[4] for b in batch]

                actions_t = torch.LongTensor(actions_batch).to(device)
                rewards_t = torch.FloatTensor(rewards_batch).to(device)
                dones_t   = torch.BoolTensor(dones_batch).to(device)

                with torch.no_grad():
                    targetQ_values = targetQN(next_states_batch)
                    max_targetQ, _ = torch.max(targetQ_values, dim=1)
                    max_targetQ[dones_t] = 0.0
                    target_values = rewards_t + gamma * max_targetQ

                currentQ_values = mainQN(states_batch)
                currentQ = currentQ_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                loss = loss_fn(currentQ, target_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss for this episode
                episode_loss_accum += loss.item()
                episode_loss_count += 1

        # End of episode => record stats
        episode_rewards.append(total_reward)
        if episode_loss_count > 0:
            avg_loss = episode_loss_accum / episode_loss_count
        else:
            avg_loss = 0.0
        episode_losses.append(avg_loss)

        if len(memory.buffer) >= batch_size:
            print(f"Episode: {ep}, Steps: {t}, Epsilon: {epsilon:.3f}, "
                  f"Reward: {total_reward:.2f}, Loss: {avg_loss:.7f}")

        # Save model periodically
        if ep % save_step == 0:
            save_path = os.path.join(model_saved_path, f"Evacuation_Continuum_model_ep{ep}.pth")
            torch.save({
                'episode': ep,
                'mainQN_state_dict': mainQN.state_dict(),
                'targetQN_state_dict': targetQN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            print(f"Model saved at episode {ep}")

        # Soft-update
        if ep % update_target_every == 0:
            soft_update(targetQN, mainQN, tau)

    # Final save
    final_save_path = os.path.join(model_saved_path, f"Evacuation_Continuum_model_ep{train_episodes}.pth")
    torch.save({
        'episode': train_episodes,
        'mainQN_state_dict': mainQN.state_dict(),
        'targetQN_state_dict': targetQN.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, final_save_path)
    print(f"Final model saved at episode {train_episodes}")

    # ------------------------------------------------------------------
    # Plot and Save Figures: Rewards, Loss
    # ------------------------------------------------------------------
    rewards_plot_path = os.path.join(output_dir, "rewards.png")
    loss_plot_path    = os.path.join(output_dir, "losses.png")

    # 1) Reward plot
    plt.figure()
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(rewards_plot_path, dpi=150)
    print(f"Saved reward plot to {rewards_plot_path}")

    # 2) Loss plot
    plt.figure()
    plt.plot(episode_losses, label='Average Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Average Loss per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path, dpi=150)
    print(f"Saved loss plot to {loss_plot_path}")

    plt.show()
