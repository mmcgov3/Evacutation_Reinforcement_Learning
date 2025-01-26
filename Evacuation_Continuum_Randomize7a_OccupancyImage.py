import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Import your environment returning shape=(3, rows, cols) for the state
# ----------------------------------------------------------------------
from Continuum_Cellspace7_OccupancyImage import Cell_Space  # must produce (3, rows, cols)

# Set random seeds for reproducibility
np.random.seed(43)
torch.manual_seed(43)

########################
# Customizable parameters
########################
Number_Agent = 1

# Output directories
output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'OccupancyGrid_CNN_DQN_1Exit_0Ob')
model_saved_path = os.path.join(model_saved_path, 'OccupancyGrid_CNN_DQN_1Exit_0Ob')

# Hyperparameters
train_episodes = 20000
max_steps = 2000
gamma = 0.99

explore_start = 1.0
explore_stop = 0.1
decay_percentage = 0.7
decay_rate = 4 / decay_percentage

learning_rate = 5e54
memory_size = 10000
batch_size = 128
pretrain_length = batch_size

update_target_every = 1
tau = 0.1
save_step = 1000
train_step = 1
Cfg_save_freq = 1000
cfg_save_step = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################
# CNN-based DQN for a (3, rows, cols) environment
##################################################
class CNN_DQN(nn.Module):
    def __init__(self, in_channels=3, rows=20, cols=20, action_size=8):
        """
        We'll define a small CNN that takes input shape: (B, in_channels, rows, cols).
        Then flatten => FC => produce Q-values for each discrete action.
        """
        super(CNN_DQN, self).__init__()
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.action_size = action_size

        # Convolution layers
        # Adjust kernel_size, stride, out_channels as needed
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # We'll compute the shape after these 3 convs
        def conv_out_size(in_size, stride=2, kernel_size=3, padding=1):
            return (in_size + 2*padding - (kernel_size-1)-1)//stride +1

        c1_h = conv_out_size(rows)
        c1_w = conv_out_size(cols)
        c2_h = conv_out_size(c1_h)
        c2_w = conv_out_size(c1_w)
        c3_h = conv_out_size(c2_h)
        c3_w = conv_out_size(c2_w)
        flatten_dim = 64 * c3_h * c3_w

        # FC layers
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc2 = nn.Linear(128, action_size)

        # Weight init
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, batch_states):
        """
        batch_states => either a single state (3, rows, cols) or a list of them
        We'll convert to shape => (B, 3, rows, cols) => pass through CNN => FC => Q-values
        """
        if not isinstance(batch_states, list):
            batch_states = [batch_states]

        # Build one big array => shape (B, in_channels, rows, cols)
        state_list = []
        for st in batch_states:
            # st shape => (3, rows, cols)
            assert st.shape[0] == self.in_channels, f"Expected {self.in_channels} channels, got {st.shape[0]}"
            state_list.append(st)

        arr = np.array(state_list, dtype=np.float32)  # => shape [B, 3, rows, cols]
        x_input = torch.from_numpy(arr).to(device)

        # forward pass
        x = F.relu(self.conv1(x_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        out= self.fc2(x)  # => (B, action_size)
        return out

#############################################
# Memory Class
#############################################
class Memory:
    def __init__(self, max_size=500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        # experience => ( state_3D, action, reward, next_state_3D, done )
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
# Utility: moving average
#############################################
def moving_average(data, window_size=50):
    """
    Compute a simple moving average of 'data' with 'window_size'.
    Return an array of same length with smoothed values.
    """
    ma = []
    cum_sum = 0.0
    q = deque()
    for val in data:
        q.append(val)
        cum_sum += val
        if len(q) > window_size:
            cum_sum -= q.popleft()
        ma.append(cum_sum / len(q))
    return ma

#############################################
# Main Training Loop
#############################################
if __name__ == '__main__':
    # 1) Create environment that returns shape=(3, rows, cols)
    env = Cell_Space(
        xmin=0, xmax=10,
        ymin=0, ymax=10,
        zmin=0, zmax=2,
        dt=0.1,
        Number=Number_Agent,
        numExits=1,
        numObs=0
    )

    # Suppose shape => (3,20,20). We'll define the CNN DQN with these dims
    mainQN = CNN_DQN(in_channels=3, rows=20, cols=20, action_size=8).to(device)
    targetQN= CNN_DQN(in_channels=3, rows=20, cols=20, action_size=8).to(device)
    hard_update(targetQN, mainQN)

    optimizer = torch.optim.Adam(mainQN.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    from collections import deque
    memory = Memory(max_size=memory_size)

    # Ensure directories exist
    if not os.path.isdir(model_saved_path):
        os.mkdir(model_saved_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    start_episode = load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer)

    # Stats
    episode_rewards = []
    episode_losses = []

    step = 0
    for ep in range(start_episode, train_episodes + 1):
        state = env.reset()   # shape => (3, rows, cols)
        done = False
        total_reward = 0
        t = 0

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
                    Qs = mainQN([state]).cpu().numpy()[0]  # => shape (action_size,)
                action_candidates = np.where(Qs == np.max(Qs))[0]
                action = np.random.choice(action_candidates)

            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            t += 1

            # store
            memory.add((state, action, reward, next_state, done))

            if done:
                if ep % Cfg_save_freq == 0:
                    env.save_output(os.path.join(pathdir, f's.{t}'))
                break
            else:
                state = next_state
                if ep % Cfg_save_freq == 0 and t % cfg_save_step == 0:
                    env.save_output(os.path.join(pathdir, f's.{t}'))

            # Replay & train
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
                    targetQ_values = targetQN(next_states_batch)  # => shape (B, action_size)
                    max_targetQ, _ = torch.max(targetQ_values, dim=1)
                    max_targetQ[dones_t] = 0.0
                    target_values = rewards_t + gamma * max_targetQ

                currentQ_values = mainQN(states_batch)  # => (B, action_size)
                currentQ = currentQ_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                loss = loss_fn(currentQ, target_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss_accum += loss.item()
                episode_loss_count += 1

        # end of episode
        episode_rewards.append(total_reward)
        if episode_loss_count>0:
            avg_loss=episode_loss_accum / episode_loss_count
        else:
            avg_loss=0.0
        episode_losses.append(avg_loss)

        if len(memory.buffer) >= batch_size:
            print(f"Episode: {ep}, Steps: {t}, Epsilon: {epsilon:.3f}, "
                  f"Reward: {total_reward:.2f}, Loss: {avg_loss:.7f}")

        # Save model periodically
        if ep % save_step==0:
            save_path = os.path.join(model_saved_path, f"Evacuation_Continuum_model_ep{ep}.pth")
            torch.save({
                'episode': ep,
                'mainQN_state_dict': mainQN.state_dict(),
                'targetQN_state_dict': targetQN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            print(f"Model saved at episode {ep}")

        # Soft-update
        if ep % update_target_every==0:
            for _ in range(1):
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

    # ---------------------------------------
    # Plot: Rewards & Loss with moving avg
    # ---------------------------------------
    rewards_plot_path = os.path.join(output_dir, "rewards.png")
    loss_plot_path    = os.path.join(output_dir, "losses.png")

    # 1) Reward plot + moving average
    plt.figure()
    plt.plot(episode_rewards, label='Episode Reward')
    ma_rewards = moving_average(episode_rewards, window_size=50)
    plt.plot(ma_rewards, label='Moving Avg (50)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(rewards_plot_path, dpi=150)
    print(f"Saved reward plot to {rewards_plot_path}")

    # 2) Loss plot + moving average
    plt.figure()
    plt.plot(episode_losses, label='Average Loss')
    ma_loss = moving_average(episode_losses, window_size=50)
    plt.plot(ma_loss, label='Moving Avg (50)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Average Loss per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path, dpi=150)
    print(f"Saved loss plot to {loss_plot_path}")

    plt.show()
