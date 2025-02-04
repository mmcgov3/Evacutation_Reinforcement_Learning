import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt  # For plotting
from Continuum_Cellspace import *  

# Set random seeds for reproducibility
np.random.seed(43)
torch.manual_seed(43)

Number_Agent = 1

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_1Exit_Obs_DoubleDQN_CornerSampling_Fully_Pytorch')
model_saved_path = os.path.join(model_saved_path, 'Continuum_1Exit_Obs_DQN_CornerSampling_Fully_Pytorch')

name_mainQN = 'main_qn_1exit_obs'
name_targetQN = 'target_qn_1exit_obs'

# Clear any previous data in the global obstacle lists
Exit.clear()
Ob.clear()
Ob_size.clear()

# Add the exit
Exit.append(np.array([0.5, 1.0, 0.5]))  # The top wall exit

# Add obstacle 1 just in front of exit
Ob1 = []
Ob1.append(np.array([0.5, 0.85, 0.5]))
Ob.append(Ob1)
Ob_size.append(3.0)

# Add obstacle 2 in an arbitrary location
Ob2 = []
Ob2.append(np.array([0.8, 0.35, 0.5]))
Ob.append(Ob2)
Ob_size.append(5.0)

# Obstacle 3: U-shaped (concave) obstacle with the opening facing away from the exit.
obs3 = [np.array([0.35, 0.6, 0.5]),
        np.array([0.35, 0.5, 0.5]),
        np.array([0.45, 0.45, 0.5]),
        np.array([0.55, 0.45, 0.5]),
        np.array([0.65, 0.5, 0.5]),
        np.array([0.65, 0.6, 0.5])]
Ob.append(obs3)
Ob_size.append(4.0)

# Hyperparameters and configuration
GRID_SIZE = 10
train_episodes = 10000      # max number of episodes
max_steps = 1500            # max steps in an episode
gamma = 0.99                # future reward discount

explore_start = 1.0         # initial exploration probability
explore_stop = 0.1          # minimum exploration probability
# Linear decay: epsilon will decrease linearly from explore_start to explore_stop over train_episodes
def get_epsilon(ep):
    return max(explore_stop, explore_start - (explore_start - explore_stop) * (ep / train_episodes))

learning_rate = 1e-4        # Q-network learning rate
memory_size = 10000         # replay memory capacity
batch_size = 64             # mini-batch size

update_target_every = 1     # update target network frequency (in episodes)
tau = 0.1                   # soft update factor
save_step = 3000            # steps to save model
train_step = 1              # training every this many steps
Cfg_save_freq = 3000        # frequency to save cfg (every #episodes)
cfg_save_step = 1           # steps to save env state within an episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################
# Define the DQN network architecture
#####################################
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

#####################################
# Prioritized Experience Replay Buffer
#####################################
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, experience):
        """Add a new experience with maximum priority so far (or 1 if buffer is empty)."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize for stability
        return samples, indices, weights

    def update_priorities(self, indices, errors, epsilon=1e-6):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + epsilon

#####################################
# Learning rate scheduler: We use StepLR.
#####################################
# Create environment for network training
env = Cell_Space(0, GRID_SIZE, 0, GRID_SIZE, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
state = env.reset()

# Create prioritized replay buffer
memory = PrioritizedReplayBuffer(capacity=memory_size, alpha=0.6)

# Q-networks
mainQN = DQN().to(device)
targetQN = DQN().to(device)

# Hard-update target network initially
def hard_update(target_net, main_net):
    target_net.load_state_dict(main_net.state_dict())

hard_update(targetQN, mainQN)

optimizer = torch.optim.Adam(mainQN.parameters(), lr=learning_rate)
# Define a step LR scheduler: every 1000 episodes, reduce LR by multiplying by 0.95.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
loss_fn = nn.SmoothL1Loss(reduction='none')  # we'll apply importance weights manually

#####################################
# Utility functions for network updates
#####################################
def soft_update(target_net, main_net, tau):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

def load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer):
    # Mimic original logic: If there is a checkpoint, load it
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
        # Remove checkpoint files after loading (mimic original logic)
        for filename in os.listdir(model_saved_path):
            filepath = os.path.join(model_saved_path, filename)
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        print("Removing checkpoint files. Done")
        return start_episode
    else:
        print("Could not find old network weights. Run with initialization")
        return 1

def rolling_window_average(values, window_size=250):
    result = []
    cumsum = np.cumsum(np.insert(values, 0, 0))
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_sum = cumsum[i + 1] - cumsum[start_index]
        window_len = i - start_index + 1
        result.append(window_sum / window_len)
    return result

#####################################
# Training Loop
#####################################
start_episode = load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer)
step = 0

# For storing stats per episode
episode_losses = []   # average loss per episode
episode_rewards = []  # total reward per episode

for ep in range(start_episode, train_episodes + 1):
    total_reward = 0
    t = 0
    episode_loss_sum = 0.0
    episode_loss_count = 0

    # For saving configuration snapshots
    if ep % Cfg_save_freq == 0:
        pathdir = os.path.join(output_dir, 'case_' + str(ep))
        if not os.path.isdir(pathdir):
            os.mkdir(pathdir)
        env.save_output(pathdir + '/s.' + str(t))

    state = env.reset()

    # Force agent into one of the four corners with 15% probability
    if np.random.rand() < 0.15:
        # Pick x-range
        if np.random.rand() < 0.5:
            x_val = np.random.uniform(0.0, 0.25)   # left side
        else:
            x_val = np.random.uniform(0.75, 1.0)   # right side
        # Pick y-range
        if np.random.rand() < 0.5:
            y_val = np.random.uniform(0.0, 0.25)   # bottom side
        else:
            y_val = np.random.uniform(0.75, 1.0)   # top side
        # Convert normalized coordinates to environment domain [0..GRID_SIZE]
        agent_x = x_val * GRID_SIZE
        agent_y = y_val * GRID_SIZE
        env.agent.position[0] = agent_x
        env.agent.position[1] = agent_y

    state = (env.agent.position[0], env.agent.position[1],
             env.agent.velocity[0], env.agent.velocity[1])
    
    done = False

    # Anneal beta for prioritized replay (linearly from 0.4 to 1.0 over training)
    beta = min(1.0, 0.4 + (ep / train_episodes) * (1.0 - 0.4))
    epsilon = get_epsilon(ep)

    while t < max_steps:
        # Build feed_state for network (normalize first two coordinates)
        feed_state = np.array(state, dtype=np.float32)
        feed_state[:2] = env.Normalization_XY(feed_state[:2])
        feed_state_tensor = torch.from_numpy(feed_state).unsqueeze(0).to(device)

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.choose_random_action()
        else:
            with torch.no_grad():
                Qs = mainQN(feed_state_tensor).cpu().numpy()[0]
            action_candidates = np.where(Qs == np.max(Qs))[0]
            action = np.random.choice(action_candidates)

        # Step in environment
        next_state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        t += 1

        feed_next_state = np.array(next_state, dtype=np.float32)
        feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])
        # Store transition in prioritized replay buffer
        memory.add((feed_state, action, reward, feed_next_state, done))

        if done:
            if ep % Cfg_save_freq == 0:
                env.save_output(pathdir + '/s.' + str(t))
            break
        else:
            state = next_state
            if ep % Cfg_save_freq == 0 and t % cfg_save_step == 0:
                env.save_output(pathdir + '/s.' + str(t))

        # Training step: start training once the buffer is full (or after a minimal number of samples)
        if len(memory.buffer) >= batch_size and t % train_step == 0:
            batch, indices, weights = memory.sample(batch_size, beta)
            # Unpack batch elements
            b_states   = np.array([ex[0] for ex in batch], dtype=np.float32)
            b_actions  = np.array([ex[1] for ex in batch], dtype=np.int64)
            b_rewards  = np.array([ex[2] for ex in batch], dtype=np.float32)
            b_next     = np.array([ex[3] for ex in batch], dtype=np.float32)
            b_done     = np.array([ex[4] for ex in batch], dtype=np.bool_)

            states_tensor      = torch.from_numpy(b_states).to(device)
            actions_tensor     = torch.from_numpy(b_actions).to(device)
            rewards_tensor     = torch.from_numpy(b_rewards).to(device)
            next_states_tensor = torch.from_numpy(b_next).to(device)
            dones_tensor       = torch.from_numpy(b_done.astype(np.uint8)).to(device)
            weights_tensor     = torch.tensor(weights, dtype=torch.float32).to(device)

            # Double DQN target calculation:
            # Main network selects the best action for the next state
            with torch.no_grad():
                next_q_main = mainQN(next_states_tensor)
                next_actions = torch.argmax(next_q_main, dim=1, keepdim=True)
                next_q_target = targetQN(next_states_tensor).gather(1, next_actions).squeeze(1)
                next_q_target[dones_tensor] = 0.0
                target_values = rewards_tensor + gamma * next_q_target

            current_q = mainQN(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            # Compute per-sample loss and weight them with IS weights
            loss_all = loss_fn(current_q, target_values)
            weighted_loss = (weights_tensor * loss_all).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            # Update priorities using the TD errors (absolute error)
            td_errors = (current_q - target_values).detach().cpu().numpy()
            memory.update_priorities(indices, np.abs(td_errors))

            episode_loss_sum += weighted_loss.item()
            episode_loss_count += 1

    if episode_loss_count > 0:
        avg_loss = episode_loss_sum / episode_loss_count
    else:
        avg_loss = 0.0

    episode_losses.append(avg_loss)
    episode_rewards.append(total_reward)

    if len(memory.buffer) >= batch_size:
        print(f"Episode: {ep}, Steps: {t}, Epsilon: {epsilon:.3f}, Beta: {beta:.3f}, "
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

    # Update target network with soft update
    if ep % update_target_every == 0:
        soft_update(targetQN, mainQN, tau)

    # Step the learning rate scheduler after each episode
    scheduler.step()

# Final save of the model
save_path = os.path.join(model_saved_path, f"Evacuation_Continuum_model_ep{train_episodes}.pth")
torch.save({
    'episode': train_episodes,
    'mainQN_state_dict': mainQN.state_dict(),
    'targetQN_state_dict': targetQN.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, save_path)

# ========================
# Plot & save the results
# ========================
ma_losses = rolling_window_average(episode_losses, window_size=250)
ma_rewards = rolling_window_average(episode_rewards, window_size=250)

# Plot 1: Loss per episode
plt.figure()
plt.plot(episode_losses, label='Loss per Episode')
plt.plot(ma_losses, label='Rolling Avg (250) Loss', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('DQN Loss per Episode')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.close()

# Plot 2: Reward per episode
plt.figure()
plt.plot(episode_rewards, label='Reward per Episode')
plt.plot(ma_rewards, label='Rolling Avg (250) Reward', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Reward per Episode')
plt.legend()
plt.savefig(os.path.join(output_dir, 'reward_plot.png'))
plt.close()

print("Training complete. Plots saved in:", output_dir)
