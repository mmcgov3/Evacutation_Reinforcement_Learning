import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt  # For plotting
from Continuum_Cellspace import *  # Assumed to be unchanged and compatible

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

Number_Agent = 1
Exit.append(np.array([0.5, 1.0, 0.5]))  # Add exit on the top wall

Ob1 = []
Ob1.append(np.array([0.5, 0.7, 0.5]))
Ob.append(Ob1)
Ob_size.append(2.0)

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_1Exit_Ob_DQN_Fully_Pytorch')
model_saved_path = os.path.join(model_saved_path, 'Continuum_1Exit_Ob_DQN_Fully_Pytorch')

name_mainQN = 'main_qn_1exit_ob'
name_targetQN = 'target_qn_1exit_ob'

# Hyperparameters and configuration
train_episodes = 10000      # max number of episodes
max_steps = 1500            # max steps in an episode
gamma = 0.99                # future reward discount

explore_start = 1.0         # initial exploration probability
explore_stop = 0.1          # minimum exploration probability
decay_percentage = 0.5
decay_rate = 4 / decay_percentage  # exploration decay rate

learning_rate = 1e-4        # Q-network learning rate
memory_size = 10000         # replay memory size
batch_size = 64             # mini-batch size

update_target_every = 1     # update target network frequency (in episodes)
tau = 0.1                   # soft update factor
save_step = 3000            # steps to save model
train_step = 1              # training every this many steps
Cfg_save_freq = 3000        # frequency to save cfg (every #episodes)
cfg_save_step = 1           # steps to save env state within an episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Memory:
    def __init__(self, max_size=500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

def soft_update(target_net, main_net, tau):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

def hard_update(target_net, main_net):
    target_net.load_state_dict(main_net.state_dict())

def load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer):
    # Mimic original logic: If there is a checkpoint, load it
    checkpoint_files = [f for f in os.listdir(model_saved_path) if f.endswith('.pth')]
    if len(checkpoint_files) > 0:
        # For simplicity, pick the first or latest checkpoint
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
        print("Removing check point and files")
        print("Done")
        return start_episode
    else:
        print("Could not find old network weights. Run with initialization")
        return 1

def rolling_window_average(values, window_size=250):
    """
    Returns a list of the same length as 'values',
    where the ith element is the average of the last 'window_size' values up to i,
    or fewer if i < window_size.
    """
    result = []
    cumsum = np.cumsum(np.insert(values, 0, 0))
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_sum = cumsum[i + 1] - cumsum[start_index]
        window_len = i - start_index + 1
        result.append(window_sum / window_len)
    return result

if __name__ == '__main__':
    # Build environment
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()

    # Replay buffer
    memory = Memory(max_size=memory_size)

    # Q-networks
    mainQN = DQN().to(device)
    targetQN = DQN().to(device)
    hard_update(targetQN, mainQN)

    optimizer = torch.optim.Adam(mainQN.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    # Create directories if needed
    if not os.path.isdir(model_saved_path):
        os.mkdir(model_saved_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load checkpoint if exists
    start_episode = load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer)

    step = 0

    # To store stats per episode
    episode_losses = []   # will hold average loss for each episode
    episode_rewards = []  # will hold total reward for each episode

    for ep in range(start_episode, train_episodes + 1):
        total_reward = 0
        t = 0

        # Track losses within the episode
        episode_loss_sum = 0.0
        episode_loss_count = 0

        # For saving configuration snapshots
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, 'case_' + str(ep))
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(pathdir + '/s.' + str(t))

        state = env.reset()
        done = False

        while t < max_steps:
            # Epsilon decay
            epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * ep / train_episodes)

            # Build feed_state for network
            feed_state = np.array(state, dtype=np.float32)
            feed_state[:2] = env.Normalization_XY(feed_state[:2])
            feed_state_tensor = torch.from_numpy(feed_state).unsqueeze(0).to(device)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.choose_random_action()
            else:
                with torch.no_grad():
                    Qs = mainQN(feed_state_tensor).cpu().numpy()[0]
                action_list = np.where(Qs == np.max(Qs))[0]
                action = np.random.choice(action_list)

            # Step in environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            t += 1

            feed_next_state = np.array(next_state, dtype=np.float32)
            feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])

            # Store transition in replay buffer
            memory.add((feed_state, action, reward, feed_next_state, done))

            if done:
                if ep % Cfg_save_freq == 0:
                    env.save_output(pathdir + '/s.' + str(t))
                break
            else:
                state = next_state
                
                if ep % Cfg_save_freq == 0 and t % cfg_save_step == 0:
                    env.save_output(pathdir + '/s.' + str(t))

            # Training step
            if len(memory.buffer) == memory_size and t % train_step == 0:
                batch = memory.sample(batch_size)
                
                # Instead of building each tensor from a list-of-lists
                # we create a NumPy array first, then convert to a tensor.
                b_states   = np.array([ex[0] for ex in batch], dtype=np.float32)
                b_actions  = np.array([ex[1] for ex in batch], dtype=np.int64)
                b_rewards  = np.array([ex[2] for ex in batch], dtype=np.float32)
                b_next     = np.array([ex[3] for ex in batch], dtype=np.float32)
                b_done     = np.array([ex[4] for ex in batch], dtype=np.bool_)

                states      = torch.from_numpy(b_states).to(device)
                actions     = torch.from_numpy(b_actions).to(device)
                rewards     = torch.from_numpy(b_rewards).to(device)
                next_states = torch.from_numpy(b_next).to(device)
                finish      = torch.from_numpy(b_done).to(device)

                # Compute target Q-values
                with torch.no_grad():
                    targetQ = targetQN(next_states)
                    max_targetQ, _ = torch.max(targetQ, dim=1)
                    max_targetQ[finish] = 0.0  # terminal states have 0 future value
                    target_values = rewards + gamma * max_targetQ

                # Compute predicted Q-values
                currentQ = mainQN(states)
                currentQ = currentQ.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute loss
                loss = loss_fn(currentQ, target_values)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss_sum += loss.item()
                episode_loss_count += 1

        # If we had any training steps within this episode, compute the average loss
        if episode_loss_count > 0:
            avg_loss = episode_loss_sum / episode_loss_count
        else:
            avg_loss = 0.0

        episode_losses.append(avg_loss)
        episode_rewards.append(total_reward)

        if len(memory.buffer) == memory_size:
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

        # Update target network
        if ep % update_target_every == 0:
            soft_update(targetQN, mainQN, tau)

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

    # Rolling averages over the last 250 episodes
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
