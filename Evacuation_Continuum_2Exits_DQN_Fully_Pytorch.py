import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from Continuum_Cellspace import *  # Assumed to be unchanged and compatible

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

Number_Agent = 1
Exit.append(np.array([0.5, 1.0, 0.5]))  # Add Up exit
Exit.append(np.array([0.5, 0, 0.5]))    # Add Down Exit

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_2Exit_DQN_Fully_Pytorch')
model_saved_path = os.path.join(model_saved_path, 'Continuum_2Exit_DQN_Fully_Pytorch')

name_mainQN = 'main_qn_2exit'
name_targetQN = 'target_qn_2exit'

# Hyperparameters and configuration
train_episodes = 10000      # max number of episodes
max_steps = 1500           # max steps in an episode
gamma = 0.999               # future reward discount

explore_start = 1.0         # initial exploration probability
explore_stop = 0.1          # minimum exploration probability
decay_percentage = 0.5
decay_rate = 4 / decay_percentage  # exploration decay rate

learning_rate = 1e-4        # Q-network learning rate
memory_size = 1000          # replay memory size
batch_size = 50             # mini-batch size
pretrain_length = batch_size

update_target_every = 1     # update target network frequency (in episodes)
tau = 0.1                   # soft update factor
save_step = 1000            # steps to save model
train_step = 1              # training every this many steps
Cfg_save_freq = 100         # frequency to save cfg (every #episodes)
#cfg_save_step = 1000         # steps to save env state within an episode

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
        # Assuming there's a single model checkpoint or picking the latest
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

if __name__ == '__main__':
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()

    memory = Memory(max_size=memory_size)

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

    start_episode = load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer)

    step = 0
    for ep in range(start_episode, train_episodes+1):
        total_reward = 0
        t = 0

        # For saving configuration snapshots
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, 'case_' + str(ep))
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(pathdir + '/s.' + str(t))

        state = env.reset()
        done = False
        while t < max_steps:
            epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * ep / train_episodes)

            feed_state = np.array(state)
            feed_state[:2] = env.Normalization_XY(feed_state[:2])
            feed_state_tensor = torch.FloatTensor(feed_state).unsqueeze(0).to(device)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.choose_random_action()
            else:
                with torch.no_grad():
                    Qs = mainQN(feed_state_tensor).cpu().numpy()[0]
                action_list = np.where(Qs == np.max(Qs))[0]
                action = np.random.choice(action_list)

            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            t += 1

            feed_next_state = np.array(next_state)
            feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])

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
                states = torch.FloatTensor([each[0] for each in batch]).to(device)
                actions = torch.LongTensor([each[1] for each in batch]).to(device)
                rewards = torch.FloatTensor([each[2] for each in batch]).to(device)
                next_states = torch.FloatTensor([each[3] for each in batch]).to(device)
                finish = torch.BoolTensor([each[4] for each in batch]).to(device)

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

        if len(memory.buffer) == memory_size:
            print(f"Episode: {ep}, Loss: {loss.item()}, steps per episode: {t}")

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

    # Final save
    save_path = os.path.join(model_saved_path, f"Evacuation_Continuum_model_ep{train_episodes}.pth")
    torch.save({
        'episode': train_episodes,
        'mainQN_state_dict': mainQN.state_dict(),
        'targetQN_state_dict': targetQN.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
