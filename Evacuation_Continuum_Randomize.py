import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# Import your environment after you have updated it
# so that reset() and step() both return a 6D state
from Continuum_Cellspace2 import Cell_Space

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

########################
# Customizable parameters
########################
Number_Agent = 1
numExits = 1     # example: 2 exits
numObs = 0       # example: 0 obstacles

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_1ExitRandom_NearestDistances_DQN_Fully_Pytorch')
model_saved_path = os.path.join(model_saved_path, 'Continuum_1ExitRandom_NearestDistances_DQN_Fully_Pytorch')

name_mainQN = 'main_qn_1ExitRandom_NearestDistances'
name_targetQN = 'target_qn_1ExitRandom_NearestDistances'

# Hyperparameters and configuration
train_episodes = 20000      # max number of episodes
max_steps = 2000            # max steps in an episode
gamma = 0.999               # future reward discount

explore_start = 1.0         # initial exploration probability
explore_stop = 0.1          # minimum exploration probability
decay_percentage = 0.5
decay_rate = 4 / decay_percentage  # exploration decay rate

learning_rate = 1e-4        # Q-network learning rate
memory_size = 10000         # replay memory size
batch_size = 50             # mini-batch size
pretrain_length = batch_size

update_target_every = 1     # update target network frequency (in episodes)
tau = 0.1                   # soft update factor
save_step = 1000            # steps to save model
train_step = 1              # training every this many steps
Cfg_save_freq = 1000        # frequency to save cfg (every #episodes)
cfg_save_step = 1           # time steps within an episode at which to save environment state (optional)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# DQN Class
#############################################
class DQN(nn.Module):
    def __init__(self, state_size, action_size=8):
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

#############################################
# Memory Class
#############################################
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

#############################################
# Soft and Hard Update Functions
#############################################
def soft_update(target_net, main_net, tau):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(
            tau * main_param.data + (1.0 - tau) * target_param.data
        )

def hard_update(target_net, main_net):
    target_net.load_state_dict(main_net.state_dict())

#############################################
# Load Checkpoint Function
#############################################
def load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer):
    # Check for any .pth file
    checkpoint_files = [f for f in os.listdir(model_saved_path) if f.endswith('.pth')]
    if len(checkpoint_files) > 0:
        # Sort them, pick the last one
        checkpoint_files.sort()
        latest_ckpt = os.path.join(model_saved_path, checkpoint_files[-1])
        checkpoint = torch.load(latest_ckpt, map_location=device)
        mainQN.load_state_dict(checkpoint['mainQN_state_dict'])
        targetQN.load_state_dict(checkpoint['targetQN_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        print("Successfully loaded:", latest_ckpt)

        # Remove all checkpoint files to mimic original logic
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
    # 1) Instantiate environment with desired number of exits/obstacles
    env = Cell_Space(
        xmin=0, xmax=10,
        ymin=0, ymax=10,
        zmin=0, zmax=2,
        rcut=1.5,
        dt=0.1,
        Number=Number_Agent,
        numExits=numExits,
        numObs=numObs
    )

    # 2) Because the environment now returns a 6D state: 
    #    (x, y, vx, vy, distToNearestExit, distToNearestObstacle)
    #    we fix state_size = 6
    state_size = 6
    action_size = 8  # environment has 8 discrete actions

    # 3) Create the main Q-network and target Q-network
    mainQN = DQN(state_size=state_size, action_size=action_size).to(device)
    targetQN = DQN(state_size=state_size, action_size=action_size).to(device)
    hard_update(targetQN, mainQN)

    # 4) Create the optimizer, loss function, and memory buffer
    optimizer = torch.optim.Adam(mainQN.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    memory = Memory(max_size=memory_size)

    # Ensure directories exist
    if not os.path.isdir(model_saved_path):
        os.mkdir(model_saved_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # 5) Load checkpoint if available
    start_episode = load_checkpoint_if_exists(model_saved_path, mainQN, targetQN, optimizer)

    step = 0
    # 6) Training episodes
    for ep in range(start_episode, train_episodes + 1):
        # Reset the environment at the start of each episode
        state = env.reset()
        done = False
        total_reward = 0
        t = 0

        # Possibly create a directory for saving environment states every Cfg_save_freq episodes
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, f'case_{ep}')
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(os.path.join(pathdir, f's.{t}'))

        # Episode main loop
        while t < max_steps:
            # Epsilon-decay
            epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * ep / train_episodes)

            # Convert state to torch, shape => (1, state_size)
            feed_state = np.array(state, dtype=np.float32)
            feed_state_tensor = torch.FloatTensor(feed_state).unsqueeze(0).to(device)

            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.choose_random_action()
            else:
                with torch.no_grad():
                    Qs = mainQN(feed_state_tensor).cpu().numpy()[0]
                # pick an action that yields max Q
                action_candidates = np.where(Qs == np.max(Qs))[0]
                action = np.random.choice(action_candidates)

            # Step in the environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            t += 1

            # Add (s, a, r, s', done) to memory
            memory.add((feed_state, action, reward, np.array(next_state, dtype=np.float32), done))

            # If done, possibly save config file
            if done:
                if ep % Cfg_save_freq == 0:
                    env.save_output(os.path.join(pathdir, f's.{t}'))
                break
            else:
                state = next_state
                # Optionally save env state at intervals
                if ep % Cfg_save_freq == 0 and t % cfg_save_step == 0:
                    env.save_output(os.path.join(pathdir, f's.{t}'))

            # Replay and train
            if len(memory.buffer) >= batch_size and t % train_step == 0:
                batch = memory.sample(batch_size)
                states = torch.FloatTensor([b[0] for b in batch]).to(device)
                actions = torch.LongTensor([b[1] for b in batch]).to(device)
                rewards = torch.FloatTensor([b[2] for b in batch]).to(device)
                next_states = torch.FloatTensor([b[3] for b in batch]).to(device)
                dones = torch.BoolTensor([b[4] for b in batch]).to(device)

                # Target Q
                with torch.no_grad():
                    targetQ_values = targetQN(next_states)
                    max_targetQ, _ = torch.max(targetQ_values, dim=1)
                    max_targetQ[dones] = 0.0
                    target_values = rewards + gamma * max_targetQ

                # Main Q
                currentQ = mainQN(states)
                currentQ = currentQ.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Loss
                loss = loss_fn(currentQ, target_values)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Print some info
        if len(memory.buffer) >= batch_size:
            print(f"Episode: {ep}, Steps: {t}, Epsilon: {epsilon:.3f}, Reward: {total_reward:.2f}, Loss: {loss.item():.7f}")

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

        # Update target net (soft update)
        if ep % update_target_every == 0:
            soft_update(targetQN, mainQN, tau)
            # Uncomment for debug
            # print(f"Target network updated at episode {ep}")

    # Final save
    final_save_path = os.path.join(model_saved_path, f"Evacuation_Continuum_model_ep{train_episodes}.pth")
    torch.save({
        'episode': train_episodes,
        'mainQN_state_dict': mainQN.state_dict(),
        'targetQN_state_dict': targetQN.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, final_save_path)
    print(f"Final model saved at episode {train_episodes}")
