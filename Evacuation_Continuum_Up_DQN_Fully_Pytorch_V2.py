# Training agents Evacuation at 4 exits using Double DQN with Prioritized Experience Replay in PyTorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from collections import deque
from Continuum_Cellspace import *
from Grid_Space import *
import matplotlib.pyplot as plt

door_size = 1.0
Number_Agent = 1
Exit.append(np.array([0.5, 1.0, 0.5]))   # Add Up exit
#Exit.append(np.array([0.5, 0, 0.5]))     # Add Down Exit
#Exit.append(np.array([0, 0.5, 0.5]))     # Add Left exit
#Exit.append(np.array([1.0, 0.5, 0.5]))   # Add Right Exit

output_dir = './output'
figureSaveDir = './PrioitizedExpReplay6'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_Up_DoubleDQN_PER_PyTorch_V6')
model_saved_path = os.path.join(model_saved_path, 'Continuum_Up_DoubleDQN_PER_PyTorch_V6')
name_mainQN = 'main_qn_up_Pytorch'
name_targetQN = 'target_qn_up_Pytorch'

class DQN(nn.Module):
    def __init__(self, name, learning_rate=0.0001, gamma=0.99,
                 action_size=8, batch_size=20):
        super(DQN, self).__init__()
        self.name = name
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Network architecture
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_size)
        self.elu = nn.ELU()

        # Initialize weights using He normal initialization
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

    def get_parameters(self):
        return self.parameters()

# Memory replay with Prioritized Experience Replay
class Memory():
    def __init__(self, max_size=500, alpha=0.6):
        self.capacity = max_size
        self.buffer = []
        self.pos = 0
        self.alpha = alpha  # Determines how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def add(self, experience, td_error=None):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        if td_error is None:
            self.priorities[self.pos] = max_prio
        else:
            self.priorities[self.pos] = abs(td_error) + 1e-6  # Avoid zero priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Calculate probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) + 1e-6  # Avoid zero priority

if __name__ == '__main__':

    train_episodes = 10000
    loss_arr = np.zeros(train_episodes)  # Array to keep track of loss
    epsilon_arr = np.zeros(train_episodes) # Array to plot epsilon
    rewards_per_episode = np.zeros(train_episodes)
    max_steps = 1500
    gamma = 0.999

    explore_start = 1.0
    explore_stop = 0.1
    decay_percentage = 0.5
    decay_rate = -np.log(explore_stop) / (decay_percentage * train_episodes)

    learning_rate = 1e-4

    memory_size = 1000
    batch_size = 64
    pretrain_length = batch_size

    update_target_every = 100
    tau = 0.005  # Tau should be very small
    save_step = 500
    train_step = 1

    Cfg_save_freq = 500
    #cfg_save_step = 1

    beta_start = 0.4
    alpha = 0.6  # Degree of prioritization (0 - no prioritization, 1 - full prioritization)

    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()

    memory = Memory(max_size=memory_size, alpha=alpha)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the main and target networks
    mainQN = DQN(name=name_mainQN, learning_rate=learning_rate, gamma=gamma,
                 action_size=8, batch_size=batch_size).to(device)
    targetQN = DQN(name=name_targetQN, learning_rate=learning_rate, gamma=gamma,
                   action_size=8, batch_size=batch_size).to(device)

    # Copy the weights from mainQN to targetQN
    targetQN.load_state_dict(mainQN.state_dict())
    targetQN.eval()

    # Optimizer
    optimizer = optim.Adam(mainQN.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.SmoothL1Loss(reduction='none')  # Use 'none' to compute loss per sample

    step = 0

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Check for saved model to continue or start from initialization
    if not os.path.isdir(model_saved_path):
        os.mkdir(model_saved_path)

    checkpoint_files = [f for f in os.listdir(model_saved_path) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(model_saved_path, latest_checkpoint)
        mainQN.load_state_dict(torch.load(checkpoint_path))
        targetQN.load_state_dict(mainQN.state_dict())
        print("Successfully loaded:", checkpoint_path)

        # Remove old checkpoint files
        print("Removing checkpoint files")
        for filename in os.listdir(model_saved_path):
            file_path = os.path.join(model_saved_path, filename)
            os.remove(file_path)
        print("Done")
    else:
        print("Could not find old network weights. Run with the initialization")

    # Training Epoch Loop
    for ep in range(1, train_episodes + 1):
        # Shrink door to 1m after 1000 episodes
        if train_episodes >= 1000:
            door_size = 1.0

        total_reward = 0
        t = 0

        state = env.reset()

        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, 'case_' + str(ep))
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(pathdir + '/s.' + str(t))

        # Steps within 1 epoch
        while t < max_steps:
            # Calculate epsilon
            epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * ep)

            feed_state = np.array(state)
            feed_state[:2] = env.Normalization_XY(feed_state[:2])

            if np.random.rand() < epsilon:
                # Explore: select a random action
                action = env.choose_random_action()
            else:
                # Exploit: select the action with max Q-value
                mainQN.eval()
                with torch.no_grad():
                    # Send a tensor of the current environment state to the DQN
                    state_tensor = torch.FloatTensor(feed_state).unsqueeze(0).to(device)
                    Qs = mainQN(state_tensor).cpu().numpy()[0]

                # Make a list of all the actions with the max Q values. Select any of those actions
                action_list = [idx for idx, val in enumerate(Qs) if val == np.max(Qs)]
                action = np.random.choice(action_list)

            next_state, reward, done = env.step(action)

            total_reward += reward
            t += 1
            step += 1

            feed_next_state = np.array(next_state)
            feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])

            # When adding to memory, we don't have a TD error yet, so we set it to None
            memory.add((feed_state, action, reward, feed_next_state, done), td_error=None)

            if done:
                if ep % Cfg_save_freq == 0:
                    env.save_output(pathdir + '/s.' + str(t))
                #state = env.reset()
                break
            else:
                state = next_state

                if ep % Cfg_save_freq == 0:
                    if t % cfg_save_step == 0:
                        env.save_output(pathdir + '/s.' + str(t))

            if len(memory.buffer) >= batch_size and t % train_step == 0:
                beta = min(1.0, beta_start + ep * (1.0 - beta_start) / train_episodes)
                batch, indices, weights = memory.sample(batch_size, beta=beta)

                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                finish = np.array([each[4] for each in batch])

                # Convert to tensors
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                finish = torch.BoolTensor(finish).to(device)
                weights = torch.FloatTensor(weights).to(device)

                # Get Q-values from main network for current states
                mainQN.train()
                outputs = mainQN(states)
                actions = actions.unsqueeze(1)
                pred_Qs = outputs.gather(1, actions).squeeze(1)

                # Double DQN target calculation
                mainQN.eval()
                with torch.no_grad():
                    next_main_Qs = mainQN(next_states)
                    next_actions = next_main_Qs.argmax(dim=1).unsqueeze(1)

                targetQN.eval()
                with torch.no_grad():
                    next_target_Qs = targetQN(next_states)
                    max_next_Qs = next_target_Qs.gather(1, next_actions).squeeze(1)
                    max_next_Qs[finish] = 0.0  # Zero out terminal states

                targets = rewards + gamma * max_next_Qs

                # Compute loss with importance sampling weights
                loss = (criterion(pred_Qs, targets) * weights).mean()

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mainQN.parameters(), 10)
                optimizer.step()

                # Update priorities
                td_errors = (pred_Qs - targets).detach().cpu().numpy()
                memory.update_priorities(indices, td_errors)
        
                # Soft update the target network
                target_QN_state_dict = targetQN.state_dict()
                mainQN_state_dict = mainQN.state_dict()
                for key in mainQN_state_dict:
                    target_QN_state_dict[key] = mainQN_state_dict[key] * tau + target_QN_state_dict[key] * (1 - tau)
                targetQN.load_state_dict(target_QN_state_dict)

        #Hard Update Mechanism
        #if (ep % update_target_every == 0):
        #    targetQN.load_state_dict(mainQN.state_dict())

        if len(memory.buffer) >= batch_size:
            print(f"Episode: {ep}, Loss: {loss.item():.6f}, Steps per Episode: {t}, Reward per Episode: {total_reward:.6f} Epsilon: {epsilon:.4f}")
            loss_arr[ep - 1] = loss.item()
            epsilon_arr[ep-1] = epsilon
            rewards_per_episode[ep-1] = total_reward

        if ep % save_step == 0:
            if not os.path.isdir(model_saved_path):
                os.mkdir(model_saved_path)
            torch.save(mainQN.state_dict(), os.path.join(model_saved_path, f"Evacuation_Continuum_model_{ep}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(model_saved_path, f"Evacuation_Continuum_Optimizer_{ep}.pth"))

    # Save the final model
    torch.save(mainQN.state_dict(), os.path.join(model_saved_path, f"Evacuation_Continuum_model_{train_episodes}.pth"))

    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, train_episodes + 1), loss_arr)
    plt.title("Loss Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    #plt.show()
    loss_file_name = "LossPlot.png"
    if not (os.path.isdir(figureSaveDir)):
        os.mkdir(figureSaveDir)
    loss_file_path = os.path.join(figureSaveDir, loss_file_name)
    fig.savefig(loss_file_path, dpi=600)
    plt.close(fig)


    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, train_episodes + 1), epsilon_arr)
    plt.title("Epsilon Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    epsilon_file_name = "EpsilonPlot.png"
    
    epsilon_file_path = os.path.join(figureSaveDir, epsilon_file_name)
    fig.savefig(epsilon_file_path, dpi=600)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, train_episodes + 1), rewards_per_episode)
    plt.title("Rewards Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Rewardd")
    rewards_file_name = "RewardsPlot.png"
    
    rewards_file_path = os.path.join(figureSaveDir, rewards_file_name)
    fig.savefig(rewards_file_path, dpi=600)
    plt.close(fig)
