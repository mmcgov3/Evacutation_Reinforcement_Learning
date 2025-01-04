# Training agents Evacuation at 4 exits using Deep Q-Network with PyTorch

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

figureSaveDir = './DecayLearningRateExitReward'
output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_Up_Decay_LR_ExitReward')
model_saved_path = os.path.join(model_saved_path, 'Continuum_Up_Decay_LR_ExitReward')
#output_dir = os.path.join(output_dir, 'Continuum_1Exit_DQN_Fully_PyTorch_Grid_Test')
#model_saved_path = os.path.join(model_saved_path, 'Continuum_1Exit_DQN_Fully_PyTorch_Grid_Test')
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

    # Initialize weights using He normal initialization (leaky-ReLU as nonlinearity default)
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
'''
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
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.output_layer(x)
        return x

    def get_parameters(self):
        return self.parameters()
'''


# Memory replay
class Memory():
    def __init__(self, max_size=500):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            #return list(self.buffer)
            return self.buffer
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]

if __name__ == '__main__':
    
    train_episodes = 15000
    # decaying learning rate
    initial_lr = 1e-4
    final_lr = 1e-5
    decay_step = (initial_lr - final_lr) / train_episodes

    loss_arr = np.zeros(train_episodes)  # Array to keep track of loss
    epsilon_arr = np.zeros(train_episodes) # Array to plot epsilon
    rewards_per_episode = np.zeros(train_episodes)
    max_steps = 1500
    gamma = 0.999

    explore_start = 1.0
    explore_stop = 0.1
    decay_percentage = 0.5
    decay_rate = -np.log(explore_stop) / (decay_percentage * train_episodes)

    memory_size = 1500
    batch_size = 100
    pretrain_length = batch_size

    update_target_every = 500
    tau = 0.005                                       # Tau should be very small
    save_step = 500
    train_step = 1

    Cfg_save_freq = 500
    #cfg_save_step = 1

    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    # After initializing env
    #exit_position = env.Exit[0][:2]  # Extract the x and y coordinates of the exit

    # Reset the source in the grid to be at the exit's position
    #env.Grid.reset_source([exit_position], [2.0])  # The source value can remain 2.0
    #grid = GridSpace_2D(0, 10, 0, 10, dt=delta_t)
    state = env.reset()

    memory = Memory(max_size=memory_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the main and target networks
    mainQN = DQN(name=name_mainQN, learning_rate=initial_lr, gamma=gamma,
                 action_size=8, batch_size=batch_size).to(device)
    targetQN = DQN(name=name_targetQN, learning_rate=initial_lr, gamma=gamma,
                   action_size=8, batch_size=batch_size).to(device)

    # Copy the weights from mainQN to targetQN
    targetQN.load_state_dict(mainQN.state_dict())
    targetQN.eval()

    # Optimizer
    optimizer = optim.Adam(mainQN.parameters(), lr=initial_lr)

    # Loss function
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

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
        total_reward = 0
        t = 0

        # Adjust learning rate
        current_lr = max(final_lr, initial_lr - decay_step * ep)  # Ensure it doesn't go below final_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        state = env.reset()
        
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, 'case_' + str(ep))
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(pathdir + '/s.' + str(t))

        # Steps within 1 epoch
        while t < max_steps:
            # Calculate epsilon
            #epsilon = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*ep/train_episodes)
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
                    # send a tensor of the current evironment state to the DQN
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

            memory.add((feed_state, action, reward, feed_next_state, done))

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

            #if len(memory.buffer) >= memory_size and t % train_step == 0:
            if len(memory.buffer) >= batch_size and t % train_step == 0:
                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
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

                # Get Q-values from target network
                targetQN.eval()
                with torch.no_grad():
                    target_Qs = targetQN(next_states)
                    target_Qs[finish] = 0.0  # Zero out terminal states

                # Get Q-values from main network
                mainQN.train()
                outputs = mainQN(states)

                # Gather the predicted Q-values for the actions taken
                actions = actions.unsqueeze(1)
                pred_Qs = outputs.gather(1, actions).squeeze(1)

                # Compute the max Q-values for next states
                with torch.no_grad():
                    next_Qs = targetQN(next_states)
                    max_next_Qs, _ = next_Qs.max(dim=1)
                    max_next_Qs[finish] = 0.0  # Zero out terminal states

                # Compute targets using the Bellman equation
                targets = rewards + gamma * max_next_Qs

                # Compute loss
                loss = criterion(pred_Qs, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                #torch.nn.utils.clip_grad_value_(mainQN.parameters(), 10)
                optimizer.step()

                
                # Soft update the target network
                target_QN_state_dict = targetQN.state_dict()
                mainQN_state_dict = mainQN.state_dict()
                for key in mainQN_state_dict:
                    target_QN_state_dict[key] = mainQN_state_dict[key]*tau + target_QN_state_dict[key]*(1-tau)
                targetQN.load_state_dict(target_QN_state_dict)
                
                # compute for vector plot on last episode
                #if ep == train_episodes: 
                #grid.reset_source([(next_state[2],next_state[3])], [2])
                #env.Grid.step_compute()

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

        # Hard update the target network
        #if (ep % update_target_every == 0):
        #    targetQN.load_state_dict(mainQN.state_dict())

    # plot the final model
    #env.Grid.plot_gradient_direction()
    #plt.show()
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
