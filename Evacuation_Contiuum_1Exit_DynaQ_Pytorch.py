import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
import os
import matplotlib.pyplot as plt
from Continuum_Cellspace_Gym import ParticleNavEnv

#############################################
# Neural Network Definitions
#############################################

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_size)
        self.elu = nn.ELU()

        # Initialize weights using Xavier Normal initialization
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

class ModelNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ModelNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Input layer: state_size + action_size (one-hot action)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Output: state_size + 2 (for reward and done)
        self.output_layer = nn.Linear(hidden_size, state_size + 2)
        self.elu = nn.ELU()

        # Initialize weights using Xavier Normal initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, state, action):
        batch_size = state.size(0)
        
        # One-hot encode the action
        action_one_hot = torch.zeros(batch_size, self.action_size, device=state.device)
        action_one_hot[torch.arange(batch_size), action] = 1.0

        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=1)
        
        # Pass through hidden layers with ELU activation
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        
        # Output layer: next_state + reward + done
        out = self.output_layer(x)
        next_state_pred = out[:, :self.state_size]
        reward_pred = out[:, self.state_size]
        done_pred = out[:, self.state_size + 1]
        
        return next_state_pred, reward_pred, done_pred

#############################################
# Replay Buffer
#############################################

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

#############################################
# Dyna-Q Agent with Target Networks for Q and Model
#############################################

class DynaQAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 train_episodes=1000,
                 gamma=0.99,
                 lr=1e-3,
                 model_lr=1e-3,
                 batch_size=64,
                 planning_steps=5,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 decay_percentage=.5,
                 target_update_freq=1000,
                 model_target_update_freq=2000,
                 max_memory=100000,
                 hidden_size=64,
                 device='cpu'):

        self.state_size = state_size
        self.action_size = action_size
        self.train_episodes = train_episodes
        self.gamma = gamma
        self.lr = lr
        self.model_lr = model_lr
        self.batch_size = batch_size
        self.planning_steps = planning_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_percentage = decay_percentage
        self.decay_rate = -np.log(self.epsilon_end) / (self.decay_percentage * self.train_episodes)
        self.target_update_freq = target_update_freq
        self.model_target_update_freq = model_target_update_freq
        self.device = device

        # Initialize epsilon
        self.epsilon = self.epsilon_start

        # Q networks
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer_q = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Model networks (main and target)
        self.model_network = ModelNetwork(state_size, action_size, hidden_size).to(device)
        self.target_model_network = ModelNetwork(state_size, action_size, hidden_size).to(device)
        self.target_model_network.load_state_dict(self.model_network.state_dict())
        self.optimizer_model = optim.Adam(self.model_network.parameters(), lr=self.model_lr)

        self.memory = ReplayBuffer(max_size=max_memory)
        self.timestep = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update_epsilon(self, episode):
        # Logarithmic decay of epsilon
        # epsilon(t) = epsilon_end + (epsilon_start - epsilon_end)/(1 + log(1+t))
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end)*(np.exp(-self.decay_rate*episode))

    def train_q_network(self):
        if len(self.memory) < self.batch_size:
            return None  # Return None if not enough samples

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.SmoothL1Loss()(q_values, q_targets)
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()

        return loss.item()  # Return the loss value

    def train_model_network(self):
        # Train the model using real transitions
        if len(self.memory) < self.batch_size:
            return None  # Return None if not enough samples

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        pred_next_state, pred_reward, pred_done = self.model_network(states, actions)

        loss_state = nn.MSELoss()(pred_next_state, next_states)
        loss_reward = nn.MSELoss()(pred_reward, rewards)
        done_probs = torch.sigmoid(pred_done)
        # Ensure both done_probs and dones have the same shape
        loss_done = nn.BCELoss()(done_probs, dones)  # Both [batch_size]

        loss_model = loss_state + loss_reward + loss_done

        self.optimizer_model.zero_grad()
        loss_model.backward()
        self.optimizer_model.step()

        return loss_model.item()  # Return the loss value

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_target_model_network(self):
        self.target_model_network.load_state_dict(self.model_network.state_dict())

    def planning(self):
        """
        Dyna-Q planning: sample a state (and random action) from memory and simulate
        transitions using the target model network.
        """
        if len(self.memory) < self.batch_size:
            return

        for _ in range(self.planning_steps):
            batch = random.sample(self.memory.buffer, 1)  # sample one random transition
            (state, _, _, _, _) = batch[0]
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Pick a random action
            action = np.random.randint(self.action_size)
            action_t = torch.LongTensor([action]).to(self.device)

            # Predict next transition with target model network
            with torch.no_grad():
                pred_next_state, pred_reward, pred_done = self.target_model_network(state_t, action_t)

            pred_next_state = pred_next_state.squeeze(0).cpu().numpy()
            pred_reward = pred_reward.item()
            pred_done = torch.sigmoid(pred_done).item()
            pred_done = 1 if pred_done > 0.5 else 0

            # Add imagined transition to buffer
            self.memory.add(state, action, pred_reward, pred_next_state, pred_done)

            # Update Q from these imagined transitions
            self.train_q_network()

#############################################
# Training Loop Example
#############################################

def train_dyna_q_agent(
    env,
    agent,
    num_episodes=500,
    max_steps_per_episode=500,
    print_interval=50,
    # Directories to save model files
    qnetwork_save_dir="./model/1ExitDyna/QNetwork",
    modelnetwork_save_dir="./model/1ExitDyna/ModelNetwork",
    # Directory to save training plots
    training_plots_dir="./policy_plots"
):
    # Create directories if they don't exist
    os.makedirs(qnetwork_save_dir, exist_ok=True)
    os.makedirs(modelnetwork_save_dir, exist_ok=True)
    os.makedirs(training_plots_dir, exist_ok=True)

    # Lists to store training metrics
    rewards_per_episode = []
    q_losses = []
    model_losses = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        total_q_loss = 0
        total_model_loss = 0
        step_in_episode = 0

        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_in_episode += 1

            # Train Q-network and model, and accumulate losses
            q_loss = agent.train_q_network()
            if q_loss is not None:
                total_q_loss += q_loss

            model_loss = agent.train_model_network()
            if model_loss is not None:
                total_model_loss += model_loss

            # Planning step
            agent.planning()

            # Update epsilon with logarithmic decay
            agent.update_epsilon(episode)

            # Update target networks periodically
            if agent.timestep % agent.target_update_freq == 0 and agent.timestep > 0:
                agent.update_target_network()
            if agent.timestep % agent.model_target_update_freq == 0 and agent.timestep > 0:
                agent.update_target_model_network()

            agent.timestep += 1

            if done:
                break

        # Save model weights every 500 steps
        if episode % 500 == 0:
            # Save QNetwork
            qnetwork_path = os.path.join(qnetwork_save_dir, f"qnetwork_{episode}.pth")
            torch.save(agent.q_network.state_dict(), qnetwork_path)

            # Save ModelNetwork
            modelnetwork_path = os.path.join(modelnetwork_save_dir, f"modelnetwork_{episode}.pth")
            torch.save(agent.model_network.state_dict(), modelnetwork_path)

        # Compute average losses for the episode
        avg_q_loss = total_q_loss / step_in_episode if step_in_episode > 0 else 0
        avg_model_loss = total_model_loss / step_in_episode if step_in_episode > 0 else 0

        # Append metrics to lists
        rewards_per_episode.append(total_reward)
        q_losses.append(avg_q_loss)
        model_losses.append(avg_model_loss)

        # Save model weights every 500 episodes
        if episode % 500 == 0:
            # Save QNetwork
            qnetwork_path = os.path.join(qnetwork_save_dir, f"qnetwork_episode_{episode}.pth")
            torch.save(agent.q_network.state_dict(), qnetwork_path)

            # Save ModelNetwork
            modelnetwork_path = os.path.join(modelnetwork_save_dir, f"modelnetwork_episode_{episode}.pth")
            torch.save(agent.model_network.state_dict(), modelnetwork_path)

        # Print progress
        if episode % print_interval == 0:
            avg_reward = np.mean(rewards_per_episode[-print_interval:])
            avg_q = np.mean(q_losses[-print_interval:])
            avg_model = np.mean(model_losses[-print_interval:])
            print(f"Episode: {episode}, Average Reward (last {print_interval} eps): {avg_reward:.2f}, "
                  f"Average Q Loss: {avg_q:.4f}, Average Model Loss: {avg_model:.4f}, Epsilon: {agent.epsilon:.3f}")

    # After training, plot and save the loss and reward figures
    plot_training_metrics(q_losses, model_losses, rewards_per_episode, training_plots_dir)

    return rewards_per_episode

def plot_training_metrics(q_losses, model_losses, rewards, save_dir):
    """
    Plots the Q-network loss, Model-network loss, and rewards per episode.

    Args:
    - q_losses (list): List of Q-network loss per episode.
    - model_losses (list): List of Model-network loss per episode.
    - rewards (list): List of total rewards per episode.
    - save_dir (str): Directory where the plots will be saved.
    """
    episodes = np.arange(1, len(rewards) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Losses
    axs[0].plot(episodes, q_losses, label='Q-Network Loss', color='blue')
    axs[0].plot(episodes, model_losses, label='Model-Network Loss', color='orange')
    axs[0].set_title('Training Loss per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Rewards
    axs[1].plot(episodes, rewards, label='Total Reward', color='green')
    axs[1].set_title('Total Reward per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Total Reward')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    figure_filename = 'training_metrics.png'
    figure_filepath = os.path.join(save_dir, figure_filename)
    plt.savefig(figure_filepath, dpi=300)
    plt.close(fig)  # Close the figure to free memory

    print(f"Training metrics figure saved at: {figure_filepath}")

#############################################
# Usage
#############################################

if __name__ == "__main__":
    # Create environment
    env = ParticleNavEnv(env_size=10.0, num_exits=1, num_obstacles=0, change_env_interval=200)
    
    # Hyperparameters as variables
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    train_episodes = 10000
    gamma = 0.999
    lr = 5e-5
    model_lr = 1e-4
    batch_size = 64
    planning_steps = 5
    epsilon_start = 1.0
    epsilon_end = 0.1
    decay_percentage = .5
    target_update_freq = 1000
    model_target_update_freq = 1000
    hidden_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    agent = DynaQAgent(
        state_size=state_dim,
        action_size=action_dim,
        train_episodes=train_episodes,
        gamma=gamma,
        lr=lr,
        model_lr=model_lr,
        batch_size=batch_size,
        planning_steps=planning_steps,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        decay_percentage=decay_percentage,
        target_update_freq=target_update_freq,
        model_target_update_freq=model_target_update_freq,
        hidden_size=hidden_size,
        device=device
    )

    # Directories for saving models (can be easily changed)
    qnetwork_save_dir = "./model/1ExitDyna/QNetwork"
    modelnetwork_save_dir = "./model/1ExitDyna/ModelNetwork"
    training_plots_dir = "./policy_plots"

    # Train the agent and save models periodically
    train_dyna_q_agent(
        env, 
        agent, 
        num_episodes=10000, 
        max_steps_per_episode=1500, 
        print_interval=1,
        qnetwork_save_dir=qnetwork_save_dir,
        modelnetwork_save_dir=modelnetwork_save_dir,
        training_plots_dir=training_plots_dir
    )
