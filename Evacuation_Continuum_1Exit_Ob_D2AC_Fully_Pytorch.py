import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import deque
import matplotlib.pyplot as plt
from Continuum_Cellspace import *

#######################
# Hyperparameters / Config
#######################
N_STEPS = 10         # N-step returns
GAMMA = 0.95         # discount factor
CRITIC_LOSS_COEF = 0.1
LEARNING_RATE = 1e-4

ROLLING_WINDOW = 250 # for reward-plot smoothing

# Multiprocessing parameters
NUM_WORKERS = 4
EPISODES_PER_WORKER = 2500

# Hard cap on steps per episode
MAX_EPISODE_STEPS = 2000

# Periodic saving of model
SAVE_STEP = 1000    # Save the model every these many global episodes

# Periodic environment snapshot
CFG_SAVE_FREQ = 1000  # Take env snapshots every these many global episodes
CFG_SAVE_STEP = 1   # Also snapshot every N environment steps (within an episode)

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

output_dir = os.path.join(output_dir, 'Continuum_1Exit_Ob_D2AC_Fully_Pytorch')
model_saved_path = os.path.join(model_saved_path, 'Continuum_1Exit_Ob_D2AC_Fully_Pytorch')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)

#######################
# Actor-Critic Network
#######################
class ActorCritic(nn.Module):
    def __init__(self, state_size=4, action_size=8):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.actor_lin1 = nn.Linear(128, action_size)
        
        self.l3 = nn.Linear(128, 128)
        self.critic_lin1 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0) 
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic

#######################
# Rollout / Training
#######################
def run_n_steps(env, model, N_steps=10, gamma=0.95,
                do_save_output=False, pathdir=None, step_counter=None):
    """
    Runs up to N_steps or until done.
    Returns lists of (logprobs, values, rewards), plus the final value (G) if not done,
    and whether we ended with done or not.

    If do_save_output=True, we will call env.save_output(...) at each step
    if (step_counter % CFG_SAVE_STEP == 0).
    """
    state = np.array(env.reset(), dtype=np.float32)
    
    log_probs_list, values_list, rewards_list = [], [], []
    done = False
    steps = 0

    while steps < N_steps and not done:
        steps += 1
        if step_counter is not None:
            step_counter[0] += 1  # increment the shared list's first element
            current_step = step_counter[0]
        else:
            current_step = steps

        # Possibly snapshot environment
        if do_save_output and (current_step % CFG_SAVE_STEP == 0) and (pathdir is not None):
            env.save_output(os.path.join(pathdir, f"s.{current_step}"))
        
        s_t = torch.from_numpy(state)
        log_probs, value = model(s_t)
        
        dist = torch.distributions.Categorical(logits=log_probs)
        action = dist.sample()
        
        next_state, reward, done = env.step(action.item())
        next_state = np.array(next_state, dtype=np.float32)
        
        log_probs_list.append(log_probs[action])
        values_list.append(value)
        rewards_list.append(reward)
        
        state = next_state

    # If not done, we bootstrap from final state
    if not done:
        s_t = torch.from_numpy(state)
        with torch.no_grad():
            _, final_value = model(s_t)
        G = final_value.detach()
    else:
        G = torch.zeros(1)

    return log_probs_list, values_list, rewards_list, done, G

def update_params(opt, log_probs_list, values_list, rewards_list, done, G, gamma=0.95):
    """
    Compute N-step returns and update the global model.
    Returns (actor_loss, critic_loss) for logging.
    """
    log_probs_tensor = torch.stack(log_probs_list)  # shape: (n,)
    values_tensor = torch.stack(values_list).squeeze(-1)  # shape: (n,)
    rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
    n = len(rewards_list)

    returns = []
    running_return = G
    for t in reversed(range(n)):
        running_return = rewards_tensor[t] + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    returns_tensor = torch.stack(returns).squeeze(-1)

    advantage = returns_tensor - values_tensor
    critic_loss = advantage.pow(2).mean()

    actor_loss = -(log_probs_tensor * advantage.detach()).mean()

    loss = actor_loss + CRITIC_LOSS_COEF * critic_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return actor_loss.item(), critic_loss.item()

#######################
# Worker Process
#######################
def worker_process(worker_id, global_model, global_rewards, counter, params):
    """
    Each worker:
      - Creates its own env and local optimizer
      - Runs EPISODES_PER_WORKER episodes
      - Each episode can be multiple N-step rollouts until done
      - If the agent doesn't reach done, we forcibly end after MAX_EPISODE_STEPS
      - Periodically saves environment snapshots and model checkpoints
      - Logs only final results per episode
    """
    # Local environment
    env = Cell_Space(
        xmin=0, xmax=10, ymin=0, ymax=10, zmin=0, zmax=2,
        rcut=1.5, dt=0.1, Number=1
    )
    Exit.clear()
    Ob.clear()
    Ob_size.clear()
    Exit.append(np.array([0.5,1.0,0.5]))  # => (5,10,1) in domain
    Ob1 = []
    Ob1.append(np.array([0.5, 0.7, 0.5]))
    Ob.append(Ob1)
    Ob_size.append(2.0)
    
    # Local optimizer
    worker_opt = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)

    episodes_to_run = params['episodes']

    step_counter = [0]  # track environment steps in the current episode

    for ep in range(episodes_to_run):
        with counter.get_lock():
            counter.value += 1
            global_ep = counter.value

        # Decide if we do environment snapshots this episode
        do_save = (global_ep % CFG_SAVE_FREQ == 0)
        pathdir = None
        if do_save:
            # Make a folder for this episode snapshot
            pathdir = os.path.join(output_dir, f"worker_{worker_id}_case_{global_ep}")
            os.makedirs(pathdir, exist_ok=True)
            env.save_output(os.path.join(pathdir, "s.0"))

        step_counter[0] = 0
        total_reward = 0.0
        done = False
        episode_steps = 0  # track how many steps for max-step enforcement

        while not done:
            (log_probs_list, values_list, rewards_list,
             done, G) = run_n_steps(
                 env, global_model, N_STEPS, GAMMA,
                 do_save_output=do_save, pathdir=pathdir, step_counter=step_counter
             )

            actor_loss, critic_loss = update_params(
                worker_opt, 
                log_probs_list,
                values_list,
                rewards_list,
                done, G,
                gamma=GAMMA
            )

            chunk_reward = sum(rewards_list)
            total_reward += chunk_reward
            episode_steps += len(rewards_list)

            # If done or if we hit max steps, forcibly end
            if episode_steps >= MAX_EPISODE_STEPS:
                done = True

            if done and do_save:
                current_step = step_counter[0]
                env.save_output(os.path.join(pathdir, f"s.{current_step}"))

        # Record episode reward
        global_rewards.append(total_reward)

        # Possibly save the model checkpoint
        if global_ep % SAVE_STEP == 0:
            model_path = os.path.join(model_saved_path,
                                      f"ActorCritic_ep{global_ep}_worker{worker_id}.pth")
            torch.save({
                'episode': global_ep,
                'model_state_dict': global_model.state_dict(),
            }, model_path)
            print(f"[Worker {worker_id}] Saved model at global episode {global_ep} -> {model_path}")

        # Final log for the episode (only once)
        print(f"[Worker {worker_id}] Episode {ep+1}/{episodes_to_run} (GlobalEp={global_ep}) done "
              f"after {episode_steps} steps, Reward={total_reward:.2f}, "
              f"ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss:.4f}")

#######################
# MAIN
#######################
def rolling_window_average(values, window_size=250):
    result = []
    cumsum = np.cumsum(np.insert(values, 0, 0))
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_sum = cumsum[i + 1] - cumsum[start_index]
        window_len = i - start_index + 1
        result.append(window_sum / window_len)
    return result

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # safer start method on some systems

    # Global shared model
    global_model = ActorCritic(state_size=4, action_size=8)
    global_model.share_memory()

    manager = mp.Manager()
    global_rewards = manager.list()  # store all episode rewards
    counter = mp.Value('i', 0)       # global episode counter

    params = {
        'episodes': EPISODES_PER_WORKER,
    }

    processes = []
    for w_id in range(NUM_WORKERS):
        p = mp.Process(
            target=worker_process,
            args=(w_id, global_model, global_rewards, counter, params)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    # Convert to local list
    episode_rewards_list = list(global_rewards)
    print(f"Total episodes completed across all workers: {len(episode_rewards_list)}")

    # Final model save after all workers done
    final_model_path = os.path.join(
        model_saved_path, f"ActorCritic_final_{len(episode_rewards_list)}.pth"
    )
    torch.save({
        'episode': len(episode_rewards_list),
        'model_state_dict': global_model.state_dict(),
    }, final_model_path)
    print("Saved final model to:", final_model_path)

    # Plot the episode rewards and rolling average
    if len(episode_rewards_list) > 0:
        plt.figure()
        plt.plot(episode_rewards_list, label='Episode Reward')
        ma_rewards = rolling_window_average(
            episode_rewards_list, window_size=ROLLING_WINDOW
        )
        plt.plot(ma_rewards, label=f'{ROLLING_WINDOW}-Episode Avg', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Actor-Critic: Episode Rewards')
        plt.legend()
        plot_path = os.path.join(output_dir, 'actor_critic_rewards.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved reward plot to {plot_path}")

    print("Done. All workers finished.")
