"""
Example of a custom PPO training script for an image-like occupancy-grid environment.
Your environment (Cell_Space) must return states of shape (3, rows, cols),
with discrete 8 actions. We'll do 4 parallel envs, gather rollouts of length=1024
(so 4096 steps total each iteration), then run 10 epochs of PPO updates.

Hyperparameters set:
- total_timesteps = 1e7
- gamma=0.99, gae_lambda=0.95
- clip_range=0.2
- lr=1e-4
- ent_coef=0.001
- 4 parallel envs
- 10 PPO epochs
"""

import os
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random
import math
import time

# Import your environment: shape => (3,rows,cols), 8 discrete actions
from Continuum_Cellspace7_OccupancyImage import Cell_Space

# --------------------------
# 1) PPO Hyperparameters
# --------------------------
NUM_ENVS         = 4            # parallel envs
ROLLOUT_STEPS    = 1024         # per env => total=4*1024=4096 steps each iteration
PPO_EPOCHS       = 10           # number of epochs per update
BATCH_SIZE       = 256          # minibatch size
GAMMA            = 0.99
LAMBDA           = 0.95         # GAE-lambda
CLIP_RANGE       = 0.2
ENT_COEF         = 0.001
LR               = 1e-4
TOTAL_TIMESTEPS  = 10_000_000   # 1e7 steps
ROWS             = 20           # must match env
COLS             = 20           # must match env
ACTION_SIZE      = 8            # discrete
SEED             = 42

# ---------------------------------
# 2) Actor-Critic Network (CNN)
# ---------------------------------
class ActorCritic(nn.Module):
    """
    Shared CNN trunk => separate actor, critic heads.
    The actor outputs a (B, action_size) logit => categorical distribution.
    The critic outputs a (B,) scalar => value function.
    """
    def __init__(self, in_channels=3, rows=20, cols=20, action_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.action_size = action_size

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,stride=2,padding=1)

        # compute shape after these convs
        def conv_out_size(in_size):
            return (in_size+2*1 - (3-1)-1)//2 +1

        c1_h = conv_out_size(rows)
        c1_w = conv_out_size(cols)
        c2_h = conv_out_size(c1_h)
        c2_w = conv_out_size(c1_w)
        c3_h = conv_out_size(c2_h)
        c3_w = conv_out_size(c2_w)
        flatten_dim = 64*(c3_h*c3_w)

        # FC trunk
        self.fc = nn.Linear(flatten_dim, 256)

        # Actor head => produce policy logits (B, action_size)
        self.actor = nn.Linear(256, action_size)

        # Critic head => produce state-value => shape (B,)
        self.critic= nn.Linear(256, 1)

        # init
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.fc, self.actor, self.critic]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x => shape (B, 3, rows, cols)
        returns => policy_logits (B,action_size), value (B,)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        policy = self.actor(x)
        value  = self.critic(x).squeeze(-1)
        return policy, value

# ---------------------------------
# 3) SubprocVecEnv for parallel env
# ---------------------------------
# minimal approach => each env in a process => communicate via pipe
# For brevity, we do a simplified version with mp.Process

def make_env(seed_offset=0):
    """
    Create a function that returns a single env
    """
    def _init():
        env = Cell_Space(
            xmin=0, xmax=10,
            ymin=0, ymax=10,
            zmin=0, zmax=2,
            dt=0.1,
            Number=1,
            numExits=1,
            numObs=0
        )
        return env
    return _init

class Worker(mp.Process):
    """
    Each worker runs an env, communicates transitions back to the main process
    """
    def __init__(self, env_fn, pipe, worker_id):
        super().__init__()
        self.env_fn = env_fn
        self.pipe   = pipe
        self.id     = worker_id
        np.random.seed(SEED+worker_id)

    def run(self):
        env = self.env_fn()
        obs = env.reset()
        done= False
        while True:
            cmd, data = self.pipe.recv()
            if cmd=="step":
                action = data
                next_obs, reward, done = env.step(action)
                if done:
                    next_obs = env.reset()
                self.pipe.send((obs, action, reward, next_obs, done))
                obs = next_obs
            elif cmd=="reset":
                obs = env.reset()
                done= False
                self.pipe.send(obs)
            elif cmd=="close":
                self.pipe.close()
                break
            else:
                pass

class SubprocVecEnv:
    """
    Manage multiple worker processes => step, reset
    """
    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.parent_pipes = []
        self.procs = []
        for idx, fn in enumerate(env_fns):
            parent_pipe, child_pipe = mp.Pipe()
            proc = Worker(fn, child_pipe, idx)
            proc.daemon = True
            proc.start()
            self.parent_pipes.append(parent_pipe)
            self.procs.append(proc)

    def reset(self):
        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.parent_pipes]
        return np.array(results, dtype=object)

    def step(self, actions):
        # actions => shape (n_envs,)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(("step", act))
        results = [pipe.recv() for pipe in self.parent_pipes]
        # each => (obs, action, reward, next_obs, done)
        obs, act, rew, nxt, d = zip(*results)
        return np.array(obs,dtype=object),\
               np.array(act),\
               np.array(rew),\
               np.array(nxt,dtype=object),\
               np.array(d)

    def close(self):
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for p in self.procs:
            p.join()

# ---------------------------------
# 4) Helper: Categorical action
# ---------------------------------
def select_action(policy_logits):
    """
    policy_logits => shape (B, action_size)
    return => shape (B,) with discrete actions
    """
    dist = torch.distributions.Categorical(logits=policy_logits)
    action = dist.sample()
    return action, dist.log_prob(action), dist

# ---------------------------------
# 5) GAE advantage
# ---------------------------------
def compute_gae(rewards, dones, values, next_values, gamma=0.99, lam=0.95):
    """
    rewards: (T, n_env)
    dones:   (T, n_env)
    values:  (T, n_env)
    next_values: (n_env,) for the step after last
    Return advantage, returns => shape (T,n_env)
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        if t==T-1:
            next_val = next_values
            mask = 1.0 - dones[t]
        else:
            next_val= values[t+1]
            mask = 1.0 - dones[t]
        delta = rewards[t] + gamma*next_val*mask - values[t]
        advantages[t] = last_gae = delta + gamma*lam*mask*last_gae
    returns = advantages + values
    return advantages, returns

# ---------------------------------
# 6) PPO training loop
# ---------------------------------
def main():
    mp.set_start_method("spawn")  # for Windows or some systems

    # Create parallel envs
    env_fns = [make_env(i) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    # Create actor-critic
    net = ActorCritic(in_channels=3, rows=ROWS, cols=COLS, action_size=ACTION_SIZE)
    net.train()
    net.to(torch.device('cpu'))

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # initial reset
    obs = vec_env.reset()  # shape => (NUM_ENVS,) each an array(3,rows,cols)

    # We'll store episode returns for plotting
    episode_rewards = []
    ep_return = np.zeros(NUM_ENVS, dtype=np.float32)

    # to keep track of total steps
    total_steps = 0
    updates = 0

    # loop until we hit total_timesteps
    while total_steps < TOTAL_TIMESTEPS:
        # collect rollouts => each env does ROLLOUT_STEPS => total= ROLLOUT_STEPS*NUM_ENVS
        rollout_obs      = []
        rollout_actions  = []
        rollout_logprobs = []
        rollout_rewards  = []
        rollout_dones    = []
        rollout_values   = []

        with torch.no_grad():
            for t in range(ROLLOUT_STEPS):
                # forward pass => get policy,value
                # shape => (NUM_ENVS, 3, rows, cols) => torch => (NUM_ENVS,3,rows,cols)
                batch_arr = []
                for i in range(NUM_ENVS):
                    batch_arr.append(obs[i])  # shape (3,rows,cols)
                batch_np = np.array(batch_arr,dtype=np.float32)
                input_t = torch.from_numpy(batch_np)  # => shape(N,3,rows,cols)
                policy_logits, values = net(input_t)

                # select action
                # policy_logits => shape(N, action_size)
                # values => shape(N)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                logprob= dist.log_prob(action)

                # step in env
                actions_np = action.numpy()
                old_obs = obs
                obs_, acts_, rews_, nxt_, dns_ = vec_env.step(actions_np)
                # obs_, acts_, ... => each is shape (NUM_ENVS, ...?)

                # store
                rollout_obs.append(old_obs)
                rollout_actions.append(actions_np)
                rollout_logprobs.append(logprob.numpy())
                rollout_rewards.append(rews_)
                rollout_dones.append(dns_)
                rollout_values.append(values.numpy())

                obs = nxt_

                # update ep_return
                ep_return += rews_
                for i, d in enumerate(dns_):
                    if d:
                        # record the episode reward
                        episode_rewards.append(ep_return[i])
                        ep_return[i] = 0.0

            total_steps += ROLLOUT_STEPS*NUM_ENVS

        # we have T=ROLLOUT_STEPS steps in shape => (T, num_envs,...)
        # convert to np array
        rollout_obs      = np.array(rollout_obs, dtype=object)        # shape (T, N)
        rollout_actions  = np.array(rollout_actions, dtype=np.int64)  # (T, N)
        rollout_logprobs = np.array(rollout_logprobs, dtype=np.float32)
        rollout_rewards  = np.array(rollout_rewards, dtype=np.float32)
        rollout_dones    = np.array(rollout_dones,   dtype=bool)
        rollout_values   = np.array(rollout_values,  dtype=np.float32)

        # get final value => next_values => shape (N,)
        with torch.no_grad():
            batch_arr = []
            for i in range(NUM_ENVS):
                batch_arr.append(obs[i])
            batch_np = np.array(batch_arr,dtype=np.float32)
            input_t= torch.from_numpy(batch_np)
            pol, val = net(input_t)
            next_values = val.numpy()  # shape (N,)

        # compute advantage, returns
        advantages, returns = compute_gae(
            rewards=rollout_rewards, 
            dones= rollout_dones,
            values=rollout_values,
            next_values= next_values,
            gamma= GAMMA,
            lam= LAMBDA
        ) # => shape (T, N)

        # flatten all of them => shape (T*N, ..)
        T, N = rollout_actions.shape
        flat_obs       = rollout_obs.reshape(T*N)  # each is array(3,rows,cols)
        flat_actions   = rollout_actions.reshape(T*N)
        flat_logprobs  = rollout_logprobs.reshape(T*N)
        flat_advs      = advantages.reshape(T*N)
        flat_returns   = returns.reshape(T*N)
        flat_values    = rollout_values.reshape(T*N)

        # We'll do PPO update => run PPO_EPOCHS => each epoch => mini-batch from the T*N set
        dataset_size = T*N
        idx_arr = np.arange(dataset_size)
        for epoch_i in range(PPO_EPOCHS):
            np.random.shuffle(idx_arr)
            start=0
            while start< dataset_size:
                end= start+BATCH_SIZE
                batch_idx = idx_arr[start:end]
                start= end

                # gather batch
                obs_list  = [flat_obs[i] for i in batch_idx]   # each shape (3,rows,cols)
                act_batch = flat_actions[batch_idx]
                old_lp    = flat_logprobs[batch_idx]
                adv_batch = flat_advs[batch_idx]
                ret_batch = flat_returns[batch_idx]
                old_val   = flat_values[batch_idx]

                # forward
                # shape => (b,3,rows,cols)
                arr_np = np.array(obs_list,dtype=np.float32)
                input_t= torch.from_numpy(arr_np)
                policy_logits, new_values = net(input_t)  # shape => (b,8), (b,)

                dist   = torch.distributions.Categorical(logits=policy_logits)
                new_lp = dist.log_prob(torch.from_numpy(act_batch))

                ratio  = torch.exp(new_lp - torch.from_numpy(old_lp))

                # clipped surrogate
                adv_t  = torch.from_numpy(adv_batch)
                ratio_clipped = torch.clamp(ratio, 1.0-CLIP_RANGE, 1.0+CLIP_RANGE)
                obj1  = ratio * adv_t
                obj2  = ratio_clipped * adv_t
                policy_loss = -torch.mean(torch.min(obj1, obj2))

                # value loss => MSE( new_values, returns )
                ret_t = torch.from_numpy(ret_batch)
                val_loss = F.mse_loss(new_values, ret_t)

                # entropy
                entropy = dist.entropy().mean()

                loss= policy_loss + 0.5*val_loss - ENT_COEF*entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        updates+=1
        # optional => print progress
        if updates%10==0:
            avg_rew = np.mean(episode_rewards[-100:]) if len(episode_rewards)>0 else 0
            print(f"Update: {updates}, total_steps: {total_steps}, avg_100ep_reward: {avg_rew:.3f}")

        # we can break if total_steps> some limit
        if total_steps>= TOTAL_TIMESTEPS:
            break

    # done => plot final
    # we have episode_rewards => e.g. we can do a moving average
    rewards_plot = "ppo_occupancy_rewards.png"
    if len(episode_rewards)>0:
        window=100
        def ma(x, w=100):
            y=[]
            s=0
            d= deque()
            for val in x:
                d.append(val)
                s+=val
                if len(d)>w:
                    s-=d.popleft()
                y.append(s/len(d))
            return y
        plt.figure()
        plt.plot(episode_rewards, label="Episode reward")
        plt.plot(ma(episode_rewards, window), label=f"MA({window})", color='red')
        plt.title("PPO Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.savefig(rewards_plot)
        print(f"Saved reward plot to {rewards_plot}")

    vec_env.close()
    print("Done training PPO for occupancy-grid.")
    # optionally save the final net
    torch.save(net.state_dict(), "ppo_occupancy_net_final.pth")
