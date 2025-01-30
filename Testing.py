import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Import your environment and the DQN definition
# e.g. from Cell_Space_Env import Cell_Space
# and from your training code or a separate file import your DQN model
from Continuum_Cellspace import Cell_Space, desire_velocity, relaxation_time, agent_size, door_size

# Adjust the import for your DQN if needed
from Evacuation_Continuum_1Exit_Ob_DQN_Fully_Pytorch import DQN  # Or whichever file contains the DQN class definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualizing cases for debugging
output_dir = './testing'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, '1ExitDQN')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def load_trained_model(checkpoint_path, state_size, action_size=8):
    """
    Loads a trained model from 'checkpoint_path' into a new DQN instance.
    Returns the loaded DQN.
    """
    model = DQN(state_size=state_size, action_size=action_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['mainQN_state_dict'])
    model.eval()
    return model

def run_trained_model(env, model, case, max_steps=2000):
    """
    Runs a single episode using the trained DQN policy.
    Returns the number of steps taken until done (or until max_steps).
    """
    # Extract initial state
    state = env.get_state()  # or env.reset() if not done outside
    steps = 0
    done = False
    testing_pathdir = os.path.join(output_dir, f'case_{case}')
    if not os.path.isdir(testing_pathdir):
                os.mkdir(testing_pathdir)

    while not done and steps < max_steps:
        # Convert state into float32 tensor
        state_arr = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(device)

        # Get Q-values and choose the best action (greedy)
        with torch.no_grad():
            Qs = model(state_tensor).cpu().numpy()[0]
        action = np.argmax(Qs)

        env.save_output(os.path.join(testing_pathdir, f's.{steps}'))

        # Take step in environment
        next_state, _, done = env.step(action)
        
        # Update
        state = next_state
        steps += 1
    
    return steps

def run_optimal_policy(env, case, max_steps=2000):
    """
    Runs a single episode using step_optimal_single_particle.
    Returns the number of steps taken until the agent either exits or we hit max_steps.
    """
    steps = 0
    done = False
    testing_pathdir = os.path.join(output_dir, f'optimal_case_{case}')
    if not os.path.isdir(testing_pathdir):
                os.mkdir(testing_pathdir)

    while not done and steps < max_steps:
        env.save_output(os.path.join(testing_pathdir, f's.{steps}'))
        
        # The step_optimal_single_particle method returns (next_state, reward, done)
        next_state, _, done = env.step_optimal_single_particle()
        steps += 1

    return steps

def distance_to_nearest_exit(env):
    """
    Returns the distance from the agent to the nearest exit in the current environment.
    """
    p = env.agent
    dr_min = np.inf
    for e in env.Exit:
        dist = np.linalg.norm(p.position - e)
        if dist < dr_min:
            dr_min = dist
    return dr_min

def main():
    # -----------------------
    # Set up environment parameters
    # -----------------------
    numExits = 2
    numObs = 0

    # Same domain size & other parameters as used in training
    # The environment's reset() randomizes positions, so we must control seeds
    env_params = {
        "xmin": 0,
        "xmax": 10,
        "ymin": 0,
        "ymax": 10,
        "zmin": 0,
        "zmax": 2,
        "rcut": 1.5,
        "dt": 0.1,
        "Number": 1,
        # "numExits": numExits,
        # "numObs": numObs
    }

    # DQN's state_size: 4 + 2*numExits + 2*numObs
    state_size = 4
    action_size = 8  # environment has 8 discrete actions

    # -----------------------
    # Load trained DQN model
    # -----------------------
    # Uncomment path you wish to load
    checkpoint_path = './model/Continuum_1Exit_Ob_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
    #checkpoint_path = './model/Continuum_2ExitRandom_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep50000.pth'
    dqn_model = load_trained_model(checkpoint_path, state_size, action_size=action_size)

    # -----------------------
    # Evaluation settings
    # -----------------------
    n_episodes = 100
    max_steps = 2000

    # We'll store: (initial_dist, steps_dqn, steps_opt)
    comparison_data = []

    # -----------------------
    # Run multiple episodes
    # -----------------------
    for seed in range(n_episodes):
        # 1) Fix the random seed so that agent/exit/obs placement is reproducible
        np.random.seed(seed)

        # 2) Create the environment with that random seed
        env = Cell_Space(**env_params)
        initial_state = env.reset()  # random config, but consistent for this seed
        init_dist = distance_to_nearest_exit(env)

        # 3) Run the DQN agent
        steps_dqn = run_trained_model(env, dqn_model, seed, max_steps=max_steps)

        # 4) Rebuild the same environment again with the same seed,
        #    so we have the exact same initial positions.
        np.random.seed(seed)
        env_opt = Cell_Space(**env_params)
        env_opt.reset()

        # 5) Run the optimal policy
        steps_opt = run_optimal_policy(env_opt, seed, max_steps=max_steps)

        # 6) Store for plotting
        comparison_data.append((init_dist, steps_dqn, steps_opt))

    # ------------
    # Plot results
    # ------------
    # We'll make a scatter plot with x=init_dist, y=#steps
    # We'll plot two different markers: one for DQN, one for Optimal

    distances = [d[0] for d in comparison_data]
    dqn_steps = [d[1] for d in comparison_data]
    opt_steps = [d[2] for d in comparison_data]

    plt.figure(figsize=(8, 6))
    # Plot DQN data
    plt.scatter(distances, dqn_steps, color='blue', alpha=0.7, label='DQN Steps')
    # Plot Optimal data
    plt.scatter(distances, opt_steps, color='red', alpha=0.7, label='Optimal Steps')
    
    plt.xlabel('Initial Distance to Nearest Exit')
    plt.ylabel('Number of Steps to Exit')
    plt.title('Comparison of DQN vs. Optimal Policy\n(100 Episodes, Same Initial Conditions per Seed)')
    plt.legend()
    plt.grid(True)

    # Optionally, save the figure
    #plt.savefig('comparison_DQN_vs_Optimal.png', dpi=200, bbox_inches='tight')

    # Show the figure
    plt.show()

    # Print summary
    print("Evaluation Results (first 10 episodes):")
    for i, (dist, s_dqn, s_opt) in enumerate(comparison_data[:99]):
        print(f"Seed {i}: initDist={dist:.2f}, DQN_Steps={s_dqn}, OPT_Steps={s_opt}")

    # You could also compute average steps if you want
    avg_dqn_steps = np.mean(dqn_steps)
    avg_opt_steps = np.mean(opt_steps)
    print(f"\nAverage Steps DQN: {avg_dqn_steps:.2f} vs Optimal: {avg_opt_steps:.2f}")

if __name__ == "__main__":
    main()
