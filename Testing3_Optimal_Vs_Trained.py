import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv

# 1) Imports
# ----------------------------------------------------------------------
# This environment returns the nested state:
#   (ax, ay, avx, avy, (dist1, angle1), (dist2, angle2), (dist3, angle3))
from Continuum_Cellspace5 import Cell_Space

# This DQN must be the one that can handle nested states in 'forward()'
from Evacuation_Continuum_Randomize4 import DQN  # <-- Adjust to the file that defines your new DQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For saving output snapshots and CSV logs
output_dir = './Testing'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'Random_3Exit_NoObstacles_MultiDistAngleNestedState_RewardShaping')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# ----------------------------------------------------------------------
# 2) Loading the trained model
# ----------------------------------------------------------------------
def load_trained_model(checkpoint_path, numExits=3, action_size=8):
    """
    Loads a trained model from 'checkpoint_path' into a new DQN instance.
    Returns the loaded DQN.
    """
    model = DQN(numExits=numExits, action_size=action_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['mainQN_state_dict'])
    model.eval()
    return model

# ----------------------------------------------------------------------
# 3) Functions to run the agent
# ----------------------------------------------------------------------
def run_trained_model(env, model, case_idx, max_steps=2000):
    """
    Runs a single episode using the trained DQN policy.
    Returns the number of steps taken until done (or until max_steps).
    
    Because the environment returns a nested state, we pass [nested_state]
    to model.forward(), which expects a list of states.
    """
    state_nested = env.get_state()
    steps = 0
    done = False

    testing_pathdir = os.path.join(output_dir, f'case_{case_idx}')
    if not os.path.isdir(testing_pathdir):
        os.mkdir(testing_pathdir)

    while not done and steps < max_steps:
        # Model expects a list of nested states for batch dimension
        with torch.no_grad():
            Qs = model([state_nested]).cpu().numpy()[0]  # shape => [action_size]
        action = np.argmax(Qs)

        # Optionally save environment snapshot
        env.save_output(os.path.join(testing_pathdir, f's.{steps}'))

        # Step environment
        next_state_nested, _, done = env.step(action)
        state_nested = next_state_nested
        steps += 1

    return steps

def run_optimal_policy(env, case_idx, max_steps=2000):
    """
    Runs a single episode using step_optimal_single_particle().
    Returns the number of steps until exit or max_steps.
    """
    steps = 0
    done = False

    testing_pathdir = os.path.join(output_dir, f'optimal_case_{case_idx}')
    if not os.path.isdir(testing_pathdir):
        os.mkdir(testing_pathdir)

    while not done and steps < max_steps:
        env.save_output(os.path.join(testing_pathdir, f's.{steps}'))
        next_state_nested, _, done = env.step_optimal_single_particle()
        steps += 1

    return steps

def extract_distance_from_state(state, numExits=3):
    """
    For an environment with 'numExits' exits, the state is:
      (ax, ay, avx, avy,
       (dist1, angle1),
       (dist2, angle2),
       ...
       (distN, angleN))

    We'll return the minimum distance among all exits (the nearest exit).
    """
    # The first 4 entries are agent coords => state[:4]
    # Then from index=4 onward, each item is a tuple (dist, angle).
    # e.g. state[4], state[5], state[6] if numExits=3 => (dist1, angle1), (dist2, angle2), (dist3, angle3)
    exit_tuples = state[4:]  # list of (dist, angle)
    distances = [ex_tuple[0] for ex_tuple in exit_tuples]  # each ex_tuple[0] is distance
    return min(distances)

# ----------------------------------------------------------------------
# 4) Main
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Environment parameters
    # ------------------------------------------------------------------
    numExits = 3
    numObs = 0
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
        "numExits": numExits,
        "numObs": numObs
    }

    # We'll define the DQN's dimension by how many exits are in the environment.
    # We won't flatten states in the environment or test script,
    # but our DQN internally handles nested states of length 4 + (2 * numExits).
    action_size = 8  # environment has 8 discrete actions

    # ------------------------------------------------------------------
    # Load the trained DQN model
    # ------------------------------------------------------------------
    # Adjust checkpoint_path to wherever your 3-exit model is saved
    checkpoint_path = './model/Continuum_3ExitRandom_MultiDistAngleNestedState_RewardShaping_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep30000.pth'
    dqn_model = load_trained_model(checkpoint_path, numExits=numExits, action_size=action_size)

    # ------------------------------------------------------------------
    # Testing settings
    # ------------------------------------------------------------------
    n_episodes = 100
    max_steps = 2000

    # We'll store (initial_dist, steps_dqn, steps_opt) for each run
    comparison_data = []

    for seed in range(n_episodes):
        # 1) Fix random seed => reproducible initial environment
        np.random.seed(seed)

        # 2) Create environment & reset
        env = Cell_Space(**env_params)
        init_state = env.reset()
        init_dist = extract_distance_from_state(init_state, numExits)

        # 3) Run trained model
        steps_dqn = run_trained_model(env, dqn_model, seed, max_steps=max_steps)

        # 4) Rebuild environment with same seed => same initial conditions
        np.random.seed(seed)
        env_opt = Cell_Space(**env_params)
        env_opt.reset()

        # 5) Run optimal
        steps_opt = run_optimal_policy(env_opt, seed, max_steps=max_steps)

        # 6) Store
        comparison_data.append((init_dist, steps_dqn, steps_opt))

    # ------------------------------------------------------------------
    # Save results to CSV
    # ------------------------------------------------------------------
    csv_filename = os.path.join(output_dir, 'test_evaluation_results.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Seed', 'InitialDistance', 'DQN_Steps', 'Optimal_Steps'])
        for seed_idx, (dist, s_dqn, s_opt) in enumerate(comparison_data):
            writer.writerow([seed_idx, dist, s_dqn, s_opt])
    print(f"Saved evaluation results to {csv_filename}")

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    distances = [d[0] for d in comparison_data]
    dqn_steps = [d[1] for d in comparison_data]
    opt_steps = [d[2] for d in comparison_data]

    plt.figure(figsize=(8, 6))
    plt.scatter(distances, dqn_steps, color='blue', alpha=0.7, label='DQN Steps')
    plt.scatter(distances, opt_steps, color='red', alpha=0.7, label='Optimal Steps')
    plt.xlabel('Initial Distance to Nearest Exit (Multi-Exit Nested State)')
    plt.ylabel('Number of Steps to Exit')
    plt.title('Comparison of DQN vs. Optimal Policy\n(100 Episodes, Same Seeds, 3-Exit Env)')
    plt.legend()
    plt.grid(True)

    # Also save the plot
    plot_path = os.path.join(output_dir, 'dqn_vs_optimal.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    plt.show()

    # Print average steps
    avg_dqn = np.mean(dqn_steps)
    avg_opt = np.mean(opt_steps)
    print(f"\nAverage Steps => DQN: {avg_dqn:.2f}, Optimal: {avg_opt:.2f}")

if __name__ == "__main__":
    main()
