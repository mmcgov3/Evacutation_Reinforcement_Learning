import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class Particle:
    def __init__(self, ID, x, y, z, vx, vy, vz, mass=80.0):
        self.position = np.array([x, y, z], dtype=np.float32)
        self.velocity = np.array([vx, vy, vz], dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)
        self.mass = mass
        self.ID = ID

    def leapfrog(self, dt, stage):
        if stage == 0:
            self.velocity += dt / 2 * self.acc
            self.position += dt * self.velocity
        else:
            self.velocity += dt / 2 * self.acc

class ParticleNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 env_size=50.0,
                 num_exits=1,
                 num_obstacles=1,
                 change_env_interval=200,
                 agent_size=0.5,
                 door_size=1.0,
                 action_force=1.0,
                 desire_velocity=2.0,
                 relaxation_time=0.5,
                 delta_t=0.1,
                 f_wall_lim=100.0,
                 f_collision_lim=100.0,
                 dis_lim=None,
                 seed=None):
        super(ParticleNavEnv, self).__init__()
        
        if seed is not None:
            np.random.seed(seed)

        self.env_size = env_size
        self.num_exits = num_exits
        self.num_obstacles = num_obstacles
        self.change_env_interval = change_env_interval

        self.agent_size = agent_size
        self.door_size = door_size
        self.action_force = action_force
        self.desire_velocity = desire_velocity
        self.relaxation_time = relaxation_time
        self.delta_t = delta_t
        self.f_wall_lim = f_wall_lim
        self.f_collision_lim = f_collision_lim

        # Define agent "exit" radius (when agent is considered to have reached the exit)
        if dis_lim is None:
            self.dis_lim = (self.agent_size + self.door_size) / 2
        else:
            self.dis_lim = dis_lim

        # Boundaries: (0,0) bottom-left and (env_size, env_size) top-right in XY plane
        self.L = np.array([[0, self.env_size],
                           [0, self.env_size],
                           [0, 2]], dtype=np.float32)  # Z is trivial

        # Action space: 8 directions
        diag = np.sqrt(2) / 2
        self.base_actions = np.array([[0, 1, 0],
                                      [-diag, diag, 0],
                                      [-1, 0, 0],
                                      [-diag, -diag, 0],
                                      [0, -1, 0],
                                      [diag, -diag, 0],
                                      [1, 0, 0],
                                      [diag, diag, 0]], dtype=np.float32) * self.action_force
        self.action_space = spaces.Discrete(len(self.base_actions))

        # Observation space:
        # [agent_x, agent_y, agent_vx, agent_vy,
        #  rel_exit_x, rel_exit_y,
        #  obstacle_distances_in_8_directions]
        # Positions and velocities can be up to env_size in magnitude;
        # distances can also be up to env_size. We'll just use a large box.
        high = np.array([np.finfo(np.float32).max] * (6 + 8), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.reset_env_change_counter = 0
        self.timestep = 0

        # Initialize lists for exits and obstacles
        self.Exits = []
        self.Ob = []
        self.Ob_size = []

        self.reset()

    def reset(self):
        # Reset step counter
        self.timestep = 0
        self._randomize_environment()

        # Initialize agent
        self.agent = self._spawn_agent()

        # Create observation
        obs = self._get_observation()
        return obs

    def step(self, action):
        self.timestep += 1

        # Zero acceleration
        self.agent.acc[:] = 0.0

        # Apply forces
        self._apply_environment_forces()
        self._apply_action(action)

        # Integrate motion
        self._integrate()

        # Compute reward and done
        done, reward = self._check_done_and_reward()

        obs = self._get_observation()

        # Change environment after N steps if required
        if self.change_env_interval > 0 and (self.timestep % self.change_env_interval == 0):
            self._randomize_environment()
            # After randomizing, ensure agent not placed inside obstacle:
            # We won't move agent for simplicity, but you could re-spawn or handle overlap here.

        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1, self.env_size + 1)
        ax.set_ylim(-1, self.env_size + 1)
        ax.set_aspect('equal')
        ax.set_title('Particle Navigation Environment')

        # Draw walls as black lines
        walls = [
            ((0, 0), (self.env_size, 0)),            # Bottom
            ((self.env_size, 0), (self.env_size, self.env_size)),  # Right
            ((self.env_size, self.env_size), (0, self.env_size)),    # Top
            ((0, self.env_size), (0, 0))             # Left
        ]
        for wall in walls:
            ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=2)

        # Draw exits as red rectangles on walls
        for exit_pos in self.Exits:
            ex_x, ex_y, ex_z = exit_pos
            if ex_x == 0:  # Left wall
                rect = patches.Rectangle((ex_x - self.door_size / 2, ex_y - self.door_size / 2),
                                         self.door_size, self.door_size, linewidth=1, edgecolor='red', facecolor='red')
            elif ex_x == self.env_size:  # Right wall
                rect = patches.Rectangle((ex_x - self.door_size / 2, ex_y - self.door_size / 2),
                                         self.door_size, self.door_size, linewidth=1, edgecolor='red', facecolor='red')
            elif ex_y == 0:  # Bottom wall
                rect = patches.Rectangle((ex_x - self.door_size / 2, ex_y - self.door_size / 2),
                                         self.door_size, self.door_size, linewidth=1, edgecolor='red', facecolor='red')
            elif ex_y == self.env_size:  # Top wall
                rect = patches.Rectangle((ex_x - self.door_size / 2, ex_y - self.door_size / 2),
                                         self.door_size, self.door_size, linewidth=1, edgecolor='red', facecolor='red')
            else:
                continue  # Should not happen
            ax.add_patch(rect)

        # Draw obstacles as blue circles
        for (ob_pos, ob_size) in zip(self.Ob, self.Ob_size):
            ob_x, ob_y, ob_z = ob_pos
            circle = patches.Circle((ob_x, ob_y), ob_size / 2, linewidth=1, edgecolor='blue', facecolor='blue')
            ax.add_patch(circle)

        # Draw agent as a green circle
        agent_x, agent_y, agent_z = self.agent.position
        agent_circle = patches.Circle((agent_x, agent_y), self.agent_size, linewidth=1, edgecolor='green', facecolor='green')
        ax.add_patch(agent_circle)

        plt.show()

    def close(self):
        plt.close()

    def _randomize_environment(self):
        # Clear existing exits and obstacles
        self.Exits = []
        self.Ob = []
        self.Ob_size = []

        # Define walls
        walls = ['bottom', 'right', 'top', 'left']

        # Minimum distance from corners to place exits
        min_exit_distance = self.door_size / 2 + 0.1  # Adding a small margin

        # Maximum attempts to place each exit to avoid infinite loops
        max_attempts = 100

        # Store exit positions per wall to check for overlaps
        exits_per_wall = {wall: [] for wall in walls}

        for _ in range(self.num_exits):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                wall = random.choice(walls)
                if wall in ['bottom', 'top']:
                    # Choose x position along the wall
                    ex_x = np.random.uniform(min_exit_distance, self.env_size - min_exit_distance)
                    ex_y = 0.0 if wall == 'bottom' else self.env_size
                else:  # 'left' or 'right'
                    # Choose y position along the wall
                    ex_y = np.random.uniform(min_exit_distance, self.env_size - min_exit_distance)
                    ex_x = 0.0 if wall == 'left' else self.env_size

                # Check for overlapping exits on the same wall
                overlap = False
                for existing_ex in exits_per_wall[wall]:
                    if wall in ['bottom', 'top']:
                        distance = abs(ex_x - existing_ex[0])
                    else:
                        distance = abs(ex_y - existing_ex[1])
                    if distance < self.door_size:
                        overlap = True
                        break

                if not overlap:
                    # Place the exit
                    self.Exits.append(np.array([ex_x, ex_y, 0.5], dtype=np.float32))
                    exits_per_wall[wall].append([ex_x, ex_y])
                    placed = True
                attempts += 1

            if not placed:
                print(f"Warning: Could not place exit after {max_attempts} attempts.")

        # Place obstacles ensuring they do not block exits
        self._place_obstacles()

    def _place_obstacles(self):
        # Define buffer zone around exits where obstacles cannot be placed
        buffer_zone = self.door_size + 2.0  # Adjust buffer as needed

        for _ in range(self.num_obstacles):
            placed = False
            attempts = 0
            max_attempts = 100
            while not placed and attempts < max_attempts:
                ob_x = np.random.uniform(self.agent_size, self.env_size - self.agent_size)
                ob_y = np.random.uniform(self.agent_size, self.env_size - self.agent_size)
                ob_z = 0.5  # Trivial Z

                # Random obstacle size between 1.5 and 3.0
                ob_size = np.random.uniform(1.5, 3.0)

                # Check buffer zones around exits
                too_close = False
                for ex in self.Exits:
                    dist = np.linalg.norm(np.array([ob_x, ob_y]) - np.array([ex[0], ex[1]]))
                    if dist < buffer_zone:
                        too_close = True
                        break

                if too_close:
                    attempts += 1
                    continue

                # Check for overlapping with existing obstacles
                overlapping = False
                for (existing_ob, existing_size) in zip(self.Ob, self.Ob_size):
                    dist = np.linalg.norm(np.array([ob_x, ob_y]) - np.array([existing_ob[0], existing_ob[1]]))
                    if dist < (ob_size + existing_size) / 2:
                        overlapping = True
                        break

                if not overlapping:
                    # Place the obstacle
                    self.Ob.append(np.array([ob_x, ob_y, ob_z], dtype=np.float32))
                    self.Ob_size.append(ob_size)
                    placed = True
                attempts += 1

            if not placed:
                print(f"Warning: Could not place obstacle after {max_attempts} attempts.")

    def _spawn_agent(self):
        # Spawn agent at random location not overlapping with obstacles or too close to exits
        while True:
            x = np.random.uniform(self.agent_size, self.env_size - self.agent_size)
            y = np.random.uniform(self.agent_size, self.env_size - self.agent_size)
            pos = np.array([x, y, 0.5], dtype=np.float32)

            # Check distance from exits
            too_close_to_exit = False
            for ex in self.Exits:
                dist = np.linalg.norm(pos[:2] - ex[:2])
                if dist < (self.dis_lim + self.agent_size + self.door_size / 2):
                    too_close_to_exit = True
                    break
            if too_close_to_exit:
                continue

            # Check distance from obstacles
            overlaps_obstacle = False
            for (ob_pos, ob_size) in zip(self.Ob, self.Ob_size):
                dist = np.linalg.norm(pos[:2] - ob_pos[:2])
                if dist < (self.agent_size + ob_size / 2):
                    overlaps_obstacle = True
                    break
            if not overlaps_obstacle:
                break

        vx, vy = 0.0, 0.0
        return Particle(0, x, y, 0.5, vx, vy, 0.0)

    def _overlaps_obstacle(self, pos):
        for (o, sz) in zip(self.Ob, self.Ob_size):
            dist = np.linalg.norm(pos - o)
            if dist < (self.agent_size + sz) * 0.5:
                return True
        return False

    def _apply_environment_forces(self):
        # Wall repulsion
        # Distance to boundaries:
        # left: position[0], right: L[0,1]-pos[0]
        # bottom: position[1], top: L[1,1]-pos[1]
        p = self.agent.position
        px, py = p[0], p[1]

        # Wall force in x
        left_dist = px - self.L[0, 0]
        right_dist = self.L[0, 1] - px
        if left_dist < self.agent_size:
            f = self.f_wall_lim * np.exp((self.agent_size - left_dist) / 0.08)
            self.agent.acc[0] += f / self.agent.mass
        if right_dist < self.agent_size:
            f = self.f_wall_lim * np.exp((self.agent_size - right_dist) / 0.08)
            self.agent.acc[0] -= f / self.agent.mass

        # Wall force in y
        bottom_dist = py - self.L[1, 0]
        top_dist = self.L[1, 1] - py
        if bottom_dist < self.agent_size:
            f = self.f_wall_lim * np.exp((self.agent_size - bottom_dist) / 0.08)
            self.agent.acc[1] += f / self.agent.mass
        if top_dist < self.agent_size:
            f = self.f_wall_lim * np.exp((self.agent_size - top_dist) / 0.08)
            self.agent.acc[1] -= f / self.agent.mass

        # Obstacles repulsion
        for (o, sz) in zip(self.Ob, self.Ob_size):
            dr = p - o
            dist = np.linalg.norm(dr)
            dis_eq = (self.agent_size + sz) / 2
            if dist < dis_eq:
                if dist == 0:
                    # Prevent division by zero
                    direction = np.random.uniform(-1, 1, size=2)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = dr[:2] / dist
                f = self.f_collision_lim * np.exp((dis_eq - dist) / 0.08)
                self.agent.acc[:2] += (f * direction) / self.agent.mass

        # Friction force
        friction = -self.agent.mass / self.relaxation_time * self.agent.velocity[:2]
        self.agent.acc[:2] += friction / self.agent.mass

    def _apply_action(self, action):
        # Apply desired action
        act = self.base_actions[action]
        # Add driving force toward desired velocity direction
        desired = self.desire_velocity * act[:2]
        self.agent.acc[:2] += (1 / self.relaxation_time) * desired / self.agent.mass

    def _integrate(self):
        # Leapfrog integration
        self.agent.leapfrog(self.delta_t, 1)
        self.agent.leapfrog(self.delta_t, 0)

    def _check_done_and_reward(self):
        # Check if agent reached exit
        done = False
        reward = -0.1  # step cost
        for e in self.Exits:
            dist = np.linalg.norm(self.agent.position[:2] - e[:2])
            if dist < self.dis_lim:
                done = True
                reward = 0.0
                break
        return done, reward

    def _get_observation(self):
        px, py = self.agent.position[0], self.agent.position[1]
        vx, vy = self.agent.velocity[0], self.agent.velocity[1]

        # Closest exit
        closest_exit = None
        min_dist = float('inf')
        for e in self.Exits:
            dist = np.linalg.norm(self.agent.position[:2] - e[:2])
            if dist < min_dist:
                min_dist = dist
                closest_exit = e
        if closest_exit is None:
            # Should not happen as we have at least one exit
            closest_exit = np.array([self.env_size / 2, self.env_size / 2, 0.5], dtype=np.float32)
        rel_exit_x = closest_exit[0] - px
        rel_exit_y = closest_exit[1] - py

        # Obstacle distances in 8 directions
        directions = self.base_actions[:, :2] / self.action_force  # normalized directions
        obstacle_distances = []
        max_sensor_range = self.env_size  # could be less or more
        for dx, dy in directions:
            dist = max_sensor_range
            for (o, sz) in zip(self.Ob, self.Ob_size):
                rel = o[:2] - np.array([px, py])
                proj = rel[0] * dx + rel[1] * dy
                if proj > 0:
                    perp = np.linalg.norm(rel - proj * np.array([dx, dy]))
                    # If obstacle centerline is within obstacle radius in that direction
                    if perp < sz / 2 and proj < dist:
                        dist = proj
            obstacle_distances.append(dist)

        obs = np.array([px, py, vx, vy, rel_exit_x, rel_exit_y] + obstacle_distances, dtype=np.float32)
        return obs
