import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class RoomNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 room_size=10.0, 
                 agent_size=0.5, 
                 door_size=1.0, 
                 num_exits=1, 
                 num_obstacles=2, 
                 obstacle_size=1.0, 
                 mass=80.0, 
                 desire_velocity=2.0,
                 relaxation_time=0.5,
                 delta_t=0.1,
                 max_steps=1000,
                 seed=None):
        """
        Initialize the environment.

        Parameters:
        - room_size: float, size of the room in meters (square room_size x room_size).
        - agent_size: diameter of the agent in meters.
        - door_size: size of each exit door in meters.
        - num_exits: number of exits.
        - num_obstacles: number of obstacles.
        - obstacle_size: diameter of each obstacle (assumed uniform here for simplicity).
        - mass: mass of the agent.
        - desire_velocity: desired velocity magnitude (m/s).
        - relaxation_time: relaxation time for social force model.
        - delta_t: time step for integration.
        - max_steps: maximum steps before episode termination.
        - seed: random seed.
        """

        super(RoomNavigationEnv, self).__init__()
        if seed is not None:
            np.random.seed(seed)

        self.room_size = room_size
        self.agent_size = agent_size
        self.door_size = door_size
        self.num_exits = num_exits
        self.num_obstacles = num_obstacles
        self.obstacle_size = obstacle_size
        self.mass = mass
        self.desire_velocity = desire_velocity
        self.relaxation_time = relaxation_time
        self.delta_t = delta_t
        self.max_steps = max_steps

        # From the given excerpt:
        # Distance to consider exit reached:
        self.dis_lim = (self.agent_size + self.door_size) / 2.0  # 0.75 m typically

        # Action space: 8 discrete actions representing the 8 directions
        self.action_space = gym.spaces.Discrete(8)

        # We must define observation space. The state is:
        # (agent_x, agent_y, agent_vx, agent_vy, dist_to_each_exit..., dist_to_each_obstacle...)
        # Positions and distances normalized by 10. Velocities also normalized by 10.
        # Maximum normalized distances and positions ≤ 1.
        # Let's assume a max dimension of 1 for all normalized observations.
        # Number of obs = 4 + num_exits + num_obstacles
        self.obs_dim = 4 + self.num_exits + self.num_obstacles
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # Forces parameters as from excerpt:
        self.A = 100.0
        self.B = 0.08
        self.k = 8.0e4

        # Action directions (8 discrete directions)
        diag = np.sqrt(2)/2
        self.action_directions = np.array([
            [0,    1,    0],
            [-diag, diag, 0],
            [-1,   0,    0],
            [-diag,-diag, 0],
            [0,   -1,    0],
            [diag,-diag,  0],
            [1,    0,    0],
            [diag, diag,  0]
        ], dtype=np.float32)

        # Scale action force:
        self.action_force = 1.0  # unit action force
        # We apply: self-driven force = (m/τ)*v_desire*(f_action)
        # But the action direction is a unit vector, we just store these directions.

        # For rendering
        self.fig = None
        self.ax = None

        # Initialize environment variables:
        self.reset()

    def _randomize_positions(self):
        """
        Randomize positions of the agent, exits, and obstacles with the given constraints.
        """
        # Random agent initial position:
        # Place agent somewhere not too close to walls and not overlapping with obstacles.
        # Start by placing agent in the center or random inside region?
        # We'll allow the agent anywhere in [agent_size, room_size - agent_size], 
        # to avoid initially spawning outside the room.
        # We'll first place obstacles and exits, then place agent ensuring no overlap.

        # Randomize exits on walls:
        # Exits are on the perimeter. 
        # Let's pick which walls for each exit:
        self.exits = []
        for _ in range(self.num_exits):
            # Choose a wall: 0=top, 1=bottom, 2=left, 3=right
            wall = np.random.choice([0,1,2,3])
            if wall == 0:
                # top wall, y = room_size
                # exit center must be at least door_size/2 from corners: 
                # x in [door_size/2, room_size - door_size/2]
                x = np.random.uniform(self.door_size/2, self.room_size - self.door_size/2)
                y = self.room_size
            elif wall == 1:
                # bottom wall, y = 0
                x = np.random.uniform(self.door_size/2, self.room_size - self.door_size/2)
                y = 0.0
            elif wall == 2:
                # left wall, x = 0
                x = 0.0
                y = np.random.uniform(self.door_size/2, self.room_size - self.door_size/2)
            else:
                # right wall, x = room_size
                x = self.room_size
                y = np.random.uniform(self.door_size/2, self.room_size - self.door_size/2)
            self.exits.append(np.array([x,y,0.5*self.room_size]))  # z=5.0 for consistency (2D)
        
        # Randomize obstacles:
        # Obstacles must be at least agent_size away from walls and from each other.
        # We'll place them using a rejection sampling method.
        self.obstacles = []
        max_tries = 1000
        for _ in range(self.num_obstacles):
            placed = False
            for __ in range(max_tries):
                ox = np.random.uniform(self.agent_size, self.room_size - self.agent_size)
                oy = np.random.uniform(self.agent_size, self.room_size - self.agent_size)
                # Check no overlap with other obstacles:
                # We assume obstacles have a uniform size self.obstacle_size.
                # Min distance between obstacles = (obstacle_size)
                # Actually, no overlap means distance between centers >= obstacle_size.
                # Also ensure no overlap with exits (not strictly required, but let's ensure no confusion)
                # Actually, the user did not forbid overlap with exits, only no overlap among obstacles and 
                # not overlapping with the agent. We'll only ensure obstacle-obstacle no overlap.
                valid = True
                for o in self.obstacles:
                    dist = np.sqrt((ox-o[0])**2 + (oy-o[1])**2)
                    if dist < (self.obstacle_size):
                        valid = False
                        break
                if valid:
                    self.obstacles.append(np.array([ox, oy, 0.5*self.room_size]))
                    placed = True
                    break
            if not placed:
                raise RuntimeError("Could not place obstacles without overlap. Try fewer obstacles or larger room.")
        
        # Finally, place the agent:
        # Agent must not overlap with obstacles or exits.
        # Overlap with obstacles means distance < (agent_size+obstacle_size)/2
        # Overlap with exits is not explicitly stated as forbidden. 
        # But let's ensure no immediate overlap with obstacles.
        # We'll place the agent similarly using rejection sampling.
        for __ in range(max_tries):
            ax_ = np.random.uniform(self.agent_size, self.room_size - self.agent_size)
            ay_ = np.random.uniform(self.agent_size, self.room_size - self.agent_size)
            # Check no overlap with obstacles:
            agent_valid = True
            for o in self.obstacles:
                dist = np.sqrt((ax_-o[0])**2 + (ay_-o[1])**2)
                if dist < (self.agent_size + self.obstacle_size)/2:
                    agent_valid = False
                    break
            if agent_valid:
                self.agent_position = np.array([ax_, ay_, 0.5*self.room_size])
                break
        else:
            raise RuntimeError("Could not place agent without overlapping obstacles.")

        # Agent initial velocity = 0
        self.agent_velocity = np.array([0.0, 0.0, 0.0])

    def reset(self):
        """
        Reset the environment: randomize exits and obstacles, place agent, reset velocity.
        Returns initial observation.
        """
        self.steps = 0
        self._randomize_positions()

        # Compute initial observation
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        # Observation:
        # agent_x, agent_y normalized to [0,1] by dividing by room_size
        # agent_vx, agent_vy normalized by dividing by 10
        # distances to exits and obstacles also normalized by dividing by 10 (since room=10m)
        ax = self.agent_position[0]/self.room_size
        ay = self.agent_position[1]/self.room_size
        avx = (self.agent_velocity[0]/10.0)
        avy = (self.agent_velocity[1]/10.0)

        # Distances to exits:
        exit_dists = []
        for e in self.exits:
            dx = self.agent_position[0] - e[0]
            dy = self.agent_position[1] - e[1]
            dist = np.sqrt(dx*dx+dy*dy)/self.room_size
            exit_dists.append(dist)

        # Distances to obstacles:
        obstacle_dists = []
        for o in self.obstacles:
            dx = self.agent_position[0] - o[0]
            dy = self.agent_position[1] - o[1]
            dist = np.sqrt(dx*dx+dy*dy)/self.room_size
            obstacle_dists.append(dist)

        obs = np.array([ax, ay, avx, avy] + exit_dists + obstacle_dists, dtype=np.float32)
        return obs

    def step(self, action):
        """
        Execute one step of the simulation.
        """
        self.steps += 1
        # Apply physics according to social force model:
        # Forces: 
        # 1) Self-driven force: (m/τ)*v_desire*f_action
        # 2) Agent-wall and agent-obstacle repulsion (avoidance)
        # 3) Viscous damping: - (m/τ)*v

        # Zero acceleration:
        acc = np.zeros(3)

        # Self-driven force:
        f_action = self.action_directions[action]*self.action_force
        f_self = (self.mass/self.relaxation_time)*self.desire_velocity*f_action
        # Viscous damping:
        f_vis = -(self.mass/self.relaxation_time)*self.agent_velocity

        acc += (f_self + f_vis)/self.mass

        # Avoidance forces from obstacles and walls:
        # For obstacles:
        for o in self.obstacles:
            acc += self._compute_obstacle_force(self.agent_position, self.agent_size, o, self.obstacle_size)/self.mass

        # For walls:
        acc += self._compute_wall_force(self.agent_position, self.agent_size)/self.mass

        # Update velocity and position using leapfrog integration:
        # v(t+Δt/2) = v(t) + a(t)*Δt/2
        # r(t+Δt) = r(t) + v(t+Δt/2)*Δt
        # v(t+Δt) = v(t+Δt/2) + a(t+Δt)*Δt/2
        # We have only one agent, so direct integration:
        half_dt = self.delta_t*0.5
        # first half velocity update
        self.agent_velocity += acc*half_dt
        # position update
        self.agent_position += self.agent_velocity*self.delta_t
        # recompute acceleration after move (acc might change slightly due to new position)
        # Strictly, we'd recompute acceleration. But we will accept the same acc since no other agents:
        # Actually, to follow leapfrog strictly:
        # Recompute acceleration at new position:
        # Zero acceleration again
        acc_new = np.zeros(3)
        # Recompute forces:
        f_self = (self.mass/self.relaxation_time)*self.desire_velocity*f_action
        f_vis = -(self.mass/self.relaxation_time)*self.agent_velocity
        acc_new += (f_self+f_vis)/self.mass

        for o in self.obstacles:
            acc_new += self._compute_obstacle_force(self.agent_position, self.agent_size, o, self.obstacle_size)/self.mass
        acc_new += self._compute_wall_force(self.agent_position, self.agent_size)/self.mass

        # second half velocity update
        self.agent_velocity += acc_new*half_dt

        # Check termination:
        done = False
        reward = -0.1
        # Check if agent reached exit:
        for e in self.exits:
            dist = np.sqrt((self.agent_position[0]-e[0])**2 + (self.agent_position[1]-e[1])**2)
            if dist < self.dis_lim:
                done = True
                reward = 0.0
                break

        if self.steps >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    def _compute_obstacle_force(self, agent_pos, agent_size, ob_pos, ob_size):
        """
        Compute force from a single obstacle on the agent using the social force model described.
        Overall force: 
        F_iw = A * exp((d_iw-r_iw)/B)*r_hat_iw + k*g(r_iw-d_iw)*r_hat_iw + k*g(r_iw-d_iw)*(v_i*t_iw)*t_hat_iw
        For simplicity, we consider the obstacle as a fixed object with no velocity difference term (v_i*t_iw)
        If friction is considered minimal, we can still compute it. The original excerpt includes friction, 
        but since obstacle is static (v_obstacle=0), friction simplifies.

        We must be careful:
        - d_iw = agent_radius + obstacle_radius = agent_size/2 + ob_size/2
        - r_iw = actual distance between agent and obstacle center.
        """
        agent_r = agent_size/2.0
        ob_r = ob_size/2.0
        d_iw = agent_r + ob_r
        dr = agent_pos - ob_pos
        r_iw = np.sqrt(np.sum(dr**2))
        if r_iw == 0:
            # If by some numerical chance they overlap exactly
            r_hat_iw = np.array([1,0,0])
        else:
            r_hat_iw = dr/r_iw

        # avoidance force:
        F_avoid = self.A*np.exp((d_iw - r_iw)/self.B)*r_hat_iw
        # compression force:
        overlap = (r_iw - d_iw)
        # g(x) = xH(x), Heaviside function: if overlap>0 => compression
        if overlap < 0:
            # no compression if agent not inside obstacle
            F_compress = self.k*0.0*r_hat_iw
            # friction:
            F_friction = 0.0*r_hat_iw
        else:
            # If no overlap (r_iw > d_iw), no compression
            F_compress = 0.0*r_hat_iw
            F_friction = 0.0*r_hat_iw

        # Actually, from the formula: 
        # F_ij_compression = k*g(r_ij-d_ij)*r_hat_ij,
        # if we consider g(x)=xH(x), only positive overlap counts.
        # If r_iw < d_iw, then (r_iw - d_iw)<0 and no compression.
        # Wait, we must check carefully:
        # Overlap means agent and obstacle are closer than d_iw:
        # If r_iw < d_iw => (r_iw - d_iw)<0. This indicates compression.
        # Actually g(x) = xH(x), H(x)=1 if x>0 else 0.
        # If r_iw - d_iw < 0 => x<0 => no compression.
        # So we had it inverted. Let's correct:
        if r_iw < d_iw:
            # Now we have compression:
            overlap_val = (d_iw - r_iw) # positive
            F_compress = self.k*overlap_val*r_hat_iw
            # friction: tangential direction: we have only agent velocity v_i, obstacle velocity =0
            # t_ij is a unit vector perpendicular to r_hat_iw
            # choose a perpendicular direction:
            # Since we are in 2D, we can pick t_hat_iw as:
            t_hat_iw = np.array([-r_hat_iw[1], r_hat_iw[0], 0])
            # relative tangential velocity = (v_i - v_w)*t_hat_iw = v_i*t_hat_iw since v_w=0
            v_i = self.agent_velocity
            v_t = np.dot(v_i, t_hat_iw)
            F_friction = self.k*overlap_val*v_t*t_hat_iw
            return F_avoid + F_compress + F_friction
        else:
            # No compression
            return F_avoid

    def _compute_wall_force(self, agent_pos, agent_size):
        """
        Compute force from the walls. The room is [0, room_size]x[0, room_size].
        For walls, similar formula:
        F_iw = A*exp((d_iw-r_iw)/B)*r_hat_iw + ...
        d_iw = agent_radius = agent_size/2
        r_iw = distance to wall
        For each wall, find the shortest distance to it and direction.
        Walls: x=0, x=room_size, y=0, y=room_size.
        """
        agent_r = agent_size/2.0

        forces = np.zeros(3)

        # Check distance to left wall (x=0)
        dist_left = agent_pos[0] - 0.0
        if dist_left < agent_r:
            # close to left wall
            overlap_val = (agent_r - dist_left)
            r_iw = dist_left
            r_hat = np.array([1,0,0])
            F_avoid = self.A*np.exp((agent_r-r_iw)/self.B)*r_hat
            # If r_iw < d_iw means agent partially overlapping the wall => compression:
            # here d_iw = agent_r, if r_iw<agent_r => compression
            if r_iw < agent_r:
                F_compress = self.k*(agent_r-r_iw)*r_hat
                # friction same logic:
                t_hat = np.array([0,1,0]) # perpendicular to r_hat
                v_t = np.dot(self.agent_velocity, t_hat)
                F_friction = self.k*(agent_r-r_iw)*v_t*t_hat
                forces += F_avoid + F_compress + F_friction
            else:
                forces += F_avoid

        # Right wall (x=room_size)
        dist_right = self.room_size - agent_pos[0]
        if dist_right < agent_r:
            overlap_val = (agent_r - dist_right)
            r_iw = dist_right
            r_hat = np.array([-1,0,0])
            F_avoid = self.A*np.exp((agent_r-r_iw)/self.B)*r_hat
            if r_iw < agent_r:
                F_compress = self.k*(agent_r-r_iw)*r_hat
                t_hat = np.array([0,1,0])
                v_t = np.dot(self.agent_velocity, t_hat)
                F_friction = self.k*(agent_r-r_iw)*v_t*t_hat
                forces += F_avoid + F_compress + F_friction
            else:
                forces += F_avoid

        # Bottom wall (y=0)
        dist_bottom = agent_pos[1] - 0.0
        if dist_bottom < agent_r:
            r_iw = dist_bottom
            r_hat = np.array([0,1,0])
            F_avoid = self.A*np.exp((agent_r-r_iw)/self.B)*r_hat
            if r_iw < agent_r:
                F_compress = self.k*(agent_r-r_iw)*r_hat
                t_hat = np.array([1,0,0])
                v_t = np.dot(self.agent_velocity, t_hat)
                F_friction = self.k*(agent_r-r_iw)*v_t*t_hat
                forces += F_avoid + F_compress + F_friction
            else:
                forces += F_avoid

        # Top wall (y=room_size)
        dist_top = self.room_size - agent_pos[1]
        if dist_top < agent_r:
            r_iw = dist_top
            r_hat = np.array([0,-1,0])
            F_avoid = self.A*np.exp((agent_r-r_iw)/self.B)*r_hat
            if r_iw < agent_r:
                F_compress = self.k*(agent_r-r_iw)*r_hat
                t_hat = np.array([1,0,0])
                v_t = np.dot(self.agent_velocity, t_hat)
                F_friction = self.k*(agent_r-r_iw)*v_t*t_hat
                forces += F_avoid + F_compress + F_friction
            else:
                forces += F_avoid

        return forces

    def render(self, mode='human'):
        """
        Render the environment using matplotlib.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            plt.ion()
        
        self.ax.clear()
        # Draw room boundary
        self.ax.plot([0,self.room_size,self.room_size,0,0],
                     [0,0,self.room_size,self.room_size,0],
                     'k-')

        # Draw agent
        agent_circle = Circle((self.agent_position[0], self.agent_position[1]), self.agent_size/2, color='blue')
        self.ax.add_patch(agent_circle)

        # Draw exits
        # The door_size=1.0, we can represent them as small line segments or just markers
        for e in self.exits:
            # Exits are on walls. Let's draw a small line to represent the door
            if np.isclose(e[1],0.0): # bottom wall
                x_c = e[0]
                self.ax.plot([x_c - self.door_size/2, x_c + self.door_size/2],
                             [0,0], 'g-', linewidth=3)
            elif np.isclose(e[1], self.room_size): # top wall
                x_c = e[0]
                self.ax.plot([x_c - self.door_size/2, x_c + self.door_size/2],
                             [self.room_size, self.room_size], 'g-', linewidth=3)
            elif np.isclose(e[0], 0.0): # left wall
                y_c = e[1]
                self.ax.plot([0,0],
                             [y_c - self.door_size/2, y_c + self.door_size/2], 'g-', linewidth=3)
            elif np.isclose(e[0], self.room_size): # right wall
                y_c = e[1]
                self.ax.plot([self.room_size, self.room_size],
                             [y_c - self.door_size/2, y_c + self.door_size/2], 'g-', linewidth=3)
            # Mark center
            self.ax.plot(e[0], e[1], 'go')

        # Draw obstacles
        for o in self.obstacles:
            obstacle_circle = Circle((o[0], o[1]), self.obstacle_size/2, color='red')
            self.ax.add_patch(obstacle_circle)

        self.ax.set_xlim(-0.5, self.room_size+0.5)
        self.ax.set_ylim(-0.5, self.room_size+0.5)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title(f"Step: {self.steps}")
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
