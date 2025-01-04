import numpy as np
import os

########################################
# Global parameters
########################################

f_wall_lim = 100.0          # Magnitude of wall repulsion force
f_collision_lim = 100.0     # Magnitude of particle collision force
door_size = 1.0             # Size of each door
agent_size = 0.5            # Size of the agent (particle)
reward = -0.1               # Default step-by-step reward
end_reward = 0              # Reward when exiting

offset = np.array([0.5, 0.5])           # Offset for certain normalizations (not strictly used here)
dis_lim = (agent_size + door_size)/2    # Distance from exit at which agent is considered exited
action_force = 1.0                      # Unit action force
desire_velocity = 2.0                   # Desired velocity
relaxation_time = 0.5                   # Relaxation time
delta_t = 0.1                           # Integration time step
cfg_save_step = 5                       # Interval for saving .Cfg file (not strictly essential here)

# Helper arrays for neighbor-cell computations
cell_list = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [0, 0, -1],
    [1, 0, -1],
    [1, 1, -1],
    [0, 1, -1],
    [-1, 1, -1],
    [-1, 0, -1],
    [-1, -1, -1],
    [0, -1, -1],
    [1, -1, -1]
], dtype=int)

neighbor_list = np.array([
    [-1, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [-1, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0]
], dtype=int)


########################################
# Particle (Agent) class
########################################
class Particle:
    def __init__(self, ID, x, y, z, vx, vy, vz, mass=80.0, type_=1):
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.acc = np.array([0., 0., 0.])
        self.mass = mass
        self.type = type_
        self.ID = ID

    def leapfrog(self, dt, stage):
        """
        Leapfrog integration:
        stage=0 => half-step velocity + full-step position
        stage=1 => another half-step velocity
        """
        if stage == 0:
            self.velocity += dt/2 * self.acc
            self.position += dt * self.velocity
        else:
            self.velocity += dt/2 * self.acc

    def sacle_velocity(self, value=1.0):
        # Possibly intended as 'scale_velocity'
        speed = np.linalg.norm(self.velocity)
        if speed != 0:
            self.velocity = (self.velocity / speed) * value


########################################
# Cell class
########################################
class Cell:
    def __init__(self, ID, idx, idy, idz, d_cells, L, n_cells):
        self.Particles = []
        self.Neighbors = []
        self.ID_number = ID
        self.ID_index = np.array([idx, idy, idz])

        # Each cell's boundary
        self.L = np.zeros_like(L)
        self.L[:, 0] = L[:, 0] + self.ID_index * d_cells
        self.L[:, 1] = self.L[:, 0] + d_cells

        self.n_cells = n_cells
        self.find_neighbors()

    def add(self, particle):
        self.Particles.append(particle)

    def find_neighbors(self):
        idx = self.ID_index + cell_list
        valid = (idx < self.n_cells) & (idx >= 0)
        idx = idx[np.all(valid, axis=1)]
        for n in range(len(idx)):
            i = idx[n, 0]
            j = idx[n, 1]
            k = idx[n, 2]
            N = k * (self.n_cells[0] * self.n_cells[1]) + j * self.n_cells[0] + i
            self.Neighbors.append(N)


########################################
# Cell_Space (the environment)
########################################
class Cell_Space:
    def __init__(
        self,
        xmin=0.0, xmax=10.0,
        ymin=0.0, ymax=10.0,
        zmin=0.0, zmax=2.0,
        rcut=1.5,
        dt=0.1,
        Number=1,          # Number of agents
        numExits=1,        # Number of exits
        numObs=2           # Number of obstacles
    ):
        """
        A single-agent environment in 2D with a fixed z dimension.
        We'll hold exactly 'numExits' exits and 'numObs' obstacles in the domain.
        """
        self.dt = dt
        self.Number = Number      # Current number of agents (should remain 1 unless you want more)
        self.Total = Number       # Total number of agents
        self.T = 0.0             # Temperature (not strictly needed for RL)
        self.numExits = numExits
        self.numObs = numObs

        # System bounding box
        self.L = np.array([
            [xmin, xmax],
            [ymin, ymax],
            [zmin, zmax]
        ], dtype=np.float32)

        self.rcut = rcut
        # Number of cells in each dimension
        self.n_cells = (np.array([
            (xmax - xmin),
            (ymax - ymin),
            (zmax - zmin)
        ]) / rcut).astype(int)
        self.d_cells = (self.L[:, 1] - self.L[:, 0]) / self.n_cells

        # We'll store actual exit/obstacle positions (3D)
        self.Exit = []
        self.Ob = []
        self.Ob_size = []

        # Build the cell grid
        self.Cells = []
        self.initialize_cells()

        # We'll place our agent(s)
        self.initialize_particles()

        # 8 discrete actions (N, NW, W, SW, S, SE, E, NE)
        diag = np.sqrt(2) / 2
        self.action = np.array([
            [0, 1, 0],
            [-diag, diag, 0],
            [-1, 0, 0],
            [-diag, -diag, 0],
            [0, -1, 0],
            [diag, -diag, 0],
            [1, 0, 0],
            [diag, diag, 0]
        ], dtype=np.float32)
        self.action *= action_force

        # Rewards
        self.reward = reward
        self.end_reward = end_reward

    def initialize_cells(self):
        nx, ny, nz = self.n_cells
        np_2d = nx * ny
        n_total = np_2d * nz
        for n in range(n_total):
            i = n % np_2d % nx
            j = n % np_2d // nx
            k = n // np_2d
            self.Cells.append(Cell(n, i, j, k, self.d_cells, self.L, self.n_cells))

    def initialize_particles(self, file=None):
        """
        Initialize one or more agents. We'll do it randomly in the 2D domain.
        """
        if file is None:
            P_list = []
            for i in range(self.Number):
                reselect = True
                while reselect:
                    # random position in [0.05..0.95] portion of the domain
                    pos_2d = (self.L[:2, 0] +
                              0.05*(self.L[:2, 1] - self.L[:2, 0]) +
                              np.random.rand(2)*(self.L[:2, 1] - self.L[:2, 0])*0.9)
                    z_mid = self.L[2, 0] + 0.5*(self.L[2, 1] - self.L[2, 0])
                    pos = np.array([pos_2d[0], pos_2d[1], z_mid])

                    # Check overlap with P_list
                    reselect = False
                    for old_pos in P_list:
                        dis = np.linalg.norm(pos - old_pos)
                        if dis < agent_size:
                            reselect = True
                            break

                    if not reselect:
                        # Accept position
                        P_list.append(pos)

                # small random velocity
                v = np.random.randn(3) * 0.01
                v[2] = 0.
                particle = Particle(i, pos[0], pos[1], pos[2], v[0], v[1], v[2])
                if i == 0:
                    self.agent = particle
                self.insert_particle(particle)
        else:
            # Optionally read from a file
            pass

    def randomize_exits_and_obstacles(self):
        """
        Create exactly self.numExits exits (on walls, no corners) and self.numObs obstacles in the interior.
        Clear out old data each time it’s called.
        """
        self.Exit.clear()
        self.Ob.clear()
        self.Ob_size.clear()

        xmin, xmax = self.L[0, 0], self.L[0, 1]
        ymin, ymax = self.L[1, 0], self.L[1, 1]
        zmin, zmax = self.L[2, 0], self.L[2, 1]
        z_fixed = (zmin + zmax)/2

        # 1) Place self.numExits on walls (no corners)
        for _ in range(self.numExits):
            placed_exit = False
            while not placed_exit:
                wall_choice = np.random.randint(0, 4)
                if wall_choice == 0:
                    # left wall
                    x = xmin
                    y = np.random.uniform(ymin, ymax)
                    if abs(y - ymin) < 1e-3 or abs(ymax - y) < 1e-3:
                        continue
                elif wall_choice == 1:
                    # right wall
                    x = xmax
                    y = np.random.uniform(ymin, ymax)
                    if abs(y - ymin) < 1e-3 or abs(ymax - y) < 1e-3:
                        continue
                elif wall_choice == 2:
                    # bottom wall
                    y = ymin
                    x = np.random.uniform(xmin, xmax)
                    if abs(x - xmin) < 1e-3 or abs(xmax - x) < 1e-3:
                        continue
                else:
                    # top wall
                    y = ymax
                    x = np.random.uniform(xmin, xmax)
                    if abs(x - xmin) < 1e-3 or abs(xmax - x) < 1e-3:
                        continue

                new_exit_pos = np.array([x, y, z_fixed])
                self.Exit.append(new_exit_pos)
                placed_exit = True

        # 2) Place self.numObs obstacles in the interior
        #    We'll keep a default obstacle size, e.g. 2.0
        default_ob_size = 2.0

        agent_pos = self.agent.position

        for _ in range(self.numObs):
            placed_ob = False
            while not placed_ob:
                ox = np.random.uniform(xmin+0.5, xmax-0.5)
                oy = np.random.uniform(ymin+0.5, ymax-0.5)
                cand_pos = np.array([ox, oy, z_fixed])

                # check distance from agent
                if np.linalg.norm(cand_pos - agent_pos) < 1.0:
                    continue

                # check distance from existing obstacles
                candidate_radius = default_ob_size / 2
                no_overlap = True
                for idx_ob, sublist in enumerate(self.Ob):
                    for old_ob_pos in sublist:
                        dist_obs = np.linalg.norm(cand_pos - old_ob_pos)
                        exist_rad = self.Ob_size[idx_ob]/2
                        if dist_obs < (candidate_radius + exist_rad):
                            no_overlap = False
                            break
                    if not no_overlap:
                        break

                if not no_overlap:
                    continue

                # check distance from exits
                too_close_to_exit = False
                for ex_pos in self.Exit:
                    if np.linalg.norm(cand_pos - ex_pos) < 1.0:
                        too_close_to_exit = True
                        break
                if too_close_to_exit:
                    continue

                # If all good, place it
                self.Ob.append([cand_pos])   # one sublist per obstacle
                self.Ob_size.append(default_ob_size)
                placed_ob = True

    def insert_particle(self, particle):
        """
        Insert a particle into the proper cell based on position.
        """
        index = (particle.position - self.L[:, 0])/self.d_cells
        if (index < 0).any() or (index >= self.n_cells).any():
            print("Particle out of boundary!")
            return
        index = index.astype(int)
        N = index[2]*(self.n_cells[0]*self.n_cells[1]) + index[1]*self.n_cells[0] + index[0]
        self.Cells[N].add(particle)

    def Zero_acc(self):
        for c in self.Cells:
            for p in c.Particles:
                p.acc[:] = 0.0

    def region_confine(self):
        """
        Wall/obstacle repulsion + friction for all particles.
        """
        for c in self.Cells:
            for p in c.Particles:
                # 1) Wall force
                dis_abs = np.abs(p.position[:, np.newaxis] - self.L)
                # shape => (3,2): we do exponential repulsion if within agent_size
                # then sum across the two columns (lower & upper boundary)
                f = np.where(
                    dis_abs < agent_size,
                    f_wall_lim * np.exp((agent_size - dis_abs) / 0.08) * dis_abs,
                    0.0
                )
                f[:, 1] = -f[:, 1]  # flip sign for upper boundary
                f_net = f.sum(axis=1)
                p.acc += f_net / p.mass

                # 2) Obstacles
                for idx_ob, sublist in enumerate(self.Ob):
                    for obst_pos in sublist:
                        dr = p.position - obst_pos
                        dist = np.linalg.norm(dr)
                        dist_eq = (agent_size + self.Ob_size[idx_ob]) / 2
                        if dist < dist_eq:
                            f_mag = f_collision_lim * np.exp((dist_eq - dist)/0.08)
                            if dist > 1e-12:
                                f_vec = f_mag * dr/dist
                            else:
                                f_vec = 0.0
                            p.acc += f_vec / p.mass

                # 3) Friction
                friction = -p.mass/relaxation_time * p.velocity
                p.acc += friction/p.mass

    def loop_cells(self):
        """
        Intra-cell collisions among agents in the same cell.
        """
        for c in self.Cells:
            l = len(c.Particles)
            for i in range(l):
                for j in range(i+1, l):
                    p1 = c.Particles[i]
                    p2 = c.Particles[j]
                    dr = p1.position - p2.position
                    dist = np.linalg.norm(dr)
                    if dist < agent_size:
                        f_mag = f_collision_lim * np.exp((agent_size - dist)/0.08)
                        if dist > 1e-12:
                            f_vec = f_mag * dr/dist
                        else:
                            f_vec = 0.0
                        p1.acc += f_vec/p1.mass
                        p2.acc -= f_vec/p2.mass

    def loop_neighbors(self):
        """
        Inter-cell collisions among neighbor cells.
        """
        for c in self.Cells:
            for n in c.Neighbors:
                for p1 in c.Particles:
                    for p2 in self.Cells[n].Particles:
                        dr = p1.position - p2.position
                        dist = np.linalg.norm(dr)
                        if dist < agent_size:
                            f_mag = f_collision_lim * np.exp((agent_size - dist)/0.08)
                            if dist > 1e-12:
                                f_vec = f_mag * dr/dist
                            else:
                                f_vec = 0.0
                            p1.acc += f_vec/p1.mass
                            p2.acc -= f_vec/p2.mass

    def Integration(self, stage):
        self.T = 0.0
        for c in self.Cells:
            for p in c.Particles:
                p.leapfrog(dt=self.dt, stage=stage)
                self.T += 0.5 * p.mass * np.sum(p.velocity**2)
        self.T /= self.Number

    def move_particles(self):
        """
        Re-classify particles into cells if they moved out of the old cell.
        """
        for c in self.Cells:
            i = 0
            while i < len(c.Particles):
                p = c.Particles[i]
                position = p.position
                inside = (position >= c.L[:, 0]) & (position < c.L[:, 1])
                if inside.all():
                    i += 1
                else:
                    popped = c.Particles.pop(i)
                    self.insert_particle(popped)

    def get_state(self):
        """
        Return a 6D state tuple:
        ( agent_x, agent_y,
        agent_vx, agent_vy,
        distance_to_nearest_exit,
        distance_to_nearest_obstacle )
        
        If self.numObs == 0, distance_to_nearest_obstacle = np.inf.
        """

        # Agent position & velocity
        ax = self.agent.position[0]
        ay = self.agent.position[1]
        avx = self.agent.velocity[0]
        avy = self.agent.velocity[1]

        # 1) Distance to nearest exit
        dist_exit = np.inf
        for epos in self.Exit:  # each epos is a 3D position [x, y, z]
            dist_tmp = np.linalg.norm(epos - self.agent.position)
            if dist_tmp < dist_exit:
                dist_exit = dist_tmp

        # 2) Distance to nearest obstacle
        # If numObs==0, we set dist_obs = np.inf directly
        if self.numObs == 0:
            dist_obs = -1
        else:
            dist_obs = np.inf
            for sublist in self.Ob:
                # Each obstacle's sublist typically has one position, e.g. sublist[0]
                obs_pos = sublist[0]
                dist_tmp = np.linalg.norm(obs_pos - self.agent.position)
                if dist_tmp < dist_obs:
                    dist_obs = dist_tmp

        return (ax, ay, avx, avy, dist_exit, dist_obs)


    def reset(self):
        """
        Clear old particles, randomize exits/obstacles, and re-initialize.
        Return the new 6D state = (x, y, vx, vy, distExit, distObs).
        """
        # 1) Clear cells
        for cell in self.Cells:
            cell.Particles.clear()
        self.Number = self.Total

        # 2) Re-initialize agent(s)
        self.initialize_particles()

        # 3) Randomize exits/obstacles
        self.randomize_exits_and_obstacles()

        # 4) Return the new 6D state
        return self.get_state()


    def step(self, action):
        """
        Single environment step. 
        Returns (next_state, reward, done), where next_state is the 6D tuple:
            (x_agent, y_agent, vx_agent, vy_agent, dist_to_nearest_exit, dist_to_nearest_obstacle).
        """
        done = False
        r = self.reward

        # zero out accelerations
        self.Zero_acc()

        # region confine & collisions
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()

        # apply action to agent (ID=0)
        self.agent.acc += (1/relaxation_time) * desire_velocity * self.action[action]

        # integrate
        self.Integration(stage=1)
        self.Integration(stage=0)

        # reassign cells if needed
        self.move_particles()

        # check exit
        agent_pos = self.agent.position
        for epos in self.Exit:
            dist = np.linalg.norm(agent_pos - epos)
            if dist < dis_lim:
                done = True
                r = self.end_reward
                break

        # Return the new 6D state
        next_state = self.get_state()
        return next_state, r, done


    def choose_random_action(self):
        return np.random.choice(len(self.action))

    def save_output(self, file):
        """
        Save system state to a CFG file for visualization or debugging.
        (Optional usage)
        """
        N_obs = len(self.Ob)  # 1 point per obstacle sublist
        N_obs_total = 0
        for ob_sublist in self.Ob:
            N_obs_total += len(ob_sublist)  # typically 1 each

        with open(file, 'w+') as f:
            HX = self.L[0, 1] - self.L[0, 0]
            HY = self.L[1, 1] - self.L[1, 0]
            HZ = self.L[2, 1] - self.L[2, 0]

            f.write(f"Number of particles = {self.Number + len(self.Exit) + N_obs_total}\n")
            f.write(f"""A = 1.0 Angstrom (basic length-scale)
H0(1,1) = {HX} A
H0(1,2) = 0 A
H0(1,3) = 0 A
H0(2,1) = 0 A
H0(2,2) = {HY} A
H0(2,3) = 0 A
H0(3,1) = 0 A
H0(3,2) = 0 A
H0(3,3) = {HZ} A
entry_count = 7
auxiliary[0] = ID [reduced unit]
""")

            f.write("10.000000\nAt\n")
            # Exits (treated as type -1 or so)
            for epos in self.Exit:
                ex, ey, ez = self.Normalization(epos)
                f.write(f"{ex} {ey} {ez} 0 0 0 -1\n")

            # Obstacles
            # We'll treat them as type 1 (C), 2 (Si), etc. if needed
            # For simplicity, let's just label them 'C' one by one
            for idx_ob, sublist in enumerate(self.Ob):
                if idx_ob == 0:
                    f.write("1.000000\nC\n")
                elif idx_ob == 1:
                    f.write("1.000000\nSi\n")
                else:
                    f.write("1.000000\nC\n")

                for opos in sublist:
                    ox, oy, oz = self.Normalization(opos)
                    f.write(f"{ox} {oy} {oz} 0 0 0 -1\n")

            # Agent(s)
            if self.Number > 0:
                f.write("1.000000\nBr\n")  # label for agent
                for c_ in self.Cells:
                    for p in c_.Particles:
                        x, y, z = self.Normalization(p.position)
                        f.write(f"{x} {y} {z} {p.velocity[0]} {p.velocity[1]} {p.velocity[2]} {p.ID}\n")

    def Normalization(self, position):
        """
        Maps from real coords to [0,1].
        """
        return (position - self.L[:, 0])/(self.L[:, 1] - self.L[:, 0])

    def Normalization_XY(self, position_2d):
        """
        Maps XY to [0,1], minus offset => [-0.5..0.5], if needed.
        Not used in this environment, but left for completeness.
        """
        return (position_2d - self.L[:2, 0])/(self.L[:2, 1] - self.L[:2, 0]) - offset

    def step_optimal_single_particle(self):

        # This function is intended for the single-particle scenario only.
        # It replicates the exact logic of step_optimal, but:
        # 1. Only handles one particle (self.agent).
        # 2. Returns (next_state, reward, done).
        #
        # Behavior:
        # - Perform the same environment updates as step_optimal.
        # - Apply optimal action for the single particle to move toward the nearest exit.
        # - If the particle reaches the exit (done = True), return end_reward.
        # - If not, return reward.
        #

        done = False

        # If no particles, done immediately
        if self.Number == 0:
            done = True
            # No movement happened, return current state, end_reward
            next_state = (self.agent.position[0], self.agent.position[1],
                        self.agent.velocity[0], self.agent.velocity[1])
            return next_state, self.end_reward, done

        # Zero out accelerations
        self.Zero_acc()

        # Apply region confinement and neighbor interactions
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()

        # Compute optimal direction for the single particle (self.agent)
        p = self.agent
        dr = np.inf
        dr_unit = None
        for e in self.Exit:
            dr_tmp = np.sqrt(np.sum((e - p.position)**2))
            if dr_tmp < dr:
                dr = dr_tmp
                dr_unit = (e - p.position) / dr_tmp

        # Find the best action that aligns with the direction to the nearest exit
        costheta = -np.inf
        for action in self.action:
            costheta_tmp = np.matmul(action, dr_unit)
            if costheta_tmp > costheta:
                costheta = costheta_tmp
                dr_action = action

        # Update acceleration according to the best action
        p.acc += 1/relaxation_time * desire_velocity * dr_action

        # Integrate the equations of motion
        self.Integration(1)
        self.Integration(0)

        # Move particles between cells if needed
        self.move_particles()

        # Check if the particle has reached the exit
        # If reached, remove it and set done = True
        in_exit = False
        for e in self.Exit:
            dis = p.position - e
            dis = np.sqrt(np.sum(dis**2))
            if dis < (agent_size + door_size)/2:
                # Particle reached the exit
                self.Number -= 1
                in_exit = True
                break

        if in_exit:
            done = True
            # If done, reward = end_reward
            reward = self.end_reward
        else:
            # Not done, reward = self.reward
            reward = self.reward

        # Prepare next_state
        next_state = (p.position[0], p.position[1],
                    p.velocity[0], p.velocity[1])

        return next_state, reward, done
    
if __name__ == '__main__':
    # Example usage / simple test
    env = Cell_Space(
        xmin=0, xmax=20,
        ymin=0, ymax=20,
        zmin=0, zmax=2,
        dt=0.1,
        Number=1,
        numExits=2,
        numObs=3
    )
    initial_state = env.reset()
    print("Initial State:", initial_state)

    done = False
    step_count = 0
    while not done and step_count < 50:
        action = env.choose_random_action()
        next_state, reward, done = env.step(action)
        print(f"Step {step_count} -> Action:{action}, Reward:{reward}, Done:{done}")
        step_count += 1

    if not os.path.isdir("./test_cfg"):
        os.mkdir("./test_cfg")
    env.save_output("./test_cfg/s.final")
