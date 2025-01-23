import numpy as np
import os
import math  # For arctan2, etc.

########################################
# Global parameters
########################################

f_wall_lim = 100.0          # Magnitude of wall repulsion force
f_collision_lim = 100.0     # Magnitude of particle collision force
door_size = 1.0             # Size of each door
agent_size = 0.5            # Size of the agent (particle)
reward = -0.1               # Default step-by-step reward
near_end_reward = -0.2      # Reward when close to exit but not there yet
end_reward = 10             # Reward when exiting

offset = np.array([0.5, 0.5])           
dis_lim = (agent_size + door_size)/2    
action_force = 1.0                      
desire_velocity = 2.0                  
relaxation_time = 0.5                  
delta_t = 0.1                           
cfg_save_step = 5                       

# Occupancy grid resolution => 0.5 meter per cell
grid_cell_size = 0.5

# For neighbor cells in the environment's cell structure (separate concept from the occupancy grid)
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
        Number=1,          
        numExits=1,        
        numObs=2
    ):
        """
        A single-agent environment returning occupancy grid + agent (x, y, vx, vy).
        Occupancy grid uses 0.5 m/cell, labeled:
          0 = free space
          1 = exit
          2 = obstacle
        """
        self.dt = dt
        self.Number = Number
        self.Total = Number
        self.T = 0.0
        self.numExits = numExits
        self.numObs = numObs

        # Domain bounding box
        self.L = np.array([
            [xmin, xmax],
            [ymin, ymax],
            [zmin, zmax]
        ], dtype=np.float32)

        self.rcut = rcut
        self.n_cells = (np.array([
            (xmax - xmin),
            (ymax - ymin),
            (zmax - zmin)
        ]) / rcut).astype(int)
        self.d_cells = (self.L[:, 1] - self.L[:, 0]) / self.n_cells

        self.Exit = []
        self.Ob = []
        self.Ob_size = []

        self.Cells = []
        self.initialize_cells()
        self.initialize_particles()

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

        self.reward = reward
        self.end_reward = end_reward
        self.near_end_reward = near_end_reward

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
        if file is None:
            P_list = []
            for i in range(self.Number):
                tries = 0
                reselect = True
                while reselect:
                    pos_2d = (self.L[:2, 0] +
                              0.05*(self.L[:2, 1] - self.L[:2, 0]) +
                              np.random.rand(2)*(self.L[:2, 1] - self.L[:2, 0])*0.9)
                    z_mid = self.L[2, 0] + 0.5*(self.L[2, 1] - self.L[2, 0])
                    pos = np.array([pos_2d[0], pos_2d[1], z_mid])

                    reselect = False
                    for old_pos in P_list:
                        dis = np.linalg.norm(pos - old_pos)
                        if dis < agent_size:
                            reselect = True
                            break
                    
                    tries += 1
                    if tries > 10000:
                        print ("While reselect went on for 10000 times!")
                    if not reselect:
                        P_list.append(pos)

                v = np.random.randn(3) * 0.01
                v[2] = 0.
                particle = Particle(i, pos[0], pos[1], pos[2], v[0], v[1], v[2])
                if i == 0:
                    self.agent = particle
                self.insert_particle(particle)
        else:
            pass

    def randomize_exits_and_obstacles(self):
        self.Exit.clear()
        self.Ob.clear()
        self.Ob_size.clear()

        xmin, xmax = self.L[0, 0], self.L[0, 1]
        ymin, ymax = self.L[1, 0], self.L[1, 1]
        zmin, zmax = self.L[2, 0], self.L[2, 1]
        z_fixed = (zmin + zmax)/2

        for _ in range(self.numExits):
            placed_exit = False
            tries = 0
            while not placed_exit:
                wall_choice = np.random.randint(0, 4)
                if wall_choice == 0:
                    x = xmin
                    y = np.random.uniform(ymin, ymax)
                    if abs(y - ymin) < 1e-3 or abs(ymax - y) < 1e-3:
                        continue
                elif wall_choice == 1:
                    x = xmax
                    y = np.random.uniform(ymin, ymax)
                    if abs(y - ymin) < 1e-3 or abs(ymax - y) < 1e-3:
                        continue
                elif wall_choice == 2:
                    y = ymin
                    x = np.random.uniform(xmin, xmax)
                    if abs(x - xmin) < 1e-3 or abs(xmax - x) < 1e-3:
                        continue
                else:
                    y = ymax
                    x = np.random.uniform(xmin, xmax)
                    if abs(x - xmin) < 1e-3 or abs(xmax - x) < 1e-3:
                        continue
                
                tries += 1
                if tries > 10000:
                    print('Tried reselecting 10000 times. Something failed')

                new_exit_pos = np.array([x, y, z_fixed])
                self.Exit.append(new_exit_pos)
                placed_exit = True

        default_ob_size = 2.0
        agent_pos = self.agent.position

        for _ in range(self.numObs):
            placed_ob = False
            while not placed_ob:
                ox = np.random.uniform(xmin+0.5, xmax-0.5)
                oy = np.random.uniform(ymin+0.5, ymax-0.5)
                cand_pos = np.array([ox, oy, z_fixed])

                if np.linalg.norm(cand_pos - agent_pos) < 1.0:
                    continue

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

                too_close_to_exit = False
                for ex_pos in self.Exit:
                    if np.linalg.norm(cand_pos - ex_pos) < 1.0:
                        too_close_to_exit = True
                        break
                if too_close_to_exit:
                    continue

                self.Ob.append([cand_pos])
                self.Ob_size.append(default_ob_size)
                placed_ob = True

    def insert_particle(self, particle):
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
        for c in self.Cells:
            for p in c.Particles:
                dis_abs = np.abs(p.position[:, np.newaxis] - self.L)
                f = np.where(
                    dis_abs < agent_size,
                    f_wall_lim * np.exp((agent_size - dis_abs) / 0.08) * dis_abs,
                    0.0
                )
                f[:, 1] = -f[:, 1]
                f_net = f.sum(axis=1)
                p.acc += f_net / p.mass

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

                friction = -p.mass/relaxation_time * p.velocity
                p.acc += friction/p.mass

    def loop_cells(self):
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

    def reset(self):
        for cell in self.Cells:
            cell.Particles.clear()
        self.Number = self.Total
        self.initialize_particles()
        self.randomize_exits_and_obstacles()
        return self.get_state()

    def get_state(self):
        """
        Returns:
          occupancy_grid (2D NumPy array),
          (agent_x, agent_y, agent_vx, agent_vy)
        """

        # 1) Build occupancy grid based on domain size & grid_cell_size
        width  = self.L[0,1] - self.L[0,0]
        height = self.L[1,1] - self.L[1,0]

        cols = int(width  / grid_cell_size)
        rows = int(height / grid_cell_size)

        occ_grid = np.zeros((rows, cols), dtype=int)

        # Mark exits => 1
        for epos in self.Exit:
            col, row = self.world_to_grid(epos[0], epos[1], cols, rows)
            if 0 <= row < rows and 0 <= col < cols:
                occ_grid[row, col] = 1

        # Mark obstacles => 2
        for idx_ob, sublist in enumerate(self.Ob):
            for obst_pos in sublist:
                col, row = self.world_to_grid(obst_pos[0], obst_pos[1], cols, rows)
                if 0 <= row < rows and 0 <= col < cols:
                    occ_grid[row, col] = 2

        # 2) Return also the agent info
        ax = self.agent.position[0]
        ay = self.agent.position[1]
        avx = self.agent.velocity[0]
        avy = self.agent.velocity[1]

        return (occ_grid, (ax, ay, avx, avy))

    def world_to_grid(self, x, y, cols, rows):
        """
        Convert (x,y) in domain to integer (col,row) in the occupancy grid,
        each cell=0.5m. row=0 is the domain's y=0 line, row grows with y.
        """
        origin_x = self.L[0,0]
        origin_y = self.L[1,0]
        rel_x = x - origin_x
        rel_y = y - origin_y

        col = int(rel_x / grid_cell_size)
        row = int(rel_y / grid_cell_size)
        return (col, row)

    def step(self, action):
        done = False
        r = self.reward

        self.Zero_acc()
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()

        self.agent.acc += (1/relaxation_time) * desire_velocity * self.action[action]

        self.Integration(1)
        self.Integration(0)
        self.move_particles()

        agent_pos = self.agent.position
        for epos in self.Exit:
            dist = np.linalg.norm(agent_pos - epos)
            if dist < dis_lim:
                done = True
                r = self.end_reward
                break
            elif dist < 2 * dis_lim:
                r = self.near_end_reward
                break

        next_state = self.get_state()
        return next_state, r, done

    def choose_random_action(self):
        return np.random.choice(len(self.action))

    def save_output(self, file):
        N_obs_total = sum(len(ob_sublist) for ob_sublist in self.Ob)
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

            # Exits => type -1
            for epos in self.Exit:
                ex, ey, ez = self.Normalization(epos)
                f.write(f"{ex} {ey} {ez} 0 0 0 -1\n")

            # Obstacles => override type logic if needed
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

            if self.Number > 0:
                f.write("1.000000\nBr\n")
                for c_ in self.Cells:
                    for p in c_.Particles:
                        x, y, z = self.Normalization(p.position)
                        f.write(f"{x} {y} {z} {p.velocity[0]} {p.velocity[1]} {p.velocity[2]} {p.ID}\n")

    def Normalization(self, position):
        return (position - self.L[:, 0])/(self.L[:, 1] - self.L[:, 0])

    def Normalization_XY(self, position_2d):
        return (position_2d - self.L[:2, 0])/(self.L[:2, 1] - self.L[:2, 0]) - offset

    def step_optimal_single_particle(self):
        done = False

        if self.Number == 0:
            done = True
            next_state = self.get_state()
            return next_state, self.end_reward, done

        self.Zero_acc()
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()

        p = self.agent
        dr_min = float('inf')
        dr_unit = np.zeros(3)
        for e in self.Exit:
            dr_tmp = np.linalg.norm(e - p.position)
            if dr_tmp < dr_min:
                dr_min = dr_tmp
                if dr_tmp > 1e-12:
                    dr_unit = (e - p.position)/dr_tmp
                else:
                    dr_unit[:] = 0.0

        costheta_best = -np.inf
        best_action = None
        for act_vec in self.action:
            costheta_tmp = np.dot(act_vec, dr_unit)
            if costheta_tmp > costheta_best:
                costheta_best = costheta_tmp
                best_action = act_vec

        p.acc += (1/relaxation_time) * desire_velocity * best_action

        self.Integration(1)
        self.Integration(0)
        self.move_particles()

        in_exit = False
        for epos in self.Exit:
            dist = np.linalg.norm(p.position - epos)
            if dist < (agent_size + door_size)/2:
                self.Number -= 1
                in_exit = True
                break

        if in_exit:
            done = True
            reward_out = self.end_reward
        else:
            reward_out = self.reward

        next_state = self.get_state()
        return next_state, reward_out, done


if __name__ == '__main__':
    # Simple usage test
    env = Cell_Space(
        xmin=0, xmax=10,
        ymin=0, ymax=10,
        zmin=0, zmax=2,
        dt=0.1,
        Number=1,
        numExits=2,
        numObs=2
    )

    init_state = env.reset()
    # init_state => (occ_grid, (ax, ay, vx, vy))
    occ_grid, agent_tuple = init_state
    print("Initial Occupancy Grid shape:", occ_grid.shape)
    print("Agent data:", agent_tuple)  # (ax, ay, vx, vy)

    done = False
    step_count = 0
    while not done and step_count < 20:
        action = env.choose_random_action()
        next_state, reward, done = env.step(action)
        occ, (ax, ay, avx, avy) = next_state

        print(f"Step {step_count} -> action={action}, reward={reward}, done={done}")
        print(f"   agent=({ax:.2f},{ay:.2f}), vel=({avx:.2f},{avy:.2f})")
        step_count += 1

    # Optional saving
    if not os.path.isdir("./test_cfg"):
        os.mkdir("./test_cfg")
    env.save_output("./test_cfg/s.final")
