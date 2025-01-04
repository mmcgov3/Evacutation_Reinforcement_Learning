import numpy as np
import matplotlib.pyplot as plt


def compute_angle(x,y):
    if x==0 and y ==0:
        print("Error, angle vector can not be 0!")
        quit()
        
    if y > 0:
        angle = np.arccos(x/np.sqrt(x**2 + y**2))
    elif y <0:
        angle = 2*np.pi - np.arccos(x/np.sqrt(x**2 + y**2))
    elif y ==0:
        if x >0:
            angle = 0
        if x < 0:
            angle = np.pi
    
    return angle

class Grid_2D:
    def __init__(self, x, y, category = 'free', bc_value = 0.):
        self.position = np.array((x,y), dtype = 'float')
        self.type = category
        self.bc_value = bc_value
        self.C_prev = 0.
        self.C_curr = 0.
        self.xlow = None
        self.xhigh = None
        self.ylow = None
        self.yhigh = None
    
    def set_xlow(self, xlow):
        self.xlow = xlow
        
    def set_xhigh(self, xhigh):
        self.xhigh = xhigh
        
    def set_ylow(self, ylow):
        self.ylow = ylow
        
    def set_yhigh(self, yhigh):
        self.yhigh = yhigh
        
    def print_neighbor_grid(self):
        print("Position of grid is {}".format(self.position))
        if self.xlow:
            print("xlow is {}".format(self.xlow.position))
        if self.xhigh:
            print("xhigh is {}".format(self.xhigh.position))
        if self.ylow:
            print("ylow is {}".format(self.ylow.position))            
        if self.yhigh:
            print("yhigh is {}".format(self.yhigh.position))    
    
    def FTCS(self, dr, dt, D):
        cs = (self.xlow.C_prev - 2 * self.C_prev + self.xhigh.C_prev)/dr**2 + \
             (self.ylow.C_prev - 2 * self.C_prev + self.yhigh.C_prev)/dr**2
        self.C_curr = self.C_prev + dt * D * cs
    
    def first_bc(self):
        self.C_curr = self.bc_value
        
    def second_bc(self, dt):
        
        if not self.xlow:
            self.C_curr = self.xhigh.C_prev - self.bc_value * dt 

        if not self.xhigh:
            self.C_curr = self.bc_value * dt + self.xlow.C_prev 
            
        if not self.ylow:
            self.C_curr = self.yhigh.C_prev - self.bc_value * dt
            
        if not self.yhigh:
            self.C_curr = self.bc_value * dt + self.ylow.C_prev 
    
    
class GridSpace_2D:
    def __init__(self, xmin = 0., xmax = 10., ymin = 0., ymax = 10., dr= 1.0, dt = 0.01, D = 1.0):
        self.L = np.array([[xmin, xmax],[ymin, ymax]], dtype = 'float')
        self.dr = dr
        self.dt = dt
        self.step = 0
        self.record_step = 10
        self.D = D
        self.source = [[5,5]]
        self.source_value = [2.0]
        
        ####Number of grids in each dimension
        self.n_grids = (self.L[:,1] - self.L[:,0])/dr 
        self.n_grids = self.n_grids.astype('int') + 1 
        self.N_grids = np.cumprod(self.n_grids)[-1]
        
        ####grid size in each dimension
        self.d_grids = (self.L[:,1] - self.L[:,0])/(self.n_grids-1)        
        
        self.neighbor_index = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype='int')
        
        self.C = []
        self.time = []
        self.Grids = []
        self.initial_grids()
        self.set_initial_condition()

    def reset_source(self, source, source_value):
        self.source.clear()
        self.source_value.clear()       
        self.add_source(source, source_value)
            
    def add_source(self, source, source_value):
        
        for idx, s in enumerate(source):
            self.source.append([s[0], s[1]])
            self.source_value.append(source_value[idx])
        
    def reset(self):
        self.step = 0
        self.C.clear()
        self.time.clear()
        self.Grids.clear()
        self.initial_grids()
        self.set_initial_condition()
        
    def initial_grids(self):
        
        for n in range(self.N_grids): 
            boundary = False
            position = self.L[:,0] + np.array([n %self.n_grids[0], n // self.n_grids[0]]) * self.d_grids            
            
            boundary = (position[0] == self.L[0,0]) or (position[0] == self.L[0,1])\
                    or (position[1] == self.L[1,0]) or (position[1] == self.L[1,1])
                    
            if boundary:
                grid = Grid_2D(position[0], position[1], 'first', 0.)
#                grid = Grid_2D(position[0], position[1], 'second', 0.)
            else:
                grid = Grid_2D(position[0], position[1])
            self.Grids.append(grid)
            
    def index_to_number(self, index):
        
        N = index[1] * self.n_grids[0] + index[0] 
        return N

    def valid_index(self, index):
        
        valid = (index[0] >=0 and index[0] < self.n_grids[0])
        valid = valid and (index[1] >=0 and index[1] < self.n_grids[1])
        return valid
        
    def set_initial_condition(self, C=0.):
        
        ###set all grids with C
        for n, g in enumerate(self.Grids):
            g.C_prev = C               
                
            index = np.array([n %self.n_grids[0], n // self.n_grids[0]], dtype = 'int')
            
            xlow_grid_index = index + self.neighbor_index[0]
            if xlow_grid_index[0] >= 0:
                N = self.index_to_number(xlow_grid_index)
                g.set_xlow(self.Grids[N])

            xhigh_grid_index = index + self.neighbor_index[1]
            if xhigh_grid_index[0] < self.n_grids[0]:
                N = self.index_to_number(xhigh_grid_index)
                g.set_xhigh(self.Grids[N])                

            ylow_grid_index = index + self.neighbor_index[2]
            if ylow_grid_index[1] >= 0:
                N = self.index_to_number(ylow_grid_index)
                g.set_ylow(self.Grids[N])

            yhigh_grid_index = index + self.neighbor_index[3]
            if yhigh_grid_index[1] < self.n_grids[1]:
                N = self.index_to_number(yhigh_grid_index)
                g.set_yhigh(self.Grids[N])
        
        ###set source position with value
        self.set_source(current=False)
       
        self.record_concentration()
         
    def set_source(self, current = True):            
        ####set postion grid with value
        if self.source:
        
            for i in range(len(self.source)):
                source_position = self.source[i]
                source_value = self.source_value[i]
                
                on_node = False
                low_index = [ int( (source_position[0] - self.L[0,0]) /self.d_grids[0]) ,
                              int( (source_position[1] - self.L[1,0]) /self.d_grids[1]) ]       
                self.update_source(low_index, source_position, source_value, current)
                
                N = self.index_to_number(low_index)  
                if (source_position[0] == self.Grids[N].position[0] and \
                    source_position[1] == self.Grids[N].position[1] ):
                    on_node = True    
                
                if not on_node:
                    index = [ low_index[0] +1, low_index[1] ]
                    self.update_source(index, source_position, source_value, current)
                    
                    index = [ low_index[0] , low_index[1]+1 ]
                    self.update_source(index, source_position, source_value, current)
    
                    index = [ low_index[0] +1, low_index[1]+1 ]
                    self.update_source(index, source_position, source_value, current)

    def update_source(self, index, source_position, source_value, current):
        if self.valid_index(index):
            diagnal = np.sqrt(self.d_grids[0]**2 + self.d_grids[1]**2)
            N = self.index_to_number(index)            
            dis = np.sqrt( (self.Grids[N].position[0] - source_position[0])**2 + \
                           (self.Grids[N].position[1] - source_position[1])**2 )
            factor = dis / diagnal
            C = source_value * (1 - factor)
            if current:
                self.Grids[N].C_curr = max(self.Grids[N].C_curr, C)
            else:
                self.Grids[N].C_prev = max(self.Grids[N].C_prev, C)        
        

    def update_C(self):
        
        for i in self.Grids:
            i.C_prev = i.C_curr
            i.C_curr = 0

    def step_compute(self):
        
        self.step +=1
        
        #####finite difference to compute C
        self.set_source()
        for g in self.Grids:
            
            if g.type=='free':
                ##free nodes
                if g.C_curr ==0:
                    g.FTCS(self.dr, self.dt, self.D)
            elif g.type == 'first':
                ####1st boundary
                g.first_bc()
            elif g.type == 'second':
                ####2nd boundary
                g.second_bc(self.dt)

        
        self.update_C()
        if self.step % self.record_step == 0:
            self.record_concentration()
        
    def get_concentration(self, x, y):
        low_left_index = [ int( (x - self.L[0,0]) /self.d_grids[0]) ,
                           int( (y - self.L[1,0]) /self.d_grids[1]) ] 
        low_right_index = [ low_left_index[0] +1, low_left_index[1] ]
        high_left_index = [ low_left_index[0] , low_left_index[1]+1 ] 
        high_right_index = [ low_left_index[0] +1, low_left_index[1] +1 ]   
        
        low_left_N = self.index_to_number(low_left_index)
        low_right_N = self.index_to_number(low_right_index)
        high_left_N = self.index_to_number(high_left_index)
        high_right_N = self.index_to_number(high_right_index)
        
        ####linear interpolation
        C1 = self.Grids[low_left_N].C_prev
        C2 = self.Grids[low_right_N].C_prev
        C3 = self.Grids[high_left_N].C_prev
        C4 = self.Grids[high_right_N].C_prev
        
        C5 = C1 + (x - self.Grids[low_left_N].position[0])*(C2 - C1)/self.d_grids[0]
        C6 = C3 + (x - self.Grids[low_left_N].position[0])*(C4 - C3)/self.d_grids[0]
        C = C5 + (y - self.Grids[low_left_N].position[1])*(C6 - C5)/self.d_grids[1]
        return C

    def plot_concentration(self, delta = 0.01, number = 100):
        x = np.linspace(self.L[0,0], self.L[0,1]-delta, number)
        y = np.linspace(self.L[1,0], self.L[1,1]-delta, number)
        xmesh, ymesh = np.meshgrid(x,y)
        
        xflat = xmesh.ravel()
        yflat = ymesh.ravel()
        z = np.vstack((xflat, yflat)).T
        Cmesh = np.zeros(len(z))
        
        for i in range(len(z)):
            C = self.get_concentration(z[i][0], z[i][1])
            Cmesh[i] = C
        
        fig, ax = plt.subplots(figsize = (6,6))
        ax.contourf(xmesh, ymesh, Cmesh.reshape(xmesh.shape), cmap = 'viridis')

        
    def get_gradient(self, x, y):
        low_left_index = [ int( (x - self.L[0,0]) /self.d_grids[0]) ,
                           int( (y - self.L[1,0]) /self.d_grids[1]) ] 
        low_right_index = [ low_left_index[0] +1, low_left_index[1] ]
        high_left_index = [ low_left_index[0] , low_left_index[1]+1 ] 
        high_right_index = [ low_left_index[0] +1, low_left_index[1] +1 ]   
        
        low_left_N = self.index_to_number(low_left_index)
        low_right_N = self.index_to_number(low_right_index)
        high_left_N = self.index_to_number(high_left_index)
        high_right_N = self.index_to_number(high_right_index)
        
        ####fowrad scheme
        C1 = self.Grids[low_left_N].C_prev
        C2 = self.Grids[low_right_N].C_prev
        C3 = self.Grids[high_left_N].C_prev
        C4 = self.Grids[high_right_N].C_prev
        
        C5 = C1 + (x - self.Grids[low_left_N].position[0])*(C2 - C1)/self.d_grids[0]
        C6 = C3 + (x - self.Grids[low_left_N].position[0])*(C4 - C3)/self.d_grids[0]
        C7 = C1 + (y - self.Grids[low_left_N].position[1])*(C3 - C1)/self.d_grids[1]
        C8 = C2 + (y - self.Grids[low_left_N].position[1])*(C4 - C2)/self.d_grids[1]
        Gradient = ( (C8 - C7) / self.d_grids[0] , (C6- C5) / self.d_grids[1] )
        return Gradient     

    def get_gradient_from_C(self, x, y, C):
        low_left_index = [ int( (x - self.L[0,0]) /self.d_grids[0]) ,
                           int( (y - self.L[1,0]) /self.d_grids[1]) ] 
        low_right_index = [ low_left_index[0] +1, low_left_index[1] ]
        high_left_index = [ low_left_index[0] , low_left_index[1]+1 ] 
        high_right_index = [ low_left_index[0] +1, low_left_index[1] +1 ]   
        
        low_left_N = self.index_to_number(low_left_index)
        low_right_N = self.index_to_number(low_right_index)
        high_left_N = self.index_to_number(high_left_index)
        high_right_N = self.index_to_number(high_right_index)
        
        ####fowrad scheme
        C1 = C[low_left_N]
        C2 = C[low_right_N]
        C3 = C[high_left_N]
        C4 = C[high_right_N]
        
        C5 = C1 + (x - self.Grids[low_left_N].position[0])*(C2 - C1)/self.d_grids[0]
        C6 = C3 + (x - self.Grids[low_left_N].position[0])*(C4 - C3)/self.d_grids[0]
        C7 = C1 + (y - self.Grids[low_left_N].position[1])*(C3 - C1)/self.d_grids[1]
        C8 = C2 + (y - self.Grids[low_left_N].position[1])*(C4 - C2)/self.d_grids[1]
        Gradient = ( (C8 - C7) / self.d_grids[0] , (C6- C5) / self.d_grids[1] )
        return Gradient 

    def plot_gradient_direction(self, delta = 0.01, number = 20, 
                                arrow_len = 0.07, normalized = True):
        x = np.linspace(self.L[0,0] + delta, self.L[0,1]-delta, number)
        y = np.linspace(self.L[1,0] + delta, self.L[1,1]-delta, number)
        xmesh, ymesh = np.meshgrid(x,y)
        
        xflat = xmesh.ravel()
        yflat = ymesh.ravel()
        z = np.vstack((xflat, yflat)).T
        Gmesh = np.zeros_like(z)
        
        for i in range(len(z)):
            G = self.get_gradient(z[i][0], z[i][1])
            Gmesh[i] = G
            
        if normalized:
            Glength = np.sqrt(np.sum(Gmesh**2, axis =1))
            Gmesh = Gmesh / Glength[:, np.newaxis] * arrow_len

        ####plot arrows representing the gradient directions
        fig, ax = plt.subplots(figsize = (6,6))
#        ax.set_aspect(aspect=1.0)
        for idx, p in enumerate(z):
            p = p/np.array([self.L[0,1] - self.L[0,0], self.L[1,1] - self.L[1,0]])
            ax.annotate('', xy = p, xytext = Gmesh[idx]+ p,
                        arrowprops=dict(arrowstyle= '<|-',color='k',lw=1.5))

        
            
    def record_concentration(self):
        current_C = np.zeros(self.N_grids)
        for i,g in enumerate(self.Grids):
            current_C[i] = g.C_prev
        
        current_C = current_C.reshape(self.n_grids[0], self.n_grids[1])
        current_C = current_C[::-1]
        
        self.C.append(current_C)
        self.time.append(self.step * self.dt)
        
    def plot_x(self, xindex):
        fig, ax = plt.subplots()
        
        for i in self.C:
            ax.plot(self.d_grids[0] * range(self.n_grids[0]), i[xindex, :])
            
    def generate_gradients_data(self, filename, nx = 100, ny = 100, ntest = 1000, delta = 0.5):
                
        px = np.linspace(self.L[0,0] + delta, self.L[0,1] - delta, nx)
        py = np.linspace(self.L[0,0] + delta, self.L[1,1] - delta, ny)
        
        meshx, meshy = np.meshgrid(px, py)
        meshxy = np.vstack([meshx.ravel(), meshy.ravel()]).T
        
        train_value = []
        for i in range(len(meshxy)):
            g = self.get_gradient(meshxy[i,0], meshxy[i,1])
            angle = compute_angle(*g)
            train_value.append(angle)
  
        train_data = meshxy
        train_data = train_data/np.array([self.L[0,1] - self.L[0,0], 
                                          self.L[1,1] - self.L[1,0]])
        train_value = np.array(train_value)
        
        test_data = []
        test_value = []
        for i in range(ntest):
            xy = self.L[:,0] + delta + np.random.rand(2) * np.array([self.L[0,1] - self.L[0,0] -2 *delta, 
                               self.L[1,1] - self.L[1,0]- 2*delta ])
            g = self.get_gradient(xy[0], xy[1])
            angle = compute_angle(*g)
            
            xy = xy/np.array([self.L[0,1] - self.L[0,0], 
                              self.L[1,1] - self.L[1,0]])
            test_data.append(xy)
            test_value.append(angle)
            
        test_data = np.array(test_data)
        test_value = np.array(test_value)
        
        np.savez(filename, 
                 train_data = train_data,
                 train_value = train_value,
                 test_data = test_data,
                 test_value = test_value)        
    

if __name__ == '__main__':
    
    a = GridSpace_2D(0,10,0,10, dr = 0.1, dt = 0.001, D = 1)
    step_max = 1000
    s = 1
    
    while s <= step_max:
        
        a.step_compute()
        s+=1
    
#    a.plot_x(50)
    a.plot_concentration()
    a.plot_gradient_direction()
    plt.show()
    
    for i in range(1,6):
        
        x = 5 + i*0.6
        y = 5 + i*0.6
        
        a.reset_source([(x,y)], [2])
        
        s = 1
        while s <= 200:
        
            a.step_compute()
            s+=1

        a.plot_concentration()
        a.plot_gradient_direction()    
        plt.show()
#    a.generate_gradients_data('gradient_train_test_data.npz')
