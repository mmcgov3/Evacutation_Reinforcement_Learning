import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
from Continuum_Cellspace import *
import matplotlib.pyplot as plt

Number_Agent = 80
delta_t = 0.05

###### 4Exits
Exit.append(np.array([0.5, 1, 0.5]))  ## Up
#Exit.append(np.array([0.5, 0.0, 0.5]))  ## Down
#Exit.append(np.array([0, 0.5, 0.5]))    ## Left exit
#Exit.append(np.array([1.0, 0.5, 0.5]))  ## Right Exit

######### 3exits with obstacle
'''
Exit.append(np.array([0.7, 1.0, 0.5]))   ## Up exit
Exit.append(np.array([0.5, 0, 0.5]))     ## Down Exit
Exit.append(np.array([0, 0.7, 0.5]))     ## Left exit
#
Ob1 = []
Ob1.append(np.array([0.8, 0.8, 0.5]))
Ob.append(Ob1)
Ob_size.append(2.0)

Ob2 = []
Ob2.append(np.array([0.3, 0.5, 0.5]))
Ob.append(Ob2)
Ob_size.append(3.0)

############

########### Obstacle Center and Up
# Ob1 = []
# Ob1.append(np.array([0.5, 0.7, 0.5]))
# Ob.append(Ob1)
# Ob_size.append(2.0)
'''

output_dir = './Version3_Try'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # He (Kaiming) normal initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='linear')
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x
# Other network classes can be included here if needed
'''
class DQN_4exit(nn.Module):
    # Your implementation here
    pass

class DQN_3exit_Ob(nn.Module):
    # Your implementation here
    pass

class DQN_Ob(nn.Module):
    # Your implementation here
    pass
'''

if __name__ == '__main__':

    test_episodes = 0       # Max number of episodes to test
    max_steps = 10000       # Max steps in an episode
    gamma = 0.999           # Future reward discount

    explore_start = 1.0     # Exploration probability at start
    explore_stop = 0.1      # Minimum exploration probability
    # decay_rate = 0.00002  # Exponential decay rate for exploration prob
    decay_percentage = 0.5
    decay_rate = 4 / decay_percentage

    # Network parameters
    learning_rate = 5e-5     # Q-network learning rate
    memory_size = 10000      # Memory capacity
    batch_size = 50          # Experience mini-batch size
    pretrain_length = batch_size  # Number experiences to pretrain the memory

    Cfg_save_freq = 1
    cfg_save_step = 2

    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()
    state_size = 4
    action_size = 8
    
    # Initialize networks
    #policy_net = QNetwork(state_size, action_size)
    # Initialize the networks
    #mainQN_Up = DQN()
    #mainQN_Up = DQN(name='main_qn_up', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # mainQN_Down = DQN(name='main_qn_down', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # mainQN_2Exits = DQN(name='main_qn_2exits', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # mainQN_4Exits = DQN_4exit(name='main_qn_4exits', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # mainQN_3Exits_Ob = DQN_3exit_Ob(name='main_qn_3exits_ob', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # mainQN_Ob_center = DQN(name='main_qn_ob_center', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # mainQN_Ob_up = DQN(name='main_qn_ob_up', learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mainQN_Up = DQN().to(device)
    # mainQN_Down.to(device)
    # mainQN_2Exits.to(device)
    # mainQN_4Exits.to(device)
    # mainQN_3Exits_Ob.to(device)
    # mainQN_Ob_center.to(device)
    # mainQN_Ob_up.to(device)

    # Loop over model files from 500 to 15000
    for model_number in range(1000, 10001, 1000):
        model_saved_path_up = './model/Continuum_1ExitUp_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep{}.pth'.format(model_number)
        # model_saved_path_down = './model/Continuum_Down_DQN_Fully.pth'
        # model_saved_path_2exits = './model/Continuum_2Exit_DQN_Fully_PyTorch/Evacuation_Continuum_model_{}.pth'.format(model_number)
        # model_saved_path_4exits = './model/Continuum_4Exits_DQN_Fully_PyTorch/Evacuation_Continuum_model_{}.pth'.format(model_number)
        # model_saved_path_3exits_ob = './model/Continuum_3Exit_Ob_DQN_Fully_PyTorch/Evacuation_Continuum_model_{}.pth'.format(model_number)
        '''
        model_saved_path_ob_center = './model/Continuum_Ob_Center_DQN_Fully.pth'
        model_saved_path_ob_up = './model/Continuum_Ob_Up_DQN_Fully.pth'
        '''

        # Check for saved model and load it
        if os.path.isfile(model_saved_path_up):
            #policy_net.load_state_dict(torch.load(model_saved_path_up))
            mainQN_Up.load_state_dict(torch.load(model_saved_path_up, map_location=torch.device('cpu')))
            print("Successfully loaded:", model_saved_path_up)
        else:
            print("Model file not found:", model_saved_path_up)
            continue  # Skip to the next model if not found
        '''
        if os.path.isfile(model_saved_path_up):
            mainQN_Up.load_state_dict(torch.load(model_saved_path_up, map_location=device))
            print("Successfully loaded:", model_saved_path_up)
        else:
            print("Model file not found:", model_saved_path_up)
            continue  # Skip to the next model if not found
        '''
        '''
        if os.path.isfile(model_saved_path_down):
            mainQN_Down.load_state_dict(torch.load(model_saved_path_down))
            print("Successfully loaded:", model_saved_path_down)

        if os.path.isfile(model_saved_path_2exits):
            mainQN_2Exits.load_state_dict(torch.load(model_saved_path_2exits))
            print("Successfully loaded:", model_saved_path_2exits)

        if os.path.isfile(model_saved_path_4exits):
            mainQN_4Exits.load_state_dict(torch.load(model_saved_path_4exits))
            print("Successfully loaded:", model_saved_path_4exits)

        if os.path.isfile(model_saved_path_3exits_ob):
            mainQN_3Exits_Ob.load_state_dict(torch.load(model_saved_path_3exits_ob))
            print("Successfully loaded:", model_saved_path_3exits_ob)

        if os.path.isfile(model_saved_path_ob_center):
            mainQN_Ob_center.load_state_dict(torch.load(model_saved_path_ob_center))
            print("Successfully loaded:", model_saved_path_ob_center)

        if os.path.isfile(model_saved_path_ob_up):
            mainQN_Ob_up.load_state_dict(torch.load(model_saved_path_ob_up))
            print("Successfully loaded:", model_saved_path_ob_up)
        '''

        ############ Illustration of force direction
        # Assuming offset is defined somewhere in Continuum_Cellspace
        # If not, define offset here
        offset = [0.5, 0.5]

        x, y = np.meshgrid(np.linspace(0, 1, 100) - offset[0], np.linspace(0, 1, 100) - offset[1])
        x_arrow, y_arrow = np.meshgrid(np.linspace(0.05, 0.95, 15) - offset[0], np.linspace(0.05, 0.95, 15) - offset[1])
        xy = np.vstack([x.ravel(), y.ravel()]).T
        xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T

        ### Random velocity
        vxy = np.random.randn(*xy.shape) * 0.
        vxy_arrow = np.random.randn(*xy_arrow.shape) * 0.

        #### Constant velocity
        vxy[:, 1] = 0.5
        vxy_arrow[:, 1] = 0.5

        xtest = np.hstack([xy, vxy])
        x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

        # Convert to torch tensors
        #xtest_tensor = torch.from_numpy(xtest).float()
        #x_arrow_test_tensor = torch.from_numpy(x_arrow_test).float()
        xtest_tensor = torch.from_numpy(xtest).float().to(device)
        x_arrow_test_tensor = torch.from_numpy(x_arrow_test).float().to(device)

        # Get predictions from the model
        #ypred = policy_net(xtest_tensor).cpu().detach().numpy()
        #ypred_arrow = policy_net(x_arrow_test_tensor).cpu().detach().numpy()
        
        ypred = mainQN_Up(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_Up(x_arrow_test_tensor).cpu().detach().numpy()

        '''
        ypred = mainQN_Down(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_Down(x_arrow_test_tensor).cpu().detach().numpy()

        ypred = mainQN_2Exits(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_2Exits(x_arrow_test_tensor).cpu().detach().numpy()

        ypred = mainQN_4Exits(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_4Exits(x_arrow_test_tensor).cpu().detach().numpy()

        ypred = mainQN_3Exits_Ob(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_3Exits_Ob(x_arrow_test_tensor).cpu().detach().numpy()

        ypred = mainQN_Ob_center(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_Ob_center(x_arrow_test_tensor).cpu().detach().numpy()

        ypred = mainQN_Ob_up(xtest_tensor).cpu().detach().numpy()
        ypred_arrow = mainQN_Ob_up(x_arrow_test_tensor).cpu().detach().numpy()
        '''

        action_pred = np.argmax(ypred, axis=1)
        action_arrow_pred = np.argmax(ypred_arrow, axis=1)

        ### Up trim 0.5 v1
        action_pred[((xy[:,1] > 0.3) & (xy[:,0] < -0.3))] = 6
        action_pred[((np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) >0.55) & (action_pred == 1))] = 0
        action_arrow_pred[((xy_arrow[:,1] > 0.3) & (xy_arrow[:,0] < -0.3))] = 6
        action_arrow_pred[((np.sqrt((xy_arrow[:,1]- 0.5 )**2 + (xy_arrow[:,0]- 0.5 )**2 ) >0.55) & (action_arrow_pred == 1))] = 0

        '''
        ### Down trim 0.5 v1
        action_pred[((xy[:, 1] > 0) & (xy[:, 0] > -0.4) & (xy[:, 0] < 0))] = 4
        action_arrow_pred[((xy_arrow[:, 1] > 0) & (xy_arrow[:, 0] > -0.4) & (xy_arrow[:, 0] < 0))] = 4

        #####2 exits trim
        action_pred[((np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.5) & (action_pred == 0))] = 1
        action_pred[((np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45) & (action_pred == 4))] = 5
        action_pred[((np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.47) & (action_pred == 4))] = 3

        action_arrow_pred[((np.sqrt((xy_arrow[:,1]- 0.5 )**2 + (xy_arrow[:,0]- 0.5 )**2 ) <0.5) & (action_arrow_pred == 0))] = 1        
        action_arrow_pred[((np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + (xy_arrow[:,0]+ 0.5 )**2 ) <0.45) & (action_arrow_pred == 4))] = 5 
        action_arrow_pred[((np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + (xy_arrow[:,0]- 0.5 )**2 ) <0.47) & (action_arrow_pred == 4))] = 3     

        #####4 exits trim
        action_pred[((np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) >0.47) & (xy[:,0] < 0.12) & (action_pred == 7))] = 0
        action_arrow_pred[((np.sqrt((xy_arrow[:,1]- 0.5 )**2 + (xy_arrow[:,0]+ 0.5 )**2 ) >0.47) & (xy_arrow[:,0] <0.12) & (action_arrow_pred == 7))] = 0        
        '''

        action_grid = action_pred.reshape(x.shape)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'xlim': (-0.5, 0.5),
                                                                 'ylim': (-0.5, 0.5)})

        #### Contour plot
        # c_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        # contour = ax.contourf(x, y, action_grid + 0.1, colors=c_map, alpha=0.8)

        contour = ax.contourf(x, y, action_grid + 0.1, cmap=plt.cm.get_cmap('rainbow'), alpha=0.8)
        # cbar = fig.colorbar(contour, ticks=range(8))
        # cbar.set_label('Force direction')
        # cbar.set_ticklabels(['Up', 'Up-Left', 'Left','Down-Left','Down',
        #                      'Down-right', 'Right', 'Up-Right', 'Right'])

        ### Arrow
        arrow_len = 0.07
        angle = np.sqrt(2) / 2
        arrow_map = {0: [0, arrow_len], 1: [-angle * arrow_len, angle * arrow_len],
                     2: [-arrow_len, 0], 3: [-angle * arrow_len, -angle * arrow_len],
                     4: [0, -arrow_len], 5: [angle * arrow_len, -angle * arrow_len],
                     6: [arrow_len, 0], 7: [angle * arrow_len, angle * arrow_len]}
        for idx, p in enumerate(xy_arrow):
            ax.annotate('', xy=p, xytext=np.array(arrow_map[action_arrow_pred[idx]]) + p,
                        arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))

        ax.tick_params(labelsize='large')
        # plt.show()

        # Save the figure
        figure_filename = 'HeNormal_P_{}_decayLR_.png'.format(model_number)
        figure_filepath = os.path.join(output_dir, figure_filename)
        fig.savefig(figure_filepath, dpi=600)
        plt.close(fig)  # Close the figure to free memory

        print("Figure saved:", figure_filepath)

    # The rest of your testing code can remain unchanged if needed
    step = 0

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for ep in range(0, test_episodes):
        total_reward = 0
        t = 0

        print("Testing episode: {}".format(ep))

        if ep % Cfg_save_freq == 0:

            pathdir = os.path.join(output_dir, 'case_' + str(ep))
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)

            else:
                for filename in os.listdir(output_dir):
                    filepath = os.path.join(output_dir, filename)

                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)

            env.save_output(pathdir + '/s.' + str(t))

        while t < max_steps:

            # Get action from Q-network
            ######### ALL Particles
            # done = env.step_all(mainQN_Up, Normalized=True)
            # done = env.step_all(mainQN_Down, Normalized=True)
            # done = env.step_all(mainQN_2Exits, Normalized=True)
            # done = env.step_all(mainQN_4Exits, Normalized=True)
            # done = env.step_all(mainQN_Ob_center, Normalized=True)
            # done = env.step_all(mainQN_Ob_up, Normalized=True)

            done = env.step_optimal()

            step += 1
            t += 1

            if done:
                # Start new episode
                if ep % Cfg_save_freq == 0:
                    env.save_output(pathdir + '/s.' + str(t))

                state = env.reset()
                break

            else:

                if ep % Cfg_save_freq == 0:
                    if t % cfg_save_step == 0:
                        env.save_output(pathdir + '/s.' + str(t))

        print("Total steps in episode {} is : {}".format(ep, t))
