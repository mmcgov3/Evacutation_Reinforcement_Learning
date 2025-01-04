####Test of trained model for evacuation

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from Continuum_Cellspace import *

Number_Agent = 80
delta_t = 0.05

######4Exits
Exit.append( np.array([0.5, 1.0, 0.5]) )  ##Up
Exit.append( np.array([0.5, 0.0, 0.5]) )  ##Down
Exit.append( np.array([0, 0.5, 0.5]) )    ##Add Left exit
Exit.append( np.array([1.0, 0.5, 0.5]) )  ##Add Right Exit


#########3exits with ob
#Exit.append( np.array([0.7, 1.0, 0.5]) )  ##Add Up exit
#Exit.append( np.array([0.5, 0, 0.5]) )     ##Add Down Exit
#Exit.append( np.array([0, 0.7, 0.5]) )  ##Add Left exit
#
#Ob1 = []
#Ob1.append(np.array([0.8, 0.8, 0.5]))
#Ob.append(Ob1)
#Ob_size.append(2.0)
#
#Ob2 = []
#Ob2.append(np.array([0.3, 0.5, 0.5]))
#Ob.append(Ob2)
#Ob_size.append(3.0)


############

###########Ob Center and Up
#Ob1 = []
#Ob1.append(np.array([0.5, 0.7, 0.5]))
#Ob.append(Ob1)
#Ob_size.append(2.0)

output_dir = './Test'
model_saved_path_up = './model/Continuum_Up_DQN_Fully'
model_saved_path_down = './model/Continuum_Down_DQN_Fully'
model_saved_path_2exits = './model/Continuum_2Exits_DQN_Fully'
model_saved_path_4exits = './model/Continuum_4Exits_DQN_Fully'
model_saved_path_3exits_ob = './model/Continuum_3Exits_Ob_DQN_Fully'
model_saved_path_ob_center = './model/Continuum_Ob_Center_DQN_Fully'
model_saved_path_ob_up = './model/Continuum_Ob_Up_DQN_Fully'


# PyTorch model definitions corresponding to the DQN architectures
class DQN_Net(nn.Module):
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

class DQN_4exit_Net(nn.Module):
    def __init__(self, input_size=4, action_size=8):
        super(DQN_4exit_Net, self).__init__()
        self.f1 = nn.Linear(input_size, 64)
        self.f2 = nn.Linear(64, 128)
        self.f3 = nn.Linear(128, 128)
        self.f4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.elu(self.f1(x))
        x = F.elu(self.f2(x))
        x = F.elu(self.f3(x))
        x = F.elu(self.f4(x))
        x = self.out(x)
        return x

class DQN_3exit_Ob_Net(nn.Module):
    def __init__(self, input_size=4, action_size=8):
        super(DQN_3exit_Ob_Net, self).__init__()
        self.f1 = nn.Linear(input_size, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, 64)
        self.f4 = nn.Linear(64, 64)
        self.f5 = nn.Linear(64, 64)
        self.f6 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))
        x = F.relu(self.f5(x))
        x = F.relu(self.f6(x))
        x = self.out(x)
        return x

class DQN_Ob_Net(nn.Module):
    def __init__(self, input_size=4, action_size=8):
        super(DQN_Ob_Net, self).__init__()
        self.f1 = nn.Linear(input_size, 32)
        self.f2 = nn.Linear(32, 64)
        self.f3 = nn.Linear(64, 64)
        self.f4 = nn.Linear(64, 32)
        self.f5 = nn.Linear(32, 32)
        self.f6 = nn.Linear(32, 64)
        self.f7 = nn.Linear(64, 64)
        self.f8 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))
        x = F.relu(self.f5(x))
        x = F.relu(self.f6(x))
        x = F.relu(self.f7(x))
        x = F.relu(self.f8(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    
    test_episodes = 0        # max number of episodes to test
    max_steps = 10000        # max steps in an episode
    gamma = 0.999            # future reward discount

    explore_start = 1.0      # exploration probability at start
    explore_stop = 0.1       # minimum exploration probability
    # decay_rate = 0.00002    # exponential decay rate for exploration prob
    decay_percentage = 0.5
    decay_rate = 4/decay_percentage
            
    # Network parameters
    learning_rate = 1e-4     # Q-network learning rate 

    # Memory parameters
    memory_size = 10000       # memory capacity
    batch_size = 50           # experience mini-batch size
    pretrain_length = batch_size   # number experiences to pretrain the memory    
    
    Cfg_save_freq = 1
    cfg_save_step = 2
    
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut= 1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()
        
    # Instead of tf.reset_default_graph(), PyTorch doesn't need that.
    # Create model instances
    mainQN_Up = DQN_Net()
    mainQN_Down = DQN_Net()
    mainQN_2Exits = DQN_Net()
    mainQN_4Exits = DQN_4exit_Net()
    mainQN_3Exits_Ob = DQN_3exit_Ob_Net()
    mainQN_Ob_center = DQN_Net()
    mainQN_Ob_up = DQN_Net()

    # TF code:
    # Up_list = mainQN_Up.get_qnetwork_variables()
    # Down_list = mainQN_Down.get_qnetwork_variables()
    # Exits_list = mainQN_2Exits.get_qnetwork_variables()
    # Four_list = mainQN_4Exits.get_qnetwork_variables()
    # Ob_list = mainQN_3Exits_Ob.get_qnetwork_variables()
    # Ob_Center_list = mainQN_Ob_center.get_qnetwork_variables()
    # Ob_Up_list = mainQN_Ob_up.get_qnetwork_variables()
    
    # init = tf.global_variables_initializer()
    # saver_up = tf.train.Saver(Up_list)
    # saver_down = tf.train.Saver(Down_list)
    # saver_2exits = tf.train.Saver(Exits_list)
    # saver_4exits = tf.train.Saver(Four_list)
    # saver_3exits_ob = tf.train.Saver(Ob_list)
    # saver_ob_center = tf.train.Saver(Ob_Center_list)
    # saver_ob_up = tf.train.Saver(Ob_Up_list)

    ######GPU usage fraction
    # In PyTorch, we typically just move models to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mainQN_Up.to(device)
    mainQN_Down.to(device)
    mainQN_2Exits.to(device)
    mainQN_4Exits.to(device)
    mainQN_3Exits_Ob.to(device)
    mainQN_Ob_center.to(device)
    mainQN_Ob_up.to(device)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4

    # with tf.Session(config = config) as sess:
    #     sess.run(init)
        
    #     # Check saved model
    #     if not os.path.isdir(model_saved_path_up):
    #         os.mkdir(model_saved_path_up)

    #     if not os.path.isdir(model_saved_path_down):
    #         os.mkdir(model_saved_path_down)

    #     if not os.path.isdir(model_saved_path_2exits):
    #         os.mkdir(model_saved_path_2exits)

    #     if not os.path.isdir(model_saved_path_4exits):
    #         os.mkdir(model_saved_path_4exits)

    #     if not os.path.isdir(model_saved_path_3exits_ob):
    #         os.mkdir(model_saved_path_3exits_ob)

    #     if not os.path.isdir(model_saved_path_ob_center):
    #         os.mkdir(model_saved_path_ob_center)

    #     if not os.path.isdir(model_saved_path_ob_up):
    #         os.mkdir(model_saved_path_ob_up)
            
    #     checkpoint_up = tf.train.get_checkpoint_state(model_saved_path_up)
    #     if checkpoint_up and checkpoint_up.model_checkpoint_path:
    #         # saver_up.restore(sess, checkpoint_up.all_model_checkpoint_paths[2])
    #         print("Successfully loaded:", checkpoint_up.model_checkpoint_path)

    #     checkpoint_down = tf.train.get_checkpoint_state(model_saved_path_down)
    #     if checkpoint_down and checkpoint_down.model_checkpoint_path:
    #         # saver_down.restore(sess, checkpoint_down.all_model_checkpoint_paths[6])
    #         print("Successfully loaded:", checkpoint_down.model_checkpoint_path)       


    # PyTorch: If you have PyTorch saved models, you would do something like:
    # if os.path.exists(model_saved_path_up + "/model.pt"):
    #     mainQN_Up.load_state_dict(torch.load(model_saved_path_up + "/model.pt", map_location=device))
    #     print("Successfully loaded PyTorch model:", model_saved_path_up + "/model.pt")

    # if os.path.exists(model_saved_path_down + "/model.pt"):
    #     mainQN_Down.load_state_dict(torch.load(model_saved_path_down + "/model.pt", map_location=device))
    #     print("Successfully loaded PyTorch model:", model_saved_path_down + "/model.pt")

    '''
    checkpoint_2exits = tf.train.get_checkpoint_state(model_saved_path_2exits)
    if checkpoint_2exits and checkpoint_2exits.model_checkpoint_path:
#            saver_2exits.restore(sess, checkpoint_2exits.all_model_checkpoint_paths[1])
        print("Successfully loaded:", checkpoint_2exits.model_checkpoint_path)   
    
    checkpoint_4exits = tf.train.get_checkpoint_state(model_saved_path_4exits)
    if checkpoint_4exits and checkpoint_4exits.model_checkpoint_path:
        #saver_4exits.restore(sess, checkpoint_4exits.model_checkpoint_path)
        print("Successfully loaded:", checkpoint_4exits.model_checkpoint_path)            
    
    checkpoint_3exits_ob = tf.train.get_checkpoint_state(model_saved_path_3exits_ob)
    if checkpoint_3exits_ob and checkpoint_3exits_ob.model_checkpoint_path:
#            saver_3exits_ob.restore(sess, checkpoint_3exits_ob.all_model_checkpoint_paths[2])  
        print("Successfully loaded:", checkpoint_3exits_ob.model_checkpoint_path) 

    checkpoint_ob_center = tf.train.get_checkpoint_state(model_saved_path_ob_center)
    if checkpoint_ob_center and checkpoint_ob_center.model_checkpoint_path:
#            saver_ob_center.restore(sess, checkpoint_ob_center.all_model_checkpoint_paths[1])  
        print("Successfully loaded:", checkpoint_ob_center.model_checkpoint_path) 

    checkpoint_ob_up = tf.train.get_checkpoint_state(model_saved_path_ob_up)
    if checkpoint_ob_up and checkpoint_ob_up.model_checkpoint_path:
#            saver_ob_up.restore(sess, checkpoint_ob_up.all_model_checkpoint_paths[9])  
        print("Successfully loaded:", checkpoint_ob_up.model_checkpoint_path) 
    '''

    ########## For testing and plotting
    offset = [0,0] # define offset as it was used implicitly in code
    x, y = np.meshgrid(np.linspace(0,1,100)-offset[0], np.linspace(0,1,100)-offset[1])
    x_arrow, y_arrow = np.meshgrid(np.linspace(0.05,0.95,15)-offset[0], np.linspace(0.05,0.95,15)-offset[1])
    xy = np.vstack([x.ravel(), y.ravel()]).T
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T

    ###random velocity
    vxy = np.random.randn(*xy.shape)*0.
    vxy_arrow = np.random.randn(*xy_arrow.shape)*0.

    ####constant velocity
    vxy[:,1] = 0.5
    vxy_arrow[:,1] = 0.5

    xtest = np.hstack([xy, vxy])
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

    # In PyTorch, forward pass:
    mainQN_Up.eval()
    mainQN_Down.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(xtest).float().to(device)
        ypred_up = mainQN_Up(inputs).cpu().numpy()
        ypred_arrow_up = mainQN_Up(torch.from_numpy(x_arrow_test).float().to(device)).cpu().numpy()

        ypred_down = mainQN_Down(inputs).cpu().numpy()
        ypred_arrow_down = mainQN_Down(torch.from_numpy(x_arrow_test).float().to(device)).cpu().numpy()

    # The code below seems to pick actions from either up or down. It first calls mainQN_Up and then mainQN_Down.
    # Since the code later uses `ypred` and `ypred_arrow` final, let's just use `ypred_down` as the final result 
    # to mimic the original code structure where it overwrote `ypred` and `ypred_arrow`.

    ypred = ypred_down
    ypred_arrow = ypred_arrow_down

    # The rest of the code remains the same as the original
    action_pred = np.argmax(ypred, axis =1)
    action_arrow_pred = np.argmax(ypred_arrow, axis =1)

    ###up tirm 0.5 v1
    action_pred[((xy[:,1] >0.3) & (xy[:,0] <-0.3) )] = 6
    action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) >0.55)  & (action_pred == 1) )] = 0
    action_arrow_pred[((xy_arrow[:,1]>0.3) & (xy_arrow[:,0] <-0.3) )] = 6
    action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 +
                                 (xy_arrow[:,0]- 0.5 )**2 ) >0.55)
                                  &(action_arrow_pred == 1) )] = 0

    ###down tirm 0.5 v1
    action_pred[((xy[:,1]>0) & (xy[:,0] >-0.4) & (xy[:,0] <0))] = 4
    action_arrow_pred[((xy_arrow[:,1]>0) & (xy_arrow[:,0] >-0.4) & (xy_arrow[:,0] <0))] = 4

    action_grid = action_pred.reshape(x.shape)

    fig, ax = plt.subplots(1,1, figsize = (5,5), subplot_kw = {'xlim' : (-0.5, 0.5),
                               'ylim' : (-0.5, 0.5)})

    contour = ax.contourf(x,y,action_grid+0.1,cmap = plt.cm.get_cmap('rainbow'), alpha = 0.8)

    arrow_len = 0.07
    angle = np.sqrt(2)/2
    arrow_map = {0 : [0, arrow_len], 1: [-angle * arrow_len, angle * arrow_len],
                 2 : [-arrow_len, 0], 3: [-angle * arrow_len, -angle * arrow_len],
                 4 : [0, -arrow_len], 5: [angle * arrow_len, -angle * arrow_len],
                 6 : [arrow_len, 0], 7: [angle * arrow_len, angle * arrow_len],}

    for idx, p in enumerate(xy_arrow):
        ax.annotate('', xy = p, xytext = np.array(arrow_map[action_arrow_pred[idx]])+ p,
                    arrowprops=dict(arrowstyle= '<|-',color='k',lw=1.5))

    ax.tick_params(labelsize = 'large')
    plt.show()


    step = 0     

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  

    for ep in range(0, test_episodes):
        total_reward = 0
        t = 0

        print("Testing episode: {}".format(ep))

        if ep % Cfg_save_freq ==0:
            
            pathdir = os.path.join(output_dir, 'case_' + str(ep) )             
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
            # done = env.step_all(sess, mainQN_Up, Normalized=True)  
            # done = env.step_all(sess, mainQN_Down, Normalized=True) 
            # done = env.step_all(sess, mainQN_2Exits, Normalized=True) 
            # done = env.step_all(sess, mainQN_4Exits, Normalized=True) 
            # done = env.step_all(sess, mainQN_Ob_center, Normalized=True) 
            # done = env.step_all(sess, mainQN_Ob_up, Normalized=True) 

            # For PyTorch, you would do something like:
            # done = env.step_all_pytorch(mainQN_Up, device, Normalized=True)
            # But since this code is not provided, we assume step_optimal is unchanged:
            done = env.step_optimal() 

            step += 1
            t += 1

            if done:
                # Start new episode
                if ep % Cfg_save_freq ==0:
                    env.save_output(pathdir + '/s.' + str(t))

                state = env.reset()
                break

            else:

                if ep % Cfg_save_freq ==0:
                    if t%cfg_save_step ==0:
                        env.save_output(pathdir + '/s.' + str(t))

        print("Total steps in episode {} is : {}".format(ep, t))
