####Test of trained model for evacuation

import numpy as np
import tensorflow as tf
import os
import shutil
from Continuum_Cellspace import *

Number_Agent = 80
delta_t = 0.05

######4Exits
Exit.append( np.array([0.5, 1.0, 0.5]) )  ##Up
Exit.append( np.array([0.5, 0.0, 0.5]) )  ##Down
Exit.append( np.array([0, 0.5, 0.5]) )  ##Add Left exit
Exit.append( np.array([1.0, 0.5, 0.5]) )     ##Add Right Exit


#########3exits with ob
Exit.append( np.array([0.7, 1.0, 0.5]) )  ##Add Up exit
Exit.append( np.array([0.5, 0, 0.5]) )     ##Add Down Exit
Exit.append( np.array([0, 0.7, 0.5]) )  ##Add Left exit
#
Ob1 = []
Ob1.append(np.array([0.8, 0.8, 0.5]))
Ob.append(Ob1)
Ob_size.append(2.0)
#
Ob2 = []
Ob2.append(np.array([0.3, 0.5, 0.5]))
Ob.append(Ob2)
Ob_size.append(3.0)


############

###########Ob Center and Up
Ob1 = []
Ob1.append(np.array([0.5, 0.7, 0.5]))
Ob.append(Ob1)
Ob_size.append(2.0)

output_dir = './Test'
model_saved_path_up = './model/Continuum_Up_DQN_Fully'
model_saved_path_down = './model/Continuum_Down_DQN_Fully'
model_saved_path_2exits = './model/Continuum_2Exits_DQN_Fully'
model_saved_path_4exits = './model/Continuum_4Exits_DQN_Fully'
model_saved_path_3exits_ob = './model/Continuum_3Exits_Ob_DQN_Fully'
model_saved_path_ob_center = './model/Continuum_Ob_Center_DQN_Fully'
model_saved_path_ob_up = './model/Continuum_Ob_Up_DQN_Fully'

class DQN:
    def __init__(self, name, learning_rate=0.0001, gamma = 0.99,
                 action_size=8, batch_size=20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')  
            self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64, activation_fn=tf.nn.elu)
            self.f2 = tf.contrib.layers.fully_connected(self.f1, 128, activation_fn=tf.nn.elu)
            self.f3 = tf.contrib.layers.fully_connected(self.f2, 64, activation_fn=tf.nn.elu)

            self.output = tf.contrib.layers.fully_connected(self.f3, action_size, activation_fn=None)

            
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)] 
  
class DQN_4exit:
    def __init__(self, name, learning_rate=0.0001, gamma = 0.99,
                 action_size=8, batch_size=20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')  

            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn=tf.nn.elu, 
                    weights_initializer=tf.initializers.he_normal()
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 128)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 128)
                self.f4 = tf.contrib.layers.fully_connected(self.f3, 64)


            self.output = tf.contrib.layers.fully_connected(self.f4, action_size, activation_fn=None)
    
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]  


class DQN_3exit_Ob:
    def __init__(self, name, learning_rate=0.0001, gamma = 0.99,
                 action_size=8, batch_size=20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')  

            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn=tf.nn.relu,                    
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 64)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 64)
                self.f4 = tf.contrib.layers.fully_connected(self.f3, 64)
                self.f5 = tf.contrib.layers.fully_connected(self.f4, 64)
                self.f6 = tf.contrib.layers.fully_connected(self.f5, 64)

            self.output = tf.contrib.layers.fully_connected(self.f6, action_size, activation_fn=None)
            
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)] 
  
class DQN_Ob:
    def __init__(self, name, learning_rate=0.0001, gamma = 0.99,
                 action_size=8, batch_size=20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')  
            
            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn=tf.nn.relu,                    
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 32)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 64)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 64)
                self.f4 = tf.contrib.layers.fully_connected(self.f3, 32)
                self.f5 = tf.contrib.layers.fully_connected(self.f4, 32)
                self.f6 = tf.contrib.layers.fully_connected(self.f5, 64)
                self.f7 = tf.contrib.layers.fully_connected(self.f6, 64)
                self.f8 = tf.contrib.layers.fully_connected(self.f7, 32)

            self.output = tf.contrib.layers.fully_connected(self.f8, action_size, activation_fn=None)
         
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)] 
    
if __name__ == '__main__':
    
    test_episodes = 0        # max number of episodes to test
    max_steps = 10000                # max steps in an episode
    gamma = 0.999                   # future reward discount

    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
#    decay_rate = 0.00002            # exponential decay rate for exploration prob
    decay_percentage = 0.5        
    decay_rate = 4/decay_percentage
            
    # Network parameters
    learning_rate = 1e-4         # Q-network learning rate 
    
    # Memory parameters
    memory_size = 10000          # memory capacity
    batch_size = 50                # experience mini-batch size
    pretrain_length = batch_size   # number experiences to pretrain the memory    
    
    Cfg_save_freq = 1
    cfg_save_step = 2
    
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut= 1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()
        
    tf.reset_default_graph()
    mainQN_Up = DQN(name='main_qn_up', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    mainQN_Down = DQN(name='main_qn_down', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    mainQN_2Exits = DQN(name='main_qn_2exits', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    mainQN_4Exits = DQN_4exit(name='main_qn_4exits', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    mainQN_3Exits_Ob = DQN_3exit_Ob(name='main_qn_3exits_ob', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    mainQN_Ob_center = DQN(name='main_qn_ob_center', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    mainQN_Ob_up = DQN(name='main_qn_ob_up', learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
 
    Up_list = mainQN_Up.get_qnetwork_variables()
    Down_list = mainQN_Down.get_qnetwork_variables()
    Exits_list = mainQN_2Exits.get_qnetwork_variables()
    Four_list = mainQN_4Exits.get_qnetwork_variables()
    Ob_list = mainQN_3Exits_Ob.get_qnetwork_variables()
    Ob_Center_list = mainQN_Ob_center.get_qnetwork_variables()
    Ob_Up_list = mainQN_Ob_up.get_qnetwork_variables()
    
    init = tf.global_variables_initializer()
    saver_up = tf.train.Saver(Up_list)
    saver_down = tf.train.Saver(Down_list)
    saver_2exits = tf.train.Saver(Exits_list)
    saver_4exits = tf.train.Saver(Four_list)
    saver_3exits_ob = tf.train.Saver(Ob_list)
    saver_ob_center = tf.train.Saver(Ob_Center_list)
    saver_ob_up = tf.train.Saver(Ob_Up_list)

    ######GPU usage fraction
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config = config) as sess:   
        
        sess.run(init)
        ####check saved model
        if not os.path.isdir(model_saved_path_up):
            os.mkdir(model_saved_path_up)
        
        if not os.path.isdir(model_saved_path_down):
            os.mkdir(model_saved_path_down)
        
        if not os.path.isdir(model_saved_path_2exits):
            os.mkdir(model_saved_path_2exits)
            
        if not os.path.isdir(model_saved_path_4exits):
            os.mkdir(model_saved_path_4exits)
        
        if not os.path.isdir(model_saved_path_3exits_ob):
            os.mkdir(model_saved_path_3exits_ob)

        if not os.path.isdir(model_saved_path_ob_center):
            os.mkdir(model_saved_path_ob_center)

        if not os.path.isdir(model_saved_path_ob_up):
            os.mkdir(model_saved_path_ob_up)
            
            
        checkpoint_up = tf.train.get_checkpoint_state(model_saved_path_up)
        if checkpoint_up and checkpoint_up.model_checkpoint_path:
#            saver_up.restore(sess, checkpoint_up.model_checkpoint_path)
            saver_up.restore(sess, checkpoint_up.all_model_checkpoint_paths[2])
            print("Successfully loaded:", checkpoint_up.model_checkpoint_path)
            
        checkpoint_down = tf.train.get_checkpoint_state(model_saved_path_down)
        if checkpoint_down and checkpoint_down.model_checkpoint_path:
#            saver_down.restore(sess, checkpoint_down.model_checkpoint_path)
            saver_down.restore(sess, checkpoint_down.all_model_checkpoint_paths[6])
            print("Successfully loaded:", checkpoint_down.model_checkpoint_path)       
            
        checkpoint_2exits = tf.train.get_checkpoint_state(model_saved_path_2exits)
        if checkpoint_2exits and checkpoint_2exits.model_checkpoint_path:
#            saver_2exits.restore(sess, checkpoint_2exits.model_checkpoint_path)
            saver_2exits.restore(sess, checkpoint_2exits.all_model_checkpoint_paths[1])
            print("Successfully loaded:", checkpoint_2exits.model_checkpoint_path)   
        
        checkpoint_4exits = tf.train.get_checkpoint_state(model_saved_path_4exits)
        if checkpoint_4exits and checkpoint_4exits.model_checkpoint_path:
            saver_4exits.restore(sess, checkpoint_4exits.model_checkpoint_path)
#            saver_4exits.restore(sess, checkpoint_4exits.all_model_checkpoint_paths[3])
            print("Successfully loaded:", checkpoint_4exits.model_checkpoint_path)            
        
        checkpoint_3exits_ob = tf.train.get_checkpoint_state(model_saved_path_3exits_ob)
        if checkpoint_3exits_ob and checkpoint_3exits_ob.model_checkpoint_path:
#            saver_3exits_ob.restore(sess, checkpoint_3exits_ob.model_checkpoint_path)
            saver_3exits_ob.restore(sess, checkpoint_3exits_ob.all_model_checkpoint_paths[2])  
            print("Successfully loaded:", checkpoint_3exits_ob.model_checkpoint_path) 

        checkpoint_ob_center = tf.train.get_checkpoint_state(model_saved_path_ob_center)
        if checkpoint_ob_center and checkpoint_ob_center.model_checkpoint_path:
#            saver_ob_center.restore(sess, checkpoint_ob_center.model_checkpoint_path)
            saver_ob_center.restore(sess, checkpoint_ob_center.all_model_checkpoint_paths[1])  
            print("Successfully loaded:", checkpoint_ob_center.model_checkpoint_path) 

        checkpoint_ob_up = tf.train.get_checkpoint_state(model_saved_path_ob_up)
        if checkpoint_ob_up and checkpoint_ob_up.model_checkpoint_path:
#            saver_ob_up.restore(sess, checkpoint_ob_up.model_checkpoint_path)
            saver_ob_up.restore(sess, checkpoint_ob_up.all_model_checkpoint_paths[9])  
            print("Successfully loaded:", checkpoint_ob_up.model_checkpoint_path) 
        
        ############Illustration of force direction
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

        ypred = sess.run(mainQN_Up.output, feed_dict = {mainQN_Up.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_Up.output, feed_dict = {mainQN_Up.inputs_ : x_arrow_test})
        
        ypred = sess.run(mainQN_Down.output, feed_dict = {mainQN_Down.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_Down.output, feed_dict = {mainQN_Down.inputs_ : x_arrow_test})

        ypred = sess.run(mainQN_2Exits.output, feed_dict = {mainQN_2Exits.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_2Exits.output, feed_dict = {mainQN_2Exits.inputs_ : x_arrow_test})

        ypred = sess.run(mainQN_4Exits.output, feed_dict = {mainQN_4Exits.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_4Exits.output, feed_dict = {mainQN_4Exits.inputs_ : x_arrow_test})
 
        ypred = sess.run(mainQN_3Exits_Ob.output, feed_dict = {mainQN_3Exits_Ob.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_3Exits_Ob.output, feed_dict = {mainQN_3Exits_Ob.inputs_ : x_arrow_test})

        ypred = sess.run(mainQN_Ob_center.output, feed_dict = {mainQN_Ob_center.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_Ob_center.output, feed_dict = {mainQN_Ob_center.inputs_ : x_arrow_test})
     
        ypred = sess.run(mainQN_Ob_up.output, feed_dict = {mainQN_Ob_up.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_Ob_up.output, feed_dict = {mainQN_Ob_up.inputs_ : x_arrow_test})
 
              
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
        
        
        #####2 exits trim
        action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.5)  & (action_pred == 0) )] = 1
        action_pred[( (np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45)  & (action_pred == 4) )] = 5
        action_pred[( (np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.47)  & (action_pred == 4) )] = 3
# 
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                     (xy_arrow[:,0]- 0.5 )**2 ) <0.5) 
                                        &(action_arrow_pred == 0) )] = 1        
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + 
                                     (xy_arrow[:,0]+ 0.5 )**2 ) <0.45) 
                                        &(action_arrow_pred == 4) )] = 5 
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + 
                                     (xy_arrow[:,0]- 0.5 )**2 ) <0.47) 
                                        &(action_arrow_pred == 4) )] = 3     
       
        
        #####4 exits trim
        action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) >0.47) & (xy[:,0] < 0.12)  & (action_pred == 7) )] = 0

        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                     (xy_arrow[:,0]+ 0.5 )**2 ) >0.47) & (xy_arrow[:,0] <0.12)
                                        &(action_arrow_pred == 7) )] = 0        


        #####Ob center trim
        action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.48) & (action_pred == 0) )] = 7
        action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.48) & (action_pred == 0) )] = 1

        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                     (xy_arrow[:,0]+ 0.5 )**2 ) <0.48)
                                        &(action_arrow_pred == 0) )] = 7
    
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                     (xy_arrow[:,0]- 0.5 )**2 ) <0.48)
                                        &(action_arrow_pred == 0) )] = 1       
    


        #####Ob Up trim
        action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45) & (action_pred == 0) )] = 7
        action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) >0.47) &
                       (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.52)& 
                       (action_pred == 1) )] = 0

        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                     (xy_arrow[:,0]+ 0.5 )**2 ) <0.45)
                                        &(action_arrow_pred == 0) )] = 7
    
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                     (xy_arrow[:,0]- 0.5 )**2 ) <0.48)
                                        &(action_arrow_pred == 0) )] = 1      


        #####3 exits 2 ob trim
        action_pred[( (np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45) & (action_pred == 7) )] = 6
        action_pred[( (np.sqrt((xy[:,1]- 0. )**2 + (xy[:,0]- 0.5 )**2 ) <0.26) &
                       (action_pred == 2) )] = 7
#
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + 
                                     (xy_arrow[:,0]+ 0.5 )**2 ) <0.45)
                                        &(action_arrow_pred == 7) )] = 6
##    
        action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0. )**2 + 
                                     (xy_arrow[:,0]- 0.5 )**2 ) <0.26)
                                        &(action_arrow_pred == 2) )] = 7  
       
        action_grid = action_pred.reshape(x.shape)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize = (5,5), subplot_kw = {'xlim' : (-0.5, 0.5),
                               'ylim' : (-0.5, 0.5)})
        
        ####Contour plot
#        c_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#        contour = ax.contourf(x,y,action_grid+0.1,colors = c_map, alpha = 0.8)
        
        contour = ax.contourf(x,y,action_grid+0.1,cmap = plt.cm.get_cmap('rainbow'), alpha = 0.8)       
#        cbar = fig.colorbar(contour, ticks = range(8))
#        cbar.set_label('Force direction')
#        cbar.set_ticklabels(['Up', 'Up-Left', 'Left','Down-Left','Down',
#                             'Down-right', 'Right', 'Up-Right', 'Right'])
    
        
        ###text annotation
#        for idx, p in enumerate(xy_arrow):
#            ax.annotate(str(action_arrow_pred[idx]), xy = p)
    
        ###Arrow
        arrow_len = 0.07
        angle = np.sqrt(2)/2
        arrow_map = {0 : [0, arrow_len], 1: [-angle * arrow_len, angle * arrow_len],
                     2 : [-arrow_len, 0], 3: [-angle * arrow_len, -angle * arrow_len],
                     4 : [0, -arrow_len], 5: [angle * arrow_len, -angle * arrow_len],
                     6 : [arrow_len, 0], 7: [angle * arrow_len, angle * arrow_len],}
        for idx, p in enumerate(xy_arrow):
            ax.annotate('', xy = p, xytext = np.array(arrow_map[action_arrow_pred[idx]])+ p,
                        arrowprops=dict(arrowstyle= '<|-',color='k',lw=1.5))
        
        #        ax.add_patch(plt.Circle((0,0),0.1, alpha = 0.5))
#        ax.add_patch(plt.Circle((0,0.2),0.1, alpha = 0.5))
#        ax.add_patch(plt.Circle((0.3,0.3),0.1, alpha = 0.5))
#        ax.add_patch(plt.Circle((-0.2,0),0.15, alpha = 0.5))
        
        ax.tick_params(labelsize = 'large')
        plt.show()
        
#        fig.savefig('fs1a.png',dpi=600)
        
        ######multi doors and agents
#        import pandas as pd
#        fig2, ax2 = plt.subplots(1,1, figsize=(10,5))
#        a = pd.read_csv('hist_doors_steps.csv', index_col = 0)
#        a.columns = ['1 Exit', '2 Exits', '4 Exits']
#        a.plot.bar(ax = ax2, legend = False)
#        ax2.set_xlabel('Number of agents', fontsize = 'large')
#        ax2.set_ylabel('Time steps', fontsize = 'large')
#        ax2.tick_params(axis = 'x', labelrotation = 0,labelsize = 'large')
#        ax2.tick_params(axis = 'y', labelsize = 'large')
#        ax2.legend(loc = (0.1,0.76), fontsize = 'large')
#        fig2.show()
        
#        fig2.savefig('f5c.png',dpi=600)
        ##########################
        
        #######hist 
#        N_test = 1000
#        hist_optimal = []
#        hist_DQN = []
#        
#        for i in range(N_test):
#            print("Test case : {}".format(i))
#            s = 0
#            state = env.reset()
#            done = False
#            while not done:
#                done = env.step_optimal() 
#                s +=1                          
#            
#            hist_optimal.append(s)
#            
#            s = 0
#            state = env.reset()
#            done = False
#            while not done:
#                done = env.step_all(sess, mainQN_Up, Normalized=True) 
##                done = env.step_all(sess, mainQN_Down, Normalized=True) 
#                s +=1                       
#            hist_DQN.append(s)
#            
#        state = env.reset()
        #########
        
        
        ##########plot hist
#        import seaborn as sns
#        f_op = np.load('hist_Down_80_OP.npz')
#        f_dyna = np.load('hist_Up_80_OP.npz')
#        hist_optimal = f_op[f_op.files[0]] +117.6
#        hist_DQN = f_dyna[f_dyna.files[0]] +117.5    
#        
#        fig1, ax1 = plt.subplots(1,1, figsize = (7,6))
#        sns.distplot(hist_optimal, kde = False, bins = 15, ax = ax1, norm_hist = True, label = 'Social-force model')
#        sns.distplot(hist_DQN, kde = False, bins = 15, ax = ax1, norm_hist = True, label = 'Dyna-Q')
#        ax1.tick_params(labelsize = 'large')        
#        fig1.legend(loc = (0.15,0.76), fontsize = 'medium', edgecolor = None, facecolor = None)

#        
#        f_op = np.load('hist_Down_80_Dyna.npz')
#        f_dyna = np.load('hist_Up_80_Dyna_2.npz')
#        
#        hist_optimal = f_op[f_op.files[0]] +120
#        hist_DQN = f_dyna[f_dyna.files[0]] +121.5
#        
#        fig1, ax1 = plt.subplots(1,1, figsize = (7,6))
#        sns.distplot(hist_optimal, kde = False, bins = 14, ax = ax1, norm_hist = True, label = 'Social-force model')
#        sns.distplot(hist_DQN, kde = False, bins = 13, ax = ax1, norm_hist = True, label = 'Dyna-Q')
#        ax1.tick_params(labelsize = 'large')        
#        fig1.legend(loc = (0.15,0.76), fontsize = 'large', edgecolor = None, facecolor = None)
 
#        fig1.show()
        
#        fig1.savefig('80down.png',dpi=600)
#        plt.show()
        
        ###########
        
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
                #########ALL Particles
#                done = env.step_all(sess, mainQN_Up, Normalized=True)  
#                done = env.step_all(sess, mainQN_Down, Normalized=True) 
#                done = env.step_all(sess, mainQN_2Exits, Normalized=True) 
#                done = env.step_all(sess, mainQN_4Exits, Normalized=True) 
#                done = env.step_all(sess, mainQN_Ob_center, Normalized=True) 
#                done = env.step_all(sess, mainQN_Ob_up, Normalized=True) 
                
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