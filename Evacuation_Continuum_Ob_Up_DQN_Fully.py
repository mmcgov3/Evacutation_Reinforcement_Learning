####Training agents Evacuation at 2 exits using Deep Q-Network

import numpy as np
import tensorflow as tf
from collections import deque
import trfl
import os
import shutil
from Continuum_Cellspace import *

Number_Agent = 1
Exit.append( np.array([0.5, 1.0, 0.5]) )  ##Add Up exit
Exit.append( np.array([0.5, 0, 0.5]) )     ##Add Down Exit

Ob1 = []
Ob1.append(np.array([0.5, 0.7, 0.5]))
Ob.append(Ob1)
Ob_size.append(2.0)

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
            
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)
    
output_dir = output_dir + '/Continuum_Ob_Up_DQN_Fully'
model_saved_path = model_saved_path + '/Continuum_Ob_Up_DQN_Fully'    
name_mainQN = 'main_qn_ob_up'    
name_targetQN = 'target_qn_ob_up'

class DQN:
    def __init__(self, name, learning_rate=0.0001, gamma = 0.99,
                 action_size=8, batch_size=20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')  
            self.actions_ = tf.placeholder(tf.int32, [batch_size], name='actions')
            
#            self.is_training = tf.placeholder_with_default(True, shape = (), name = 'is_training')
#            self.keep_prob = 0.5
            
            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn=tf.nn.elu,                    
                    weights_initializer=tf.initializers.he_normal()
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 128)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 64)

            self.output = tf.contrib.layers.fully_connected(self.f3, action_size, activation_fn=None)

            #TRFL way
            self.targetQs_ = tf.placeholder(tf.float32, [batch_size,action_size], name='target')
            self.reward = tf.placeholder(tf.float32,[batch_size],name="reward")
            self.discount = tf.constant(gamma,shape=[batch_size],dtype=tf.float32,name="discount")
      
            #TRFL qlearing
            qloss, q_learning = trfl.qlearning(self.output,self.actions_,self.reward,self.discount,self.targetQs_)
            self.loss = tf.reduce_mean(qloss)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)] 

####Memory replay 
class Memory():
    def __init__(self, max_size = 500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        
        if len(self.buffer) < batch_size:
            return self.buffer
        
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


if __name__ == '__main__':
    
    train_episodes = 10000        # max number of episodes to learn from
    max_steps = 10000                # max steps in an episode
    gamma = 0.999                   # future reward discount

    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
    decay_percentage = 0.5          
    decay_rate = 4/decay_percentage ####exploration decay rate
            
    # Network parameters
    learning_rate = 1e-4         # Q-network learning rate 
    
    # Memory parameters
    memory_size = 1000          # memory capacity
    batch_size = 50                # experience mini-batch size
    pretrain_length = batch_size   # number experiences to pretrain the memory
    
    #target QN
    update_target_every = 1   ###target update frequency
    tau = 0.1                 ###target update factor
    save_step = 1000          ###steps to save the model
    train_step = 1            ###steps to train the model
    
    Cfg_save_freq = 100       ###Cfg save frequency (episode)
    
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut= 1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()
    
    memory = Memory(max_size=memory_size)
        
    tf.reset_default_graph()
    
    ####set mainQN for training and targetQN for updating
    mainQN = DQN(name=name_mainQN, learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    targetQN = DQN(name=name_targetQN,  learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
 
    #TRFL way to update the target network
    target_network_update_ops = trfl.update_target_variables(targetQN.get_qnetwork_variables(),mainQN.get_qnetwork_variables(),tau=tau)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep= 10) 
    
    ######GPU usage fraction
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    #####pretrain load
#    name_list = []
#    t_list = []
#    for t in mainQN.get_qnetwork_variables():
#        name_list.append('main_qn_2exits' + t.name[13:-2])
#        t_list.append(t)
#    
#    var_dict = dict(zip(name_list, t_list))
#    saver_load_main = tf.train.Saver(var_list = var_dict)
#
#    name_list = []
#    t_list = []
#    for t in targetQN.get_qnetwork_variables():
#        name_list.append('main_qn_2exits' + t.name[15:-2])
#        t_list.append(t)
#    
#    var_dict = dict(zip(name_list, t_list))
#    saver_load_target = tf.train.Saver(var_list = var_dict)  
    ##############

    
    with tf.Session(config = config) as sess:
        
#        sess.run(init)
        
        ####check saved model to continue or start from initialiation
        if not os.path.isdir(model_saved_path):
            os.mkdir(model_saved_path)
        
        checkpoint = tf.train.get_checkpoint_state(model_saved_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
#            saver_load_main.restore(sess, checkpoint.all_model_checkpoint_paths[1])
#            saver_load_target.restore(sess, checkpoint.all_model_checkpoint_paths[1])
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            
            print("Removing check point and files")
            for filename in os.listdir(model_saved_path):
                filepath = os.path.join(model_saved_path, filename)
                
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)
                
            print("Done")
            
        else:
            print("Could not find old network weights. Run with the initialization")
            sess.run(init)
            
        ####
        
        step = 0     
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        for ep in range(1, train_episodes+1):
            total_reward = 0
            t = 0
            
            if ep % Cfg_save_freq ==0:
                
                pathdir = os.path.join(output_dir, 'case_' + str(ep) )             
                if not os.path.isdir(pathdir):
                    os.mkdir(pathdir)
                    
                env.save_output(pathdir + '/s.' + str(t))
            
            while t < max_steps:

                ###### Explore or Exploit 
                epsilon = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*ep/train_episodes) 
                feed_state = np.array(state)
                feed_state[:2] = env.Normalization_XY(feed_state[:2])
                
                if np.random.rand() < epsilon:   
                    ####Get random action
                    action= env.choose_random_action()                   
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: feed_state[np.newaxis, :]}
                    Qs = sess.run(mainQN.output, feed_dict=feed)[0]          
                    
                    action_list = [idx for idx, val in enumerate(Qs) if val == np.max(Qs)]                    
                    action = np.random.choice(action_list)
                #######

                next_state, reward, done = env.step(action)
        
                total_reward += reward
                step += 1
                t += 1
                
                feed_next_state = np.array(next_state)
                feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])               
                
                memory.add((feed_state, action, reward, feed_next_state, done))
                
                if done:
                    # Start new episode
                    if ep % Cfg_save_freq ==0:
                        env.save_output(pathdir + '/s.' + str(t))
                    state = env.reset()
                    break

                else:

                    state = next_state
                    
                    if ep % Cfg_save_freq ==0:
                        if t%cfg_save_step ==0:
                            env.save_output(pathdir + '/s.' + str(t))
            
                if len(memory.buffer) == memory_size and t%train_step==0:
                    # Sample mini-batch from memory
                    batch = memory.sample(batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])
                    finish = np.array([each[4] for each in batch])
                    
                    # Train network
                    target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: next_states})
                    ####End state has 0 action values
                    target_Qs[finish == True] = 0.
                                        
                    #TRFL way, calculate td_error within TRFL
                    loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                        feed_dict={mainQN.inputs_: states,
                                                   mainQN.targetQs_: target_Qs,
                                                   mainQN.reward: rewards,
                                                   mainQN.actions_: actions})
                
            if len(memory.buffer) == memory_size:
                print("Episode: {}, Loss: {}, steps per episode: {}".format(ep,loss, t))
                
            if ep % save_step ==0:
                saver.save(sess, os.path.join(model_saved_path, "Evacuation_Continuum_model.ckpt"), global_step = ep)
            
            #update target q network
            if ep % update_target_every ==0:
                sess.run(target_network_update_ops)
            
            
        saver.save(sess, os.path.join(model_saved_path, "Evacuation_Continuum_model.ckpt"), global_step= train_episodes)
 
