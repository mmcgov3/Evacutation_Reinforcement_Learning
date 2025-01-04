import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Your provided DQN class
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume you have these saved model paths (just examples; replace with your actual paths)
model_saved_path_up = './model/Continuum_1ExitUp_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
model_saved_path_down = './model/Continuum_1ExitDown_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
model_saved_path_2exits = './model/Continuum_2Exit_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
model_saved_path_4exits = './model/Continuum_4Exit_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
#model_saved_path_3exits_ob = './model/Continuum_3Exits_Ob_DQN_Fully.pth'
model_saved_path_ob_center = './model/Continuum_1Exit_Ob_DQN_Fully_Pytorch/Evacuation_Continuum_model_ep10000.pth'
#model_saved_path_ob_up = './model/Continuum_Ob_Up_DQN_Fully.pth'

# Load the models
checkpoint_1_up = torch.load(model_saved_path_up, map_location=device)
mainQN_Up = DQN().to(device)
mainQN_Up.load_state_dict(checkpoint_1_up['mainQN_state_dict'])

checkpoint_1_down = torch.load(model_saved_path_down, map_location=device)
mainQN_down = DQN().to(device)
mainQN_down.load_state_dict(checkpoint_1_down['mainQN_state_dict'])

checkpoint_2 = torch.load(model_saved_path_2exits, map_location=device)
mainQN_2Exits = DQN().to(device)
mainQN_2Exits.load_state_dict(checkpoint_2['mainQN_state_dict'])

checkpoint_4 = torch.load(model_saved_path_4exits, map_location=device)
mainQN_4Exits = DQN().to(device)
mainQN_4Exits.load_state_dict(checkpoint_4['mainQN_state_dict'])

#mainQN_3Exits_Ob = DQN().to(device)
#mainQN_3Exits_Ob.load_state_dict(torch.load(model_saved_path_3exits_ob, map_location=device))

checkpoint_1_ob_center = torch.load(model_saved_path_ob_center, map_location=device)
mainQN_Ob_center = DQN().to(device)
mainQN_Ob_center.load_state_dict(checkpoint_1_ob_center['mainQN_state_dict'])

#mainQN_Ob_up = DQN().to(device)
#mainQN_Ob_up.load_state_dict(torch.load(model_saved_path_ob_up, map_location=device))

# Recreate the test grids and arrays as in the original code
offset = np.array([0.5,0.5])
x, y = np.meshgrid(np.linspace(0,1,100)-offset[0], np.linspace(0,1,100)-offset[1])
x_arrow, y_arrow = np.meshgrid(np.linspace(0.05,0.95,15)-offset[0], np.linspace(0.05,0.95,15)-offset[1])
xy = np.vstack([x.ravel(), y.ravel()]).T
xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T

vxy = np.random.randn(*xy.shape)*0.
vxy_arrow = np.random.randn(*xy_arrow.shape)*0.

# Constant velocity as in original code
vxy[:,1] = 0.5
vxy_arrow[:,1] = 0.5

xtest = np.hstack([xy, vxy])
x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

# Convert to torch tensors
xtest_t = torch.from_numpy(xtest).float().to(device)
x_arrow_test_t = torch.from_numpy(x_arrow_test).float().to(device)

with torch.no_grad():
    # Example: Using mainQN_Up
    # ypred_up = mainQN_Up(xtest_t).cpu().numpy()
    # ypred_arrow_up = mainQN_Up(x_arrow_test_t).cpu().numpy()
    
    # Similarly for others:
    ypred_down = mainQN_down(xtest_t).cpu().numpy()
    ypred_arrow_down = mainQN_down(x_arrow_test_t).cpu().numpy()
    
    ypred_2exits = mainQN_2Exits(xtest_t).cpu().numpy()
    ypred_arrow_2exits = mainQN_2Exits(x_arrow_test_t).cpu().numpy()
    
    ypred_4exits = mainQN_4Exits(xtest_t).cpu().numpy()
    ypred_arrow_4exits = mainQN_4Exits(x_arrow_test_t).cpu().numpy()
    
    # ypred_3exits_ob = mainQN_3Exits_Ob(xtest_t).cpu().numpy()
    # ypred_arrow_3exits_ob = mainQN_3Exits_Ob(x_arrow_test_t).cpu().numpy()
    
    ypred_ob_center = mainQN_Ob_center(xtest_t).cpu().numpy()
    ypred_arrow_ob_center = mainQN_Ob_center(x_arrow_test_t).cpu().numpy()
    
    # ypred_ob_up = mainQN_Ob_up(xtest_t).cpu().numpy()
    # ypred_arrow_ob_up = mainQN_Ob_up(x_arrow_test_t).cpu().numpy()

# Combine or select one of the model outputs as the original code did:
# In your original code, you repeatedly overwrote 'ypred' and 'ypred_arrow' as you tested different models.
# Now, you can choose which model's output you'd like to plot. Let's say we choose `ypred_up` for demonstration.

action_pred = np.argmax(ypred_ob_center, axis=1)
action_arrow_pred = np.argmax(ypred_arrow_ob_center, axis=1)

###up tirm 0.5 v1
# action_pred[((xy[:,1] >0.3) & (xy[:,0] <-0.3) )] = 6
# action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) >0.55)  & (action_pred == 1) )] = 0
# action_arrow_pred[((xy_arrow[:,1]>0.3) & (xy_arrow[:,0] <-0.3) )] = 6
# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
#                                 (xy_arrow[:,0]- 0.5 )**2 ) >0.55) 
#                                 &(action_arrow_pred == 1) )] = 0

###down tirm 0.5 v1
# action_pred[((xy[:,1]>0) & (xy[:,0] >-0.4) & (xy[:,0] <0))] = 4
# action_arrow_pred[((xy_arrow[:,1]>0) & (xy_arrow[:,0] >-0.4) & (xy_arrow[:,0] <0))] = 4


#####2 exits trim
# action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.5)  & (action_pred == 0) )] = 1
# action_pred[( (np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45)  & (action_pred == 4) )] = 5
# action_pred[( (np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.47)  & (action_pred == 4) )] = 3
# # 
# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
#                                 (xy_arrow[:,0]- 0.5 )**2 ) <0.5) 
#                                 &(action_arrow_pred == 0) )] = 1        
# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + 
#                                 (xy_arrow[:,0]+ 0.5 )**2 ) <0.45) 
#                                 &(action_arrow_pred == 4) )] = 5 
# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + 
#                                 (xy_arrow[:,0]- 0.5 )**2 ) <0.47) 
#                                 &(action_arrow_pred == 4) )] = 3     


# #####4 exits trim
# action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) >0.47) & (xy[:,0] < 0.12)  & (action_pred == 7) )] = 0

# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
#                                 (xy_arrow[:,0]+ 0.5 )**2 ) >0.47) & (xy_arrow[:,0] <0.12)
#                                 &(action_arrow_pred == 7) )] = 0        


# #####Ob center trim
action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.48) & (action_pred == 0) )] = 7
action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.48) & (action_pred == 0) )] = 1

action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                (xy_arrow[:,0]+ 0.5 )**2 ) <0.48)
                                &(action_arrow_pred == 0) )] = 7

action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
                                (xy_arrow[:,0]- 0.5 )**2 ) <0.48)
                                &(action_arrow_pred == 0) )] = 1       



# #####Ob Up trim
# action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45) & (action_pred == 0) )] = 7
# action_pred[( (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) >0.47) &
#                 (np.sqrt((xy[:,1]- 0.5 )**2 + (xy[:,0]- 0.5 )**2 ) <0.52)& 
#                 (action_pred == 1) )] = 0

# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
#                                 (xy_arrow[:,0]+ 0.5 )**2 ) <0.45)
#                                 &(action_arrow_pred == 0) )] = 7

# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0.5 )**2 + 
#                                 (xy_arrow[:,0]- 0.5 )**2 ) <0.48)
#                                 &(action_arrow_pred == 0) )] = 1      


# #####3 exits 2 ob trim
# action_pred[( (np.sqrt((xy[:,1]+ 0.5 )**2 + (xy[:,0]+ 0.5 )**2 ) <0.45) & (action_pred == 7) )] = 6
# action_pred[( (np.sqrt((xy[:,1]- 0. )**2 + (xy[:,0]- 0.5 )**2 ) <0.26) &
#                 (action_pred == 2) )] = 7
# #
# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]+ 0.5 )**2 + 
#                                 (xy_arrow[:,0]+ 0.5 )**2 ) <0.45)
#                                 &(action_arrow_pred == 7) )] = 6
# ##    
# action_arrow_pred[( (np.sqrt((xy_arrow[:,1]- 0. )**2 + 
#                                 (xy_arrow[:,0]- 0.5 )**2 ) <0.26)
#                                 &(action_arrow_pred == 2) )] = 7  


action_grid = action_pred.reshape(x.shape)

fig, ax = plt.subplots(1,1, figsize=(5,5), subplot_kw={'xlim':(-0.5,0.5), 'ylim':(-0.5,0.5)})
contour = ax.contourf(x, y, action_grid+0.1, cmap=plt.cm.get_cmap('rainbow'), alpha=0.8)

arrow_len = 0.07
angle = np.sqrt(2)/2
arrow_map = {
    0 : [0, arrow_len], 
    1: [-angle * arrow_len, angle * arrow_len],
    2 : [-arrow_len, 0],
    3: [-angle * arrow_len, -angle * arrow_len],
    4 : [0, -arrow_len],
    5: [angle * arrow_len, -angle * arrow_len],
    6 : [arrow_len, 0],
    7: [angle * arrow_len, angle * arrow_len],
}

for idx, p in enumerate(xy_arrow):
    ax.annotate('', xy=p, xytext=np.array(arrow_map[action_arrow_pred[idx]]) + p,
                arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))

ax.tick_params(labelsize='large')
plt.show()
