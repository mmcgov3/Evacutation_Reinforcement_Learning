"""
plot_contour_arrows.py

Script to load the final DQN model from training and create a contour + arrow plot
using the "old approach" in [-0.5..0.5] domain. It includes commented-out "trim" logic
which you can enable or adapt if desired.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#########################
# Paths / Config
#########################

# This should match the final model that your training script saved, e.g.
# "Evacuation_Continuum_model_ep10000.pth"
MODEL_PATH_FINAL = (
    "./model/Continuum_1Exit_Ob_DQN_CornerSampling_Fully_Pytorch/"
    "Evacuation_Continuum_model_ep10000.pth"
)

# If you want to comment out any saving:
# PLOT_SAVE_PATH = "./output/contour_arrows.png"

# Using the same DQN class as in your training code:
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load final model checkpoint
    if not os.path.isfile(MODEL_PATH_FINAL):
        raise FileNotFoundError(f"Could not find final model checkpoint at {MODEL_PATH_FINAL}")

    checkpoint = torch.load(MODEL_PATH_FINAL, map_location=device)

    net = DQN().to(device)
    if "mainQN_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["mainQN_state_dict"])
    else:
        net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    # Build grid as in the old approach, in [-0.5..0.5] domain
    offset = np.array([0.5, 0.5])
    x_vals = np.linspace(0, 1, 100) - offset[0]  # effectively [-0.5..0.5]
    y_vals = np.linspace(0, 1, 100) - offset[1]  # same
    x, y = np.meshgrid(x_vals, y_vals)
    xy = np.vstack([x.ravel(), y.ravel()]).T

    # Some velocity logic
    vxy = np.zeros_like(xy)
    # e.g. set vy=0.5
    vxy[:,1] = 0.5

    # Combined input: [x, y, vx, vy]
    xtest = np.hstack([xy, vxy])

    # Forward pass
    with torch.no_grad():
        inp = torch.from_numpy(xtest).float().to(device)
        qvals = net(inp).cpu().numpy()
    action_pred = np.argmax(qvals, axis=1)

    # Then a coarser grid for arrow overlay
    x_vals_arrow = np.linspace(0.05, 0.95, 15) - offset[0]
    y_vals_arrow = np.linspace(0.05, 0.95, 15) - offset[1]
    x_arrow, y_arrow = np.meshgrid(x_vals_arrow, y_vals_arrow)
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T

    vxy_arrow = np.zeros_like(xy_arrow)
    vxy_arrow[:,1] = 0.5
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

    with torch.no_grad():
        inp_arr = torch.from_numpy(x_arrow_test).float().to(device)
        qvals_arr = net(inp_arr).cpu().numpy()
    action_arrow_pred = np.argmax(qvals_arr, axis=1)

    # -------------------------------
    # (Optionally) Manual overrides ("trim" logic)
    # Uncomment any region you want to experiment with:

    ### Example: Ob Up trim
    # action_pred[((xy[:,1]-0.5)**2 + (xy[:,0]+0.5)**2 < 0.45**2) & (action_pred==0)] = 7
    # action_pred[(((xy[:,1]-0.5)**2 + (xy[:,0]-0.5)**2)>0.47**2) &
    #             (((xy[:,1]-0.5)**2 + (xy[:,0]-0.5)**2)<0.52**2) & 
    #             (action_pred==1)] = 0
    #
    # action_arrow_pred[(((xy_arrow[:,1]-0.5)**2 + (xy_arrow[:,0]+0.5)**2)<0.45**2) &
    #                   (action_arrow_pred==0)] = 7
    #
    # action_arrow_pred[(((xy_arrow[:,1]-0.5)**2 + (xy_arrow[:,0]-0.5)**2)<0.48**2) &
    #                   (action_arrow_pred==0)] = 1

    # More override blocks can be placed here...
    # End manual overrides
    # -------------------------------

    # Reshape action_pred for contour
    action_grid = action_pred.reshape(x.shape)

    # Plot
    fig, ax = plt.subplots(1,1, figsize=(5,5),
                           subplot_kw={'xlim':(-0.5,0.5), 'ylim':(-0.5,0.5)})
    contour = ax.contourf(x, y, action_grid+0.1, cmap=plt.cm.rainbow, alpha=0.8)

    # Build arrow directions
    arrow_len = 0.07
    angle = np.sqrt(2)/2
    arrow_map = {
        0: [0, arrow_len],  
        1: [-angle*arrow_len, angle*arrow_len],
        2: [-arrow_len, 0],
        3: [-angle*arrow_len, -angle*arrow_len],
        4: [0, -arrow_len],
        5: [angle*arrow_len, -angle*arrow_len],
        6: [arrow_len, 0],
        7: [angle*arrow_len, angle*arrow_len],
    }

    for idx, p in enumerate(xy_arrow):
        dx, dy = arrow_map[action_arrow_pred[idx]]
        ax.annotate('', xy=p, xytext=p + np.array([dx, dy]),
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))

    ax.tick_params(labelsize='large')
    plt.title("Contour + Arrows (Loaded DQN Final Model)")

    # (Optional) save figure to file (commented out):
    # if not os.path.isdir("./output"):
    #     os.mkdir("./output")
    # plot_save_path = os.path.join("./output", "contour_arrows.png")
    # plt.savefig(plot_save_path)

    plt.show()


if __name__ == "__main__":
    main()
