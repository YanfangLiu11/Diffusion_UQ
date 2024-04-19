import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS_ALPHA = 0.0001 
TIME_STEPS = 500  # time step for reverse ODE
LEARNING_RATE = 0.001   # learning rate of Neural Network
IT_SIZE = 5000 # iteration size of generating labeled data

# Seed setup for reproducibility
torch.manual_seed(12345678)
np.random.seed(12312414)

# Utility functions
def make_folder(folder):
    """Create the folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print(f"Folder '{folder}' already exists.")

def cond_alpha(t):
    return 1 - (1 - EPS_ALPHA) * t

def cond_beta2(t):
    return EPS_ALPHA + (1 - EPS_ALPHA) * t

def b(t):
    return -(1 - EPS_ALPHA) / cond_alpha(t)

def sigma_sq(t):
    return (1 - EPS_ALPHA) - 2 * b(t) * cond_beta2(t)


def ODE_solver(t_vec, zt, y_sample, x_sample, y0_test, time_steps, y_var_new):
    """Solves the ODE to simulate the diffusion process."""
    log_weight_likelihood = torch.sum(-0.5 * ((y0_test[:, None, :] - y_sample[None, :, :]) ** 2) / y_var_new, dim=2)
    for j in range(time_steps):
        t = t_vec[j + 1]
        dt = t_vec[j] - t_vec[j + 1]
        log_weight_gauss = torch.sum(-0.5 * ((zt[:, None, :] - cond_alpha(t) * x_sample[None, :, :]) ** 2) / cond_beta2(t), dim=2)
        score_gauss = -1.0 * (zt[:, None, :] - cond_alpha(t) * x_sample[None, :, :]) / cond_beta2(t)
        weight_temp = torch.exp(log_weight_gauss + log_weight_likelihood)
        weight = weight_temp / torch.sum(weight_temp, dim=1, keepdims=True)
        score = torch.sum(score_gauss * weight[:, :, None], dim=1)
        zt = zt - (b(t) * zt - 0.5 * sigma_sq(t) * score) * dt
    return zt

def Gen_labeled_data(x_sample, y_sample, zT, y0_train,y_var_new, it_size):
    """
    Generate labeled data using a given sample and initial conditions.
    """
    x_sample = torch.tensor(x_sample).to(DEVICE, dtype=torch.float32)
    y_sample = torch.tensor(y_sample).to(DEVICE, dtype=torch.float32)
    y0_train = torch.tensor(y0_train).to(DEVICE, dtype=torch.float32)
    zT = torch.tensor(zT).to(DEVICE, dtype=torch.float32)
    data_size = y0_train.shape[0]
    xTrain = torch.zeros((data_size, 1)).to(DEVICE, dtype=torch.float32)
    it_n = int(data_size / it_size)
    
    T = 1.0 
    t_vec = torch.linspace(T,0, TIME_STEPS+1).to(DEVICE)
    for jj in range(it_n):
        it_zt = zT[jj * it_size: (jj + 1) * it_size, :]
        it_y0 = y0_train[jj * it_size: (jj + 1) * it_size, :]
        x_temp = ODE_solver(t_vec, it_zt, y_sample, x_sample, it_y0, TIME_STEPS,y_var_new)
        xTrain[jj * it_size: (jj + 1) * it_size, :] = x_temp
        if jj % 10 == 0:
            print('Batch', jj, 'processed')
    return xTrain


class FN_Net(nn.Module):
    """Fully connected neural network model."""
    def __init__(self, input_dim, output_dim, hid_size=100):
        super(FN_Net, self).__init__()
        self.input = nn.Linear(input_dim, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.output = nn.Linear(hid_size, output_dim)
    
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

# Main script
if __name__ == "__main__":
    # Directory for saving results
    savedir = '/path/to/Bimodal/
    make_folder(savedir)

    x_dim = 1
    y_dim = 1
    SAMPLE_SIZE = 80000  # sample size of D_{prior} 
    TRAIN_SIZE = 50000  # size of labeled data
    
    interval_a, interval_b = -2.0, 2.0
    y_var = 0.01
    y_var_new = 0.004
    x_sample = np.linspace(interval_a, interval_b, SAMPLE_SIZE).reshape(-1, 1)
    y_sample = x_sample ** 2 + np.random.normal(0, np.sqrt(y_var), (SAMPLE_SIZE, 1))
    
    # Generate labeled data for training
    selected_row_indices = np.random.permutation(SAMPLE_SIZE)[:TRAIN_SIZE]
    y0_train = y_sample[selected_row_indices]

    savemat(os.path.join(savedir, 'sample_data.mat'), {'x_sample': x_sample, 'y_sample': y_sample})


    zT = np.random.randn(TRAIN_SIZE, x_dim)
    xTrain = Gen_labeled_data(x_sample, y_sample, zT, y0_train,y_var_new, IT_SIZE).detach().cpu().numpy()
    savemat(os.path.join(savedir, 'labeled_data.mat'), {'y0_train': y0_train, 'zT': zT, 'xTrain': xTrain})

    # Neural network training
    FN = FN_Net(y_dim+x_dim, x_dim).to(DEVICE)
    optimizer = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    yTrain = torch.tensor(np.hstack((y0_train, zT)), dtype=torch.float32).to(DEVICE)
    xTrain = torch.tensor(xTrain, dtype=torch.float32).to(DEVICE)
    
    
    # Compute the mean and standard deviation for normalization
    xTrain_mean = xTrain.mean(dim=0, keepdim=True)
    xTrain_std = xTrain.std(dim=0, keepdim=True)
    yTrain_mean = yTrain.mean(dim=0, keepdim=True)
    yTrain_std = yTrain.std(dim=0, keepdim=True)
    
    # Normalize the datasets
    xTrain_normalized = (xTrain - xTrain_mean) / xTrain_std
    yTrain_normalized = (yTrain - yTrain_mean) / yTrain_std

    
    for j in range(10000):
        optimizer.zero_grad()
        pred = FN(yTrain_normalized)
        loss = criterion(pred, xTrain_normalized)
        loss.backward()
        optimizer.step()
        if j % 100 == 0:
            print(f"Epoch {j}: Loss {loss.item()}")
    
    # Save the trained FN
    model_save_path = os.path.join(savedir, 'model_with_norm.pth')
    # Save model state dictionary and normalization parameters
    torch.save({
        'model_state_dict': FN.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'xTrain_mean': xTrain_mean,
        'xTrain_std': xTrain_std,
        'yTrain_mean': yTrain_mean,
        'yTrain_std': yTrain_std
    }, model_save_path)


Nsample_p= 5000
selected_row_indices =  np.random.permutation(SAMPLE_SIZE)[:Nsample_p]
y_test_p = y_sample[selected_row_indices]


zT_p =np.random.randn(Nsample_p,1)
xTrain_p = Gen_labeled_data(x_sample,y_sample,zT_p,y_test_p, y_var_new, IT_SIZE)
xTrain_p = xTrain_p.detach().cpu().numpy()

grid_size = 50
x_grid = np.linspace(np.min(y_test_p), np.max(y_test_p), grid_size)
y_grid = np.linspace(np.min(zT_p), np.max(zT_p), grid_size)

[xx_grid, yy_grid ]= np.meshgrid(x_grid,y_grid)

f_hat = np.zeros((grid_size,grid_size))
for j in range(grid_size):
    for i in range(grid_size):
        xy_pair = torch.tensor( np.hstack((xx_grid[j,i],yy_grid[j,i] ))).to(DEVICE, dtype=torch.float32)
        f_hat[j,i] = ( FN((xy_pair-yTrain_mean)/yTrain_std  ) * xTrain_std + xTrain_mean).detach().cpu().numpy()

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
surf = ax.plot_surface(xx_grid, yy_grid,f_hat)#, cmap='viridis')
plt.tight_layout()
plt.savefig(os.path.join(savedir, f'mesh.png'), dpi=300)
plt.show()

savemat(os.path.join(savedir+'generated_data.mat'), {'y_test_p': y_test_p, 'zT_p': zT_p, 'xTrain_p': xTrain_p, 'xx_grid': xx_grid , 'yy_grid': yy_grid, 'f_hat': f_hat  })


        
        