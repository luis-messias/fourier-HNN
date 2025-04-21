import torch
from experimentDoublePendulum.trainFourierHNN import FourierHNN, MLP
from experimentDoublePendulum.trainFourierHNN2 import FourierHNN2, MLP
from experimentDoublePendulum.trainNaiveFourierHNN import NaiveFourierHNN, MLP
from experimentDoublePendulum.trainHNN import HNN, MLP
from experimentDoublePendulum.trainMLP import MLP
from DynamicSystemIntegrator import Integrator, ClassicHamiltonian
import autograd.numpy as np
from experimentDoublePendulum.generateDataSets import get_trajectory, get_pendulum_dataset_with_cache
import matplotlib.pyplot as plt

import os

def hamiltonian_fn(coords):
    q1, q2, p1, p2 = np.split(coords,4)
    t1_num = p1**2 + 2*(p2**2) - 2*p1*p2*np.cos(q1 - q2)
    t1_denom = 2*(1 + np.sin(q1 - q2)**2)
    t2 = -2*3*np.cos(q1) - 3*np.cos(q2)
    H = t1_num/t1_denom + t2
    return H

hamiltonian = ClassicHamiltonian.Hamiltonian(2, hamiltonian_fn)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def getMLPFile(): 
    scriptPath = os.path.abspath('experimentDoublePendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    label = '-mlp'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    
    model = MLP(4, 200, 4, torch.tanh)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    model.eval()
    return model, label

def getHNNFile(): 
    scriptPath = os.path.abspath('experimentDoublePendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    label = '-hnn'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    
    nn_model = MLP(4, 200, 1, torch.tanh)
    hnn = HNN(4, differentiable_model=nn_model)
    hnn.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    hnn.eval()
    return hnn, label

def getNaiveFourierHNNFromFile(): 
    scriptPath = os.path.abspath('experimentDoublePendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    label = '-NaiveFourierHNN'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    
    nn_model = MLP(6, 200, 1, torch.tanh)
    naiveFourierHNN = NaiveFourierHNN(4, differentiable_model=nn_model)
    naiveFourierHNN.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    naiveFourierHNN.eval()
    return naiveFourierHNN, label

def getFourierHNNFromFile(b_type, forward_inputs, learn_B): 
    scriptPath = os.path.abspath('experimentDoublePendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    
    if b_type == "Basic":
        B_Fourier = torch.eye(4)
    elif b_type == "Gaussian_10_1_":# Gaussian Fourier
        basicGaussianScale = 1
        B_out_dim = 10
        B_Fourier = torch.randn(4, B_out_dim) * basicGaussianScale
    elif b_type == "Positional":
        B_Fourier = torch.cat([torch.eye(4), 2*torch.eye(4), 3*torch.eye(4)], dim=1)

    learnString = "Learn" if learn_B else "DontLearn"
    forwardString = "ForwardInputs" if forward_inputs else "DontForwardInputs"
    label=f"{b_type}_{learnString}_{forwardString}"

    label = f'-FourierHNN-{label}'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    
    fourierHNN = FourierHNN(4, 200, B_Fourier, learn_B, forward_inputs)
    fourierHNN.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    fourierHNN.eval()
    return fourierHNN, label

def getFourierHNN2FromFile(b_type, forward_inputs, learn_B): 
    scriptPath = os.path.abspath('experimentDoublePendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    
    if b_type == "Basic":
        B_Fourier = torch.eye(2)
    elif b_type == "Gaussian_10_1_":# Gaussian Fourier
        basicGaussianScale = 1
        B_out_dim = 10
        B_Fourier = torch.randn(2, B_out_dim) * basicGaussianScale
    elif b_type == "Positional":
        B_Fourier = torch.cat([torch.eye(2), 2*torch.eye(2), 3*torch.eye(2)], dim=1)

    learnString = "Learn" if learn_B else "DontLearn"
    forwardString = "ForwardInputs" if forward_inputs else "DontForwardInputs"
    label=f"{b_type}_{learnString}_{forwardString}"

    label = f'-FourierHNN2-{label}'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    
    fourierHNN2 = FourierHNN2(4, 200, B_Fourier, learn_B, forward_inputs)
    fourierHNN2.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    fourierHNN2.eval()
    return fourierHNN2, label

def getModels():
    models_list = []

    models_list.append(getMLPFile())
    models_list.append(getHNNFile())
    models_list.append(getNaiveFourierHNNFromFile())

    for forward_inputs in (True, False):
        for Learn_B in (True, False):
            for b_type in ("Basic", "Gaussian_10_1_", "Positional"):
                models_list.append(getFourierHNNFromFile(b_type, forward_inputs, Learn_B))
                
    for forward_inputs in (True, False):
        for Learn_B in (True, False):
            for b_type in ("Basic", "Gaussian_10_1_", "Positional"):
                models_list.append(getFourierHNN2FromFile(b_type, forward_inputs, Learn_B))
    return models_list

def integrateSystem(model, y0):
    t_span=[0,10] 
    timescale=15
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    def dynamics_fn(t, coords):
        y = torch.tensor([[coords[0], coords[1], coords[2], coords[3]]], dtype=torch.float32, requires_grad=True)
        dy = model.time_derivative(y).detach()
        return np.array([dy[0][0], dy[0][1], dy[0][2], dy[0][3]])

    t, y_hat, dy_hat = Integrator.integrateSystem(dynamics_fn, y0, t_eval)
    return y_hat, dy_hat

def evaluateModel(model):
    _, _, testDataSet = get_pendulum_dataset_with_cache()
    
    q1 = torch.tensor(testDataSet["q1"], dtype=torch.float32, requires_grad=True)
    q2 = torch.tensor(testDataSet["q2"], dtype=torch.float32, requires_grad=True)
    p1 = torch.tensor(testDataSet["p1"], dtype=torch.float32, requires_grad=True)
    p2 = torch.tensor(testDataSet["p2"], dtype=torch.float32, requires_grad=True)
    dq1 = torch.tensor(testDataSet["dq1"], dtype=torch.float32, requires_grad=True)
    dq2 = torch.tensor(testDataSet["dq2"], dtype=torch.float32, requires_grad=True)
    dp1 = torch.tensor(testDataSet["dp1"], dtype=torch.float32, requires_grad=True)
    dp2 = torch.tensor(testDataSet["dp2"], dtype=torch.float32, requires_grad=True)
    y_test = torch.cat((q1, q2, p1, p2), dim=1)
    dy_test = torch.cat((dq1, dq2, dp1, dp2), dim=1)

    dy_hat_test = model.time_derivative(y_test)
    test_dist = (dy_test - dy_hat_test)**2

    print('Final test loss {:.2e} +/- {:.2e}'
        .format(test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
    
    y_test_trajectory_list = np.split(y_test.detach().numpy(), 3015/45)
    dy_test_trajectory_list = np.split(dy_test.detach().numpy(), 3015/45)

    test_coords_MSE = 0
    energy_MSE = 0
    for (y_test_trajectory, dy_test_trajectory) in zip(y_test_trajectory_list, dy_test_trajectory_list):
        energy_test = hamiltonian.getEnergy(y_test_trajectory.T)
        
        y_hat, dy_hat = integrateSystem(model, y_test_trajectory[0])
        energy_hat = hamiltonian.getEnergy(y_hat)

        test_coords_MSE += ((y_test_trajectory - y_hat.T)**2).mean().item()
        energy_MSE += ((energy_test - energy_hat)**2).mean().item()

    test_coords_MSE /= len(y_test_trajectory_list)
    energy_MSE /= len(y_test_trajectory_list)
    print('Final test loss {:.2e} +/- {:.2e}'
        .format(test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
    print("Coords MSE {:.2e}".format(test_coords_MSE))
    print("Energy MSE {:.2e}".format(energy_MSE))
    return test_dist.mean().item(), test_coords_MSE, energy_MSE

if __name__ == "__main__":
    results = []
    for m in getModels():
        model, label = m
        print("Evaluating ", label)
        MSE, MSE_coords, MSE_energy = evaluateModel(model)
        results.append({"Model": label, "MSE": MSE, "MSE_coords": MSE_coords, "MSE_energy": MSE_energy})
        print("-----------------------------\n")
    
    final = {"MSE": sorted(results, key=lambda d: d['MSE']),
             "MSE_coords": sorted(results, key=lambda d: d['MSE_coords']),
             "MSE_Energy": sorted(results, key=lambda d: d['MSE_energy'])}
    print(results)
    print(final)