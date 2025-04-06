import torch
from experimentPendulum.trainFourierHNN import FourierHNN, MLP
from experimentPendulum.trainFourierHNN2 import FourierHNN2, MLP
from experimentPendulum.trainNaiveFourierHNN import NaiveFourierHNN, MLP
from experimentPendulum.trainHNN import HNN, MLP
from experimentPendulum.trainMLP import MLP
from DynamicSystemIntegrator import Integrator, ClassicHamiltonian
import autograd.numpy as np
from experimentPendulum.generateDataSets import get_trajectory, get_pendulum_dataset_with_cache
import matplotlib.pyplot as plt
import os

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
    return H

hamiltonian = ClassicHamiltonian.Hamiltonian(1, hamiltonian_fn)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def getMLPFile(): 
    scriptPath = os.path.abspath('experimentPendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    label = '-mlp'
    path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
    
    model = MLP(2, 200, 2, torch.tanh)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    model.eval()
    return model, label

def getHNNFile(): 
    scriptPath = os.path.abspath('experimentPendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    label = '-hnn'
    path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
    
    nn_model = MLP(2, 200, 1, torch.tanh)
    hnn = HNN(2, differentiable_model=nn_model)
    hnn.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    hnn.eval()
    return hnn, label

def getNaiveFourierHNNFromFile(): 
    scriptPath = os.path.abspath('experimentPendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    label = '-NaiveFourierHNN'
    path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
    
    nn_model = MLP(3, 200, 1, torch.tanh)
    naiveFourierHNN = NaiveFourierHNN(2, differentiable_model=nn_model)
    naiveFourierHNN.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    naiveFourierHNN.eval()
    return naiveFourierHNN, label

def getFourierHNNFromFile(b_type, forward_inputs, learn_B): 
    scriptPath = os.path.abspath('experimentPendulum')
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

    label = f'-FourierHNN-{label}'
    path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
    
    fourierHNN = FourierHNN(2, 200, B_Fourier, learn_B, forward_inputs)
    fourierHNN.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(device)))
    fourierHNN.eval()
    return fourierHNN, label

def getFourierHNN2FromFile(b_type, forward_inputs, learn_B): 
    scriptPath = os.path.abspath('experimentPendulum')
    dataSetFolder = os.path.join(scriptPath, "Models")
    
    if b_type == "Basic":
        B_Fourier = torch.eye(1)
    elif b_type == "Gaussian_10_1_":# Gaussian Fourier
        basicGaussianScale = 1
        B_out_dim = 10
        B_Fourier = torch.randn(1, B_out_dim) * basicGaussianScale
    elif b_type == "Positional":
        B_Fourier = torch.cat([torch.eye(1), 2*torch.eye(1), 3*torch.eye(1)], dim=1)

    learnString = "Learn" if learn_B else "DontLearn"
    forwardString = "ForwardInputs" if forward_inputs else "DontForwardInputs"
    label=f"{b_type}_{learnString}_{forwardString}"

    label = f'-FourierHNN2-{label}'
    path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
    
    fourierHNN2 = FourierHNN2(2, 200, B_Fourier, learn_B, forward_inputs)
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
    t_span=[0,3] 
    timescale=15
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    def dynamics_fn(t, coords):
        y = torch.tensor([[coords[0], coords[1]]], dtype=torch.float32, requires_grad=True)
        dy = model.time_derivative(y).detach()
        return np.array([dy[0][0], dy[0][1]])

    t, y_hat, dy_hat = Integrator.integrateSystem(dynamics_fn, y0, t_eval)
    return y_hat, dy_hat

def evaluateModel(model):
    _, _, testDataSet = get_pendulum_dataset_with_cache()

    q = torch.tensor(testDataSet["q"], dtype=torch.float32, requires_grad=True)
    p = torch.tensor(testDataSet["p"], dtype=torch.float32, requires_grad=True)
    dq = torch.tensor(testDataSet["dq"], dtype=torch.float32, requires_grad=True)
    dp = torch.tensor(testDataSet["dp"], dtype=torch.float32, requires_grad=True)

    y_test  = torch.cat((q, p), dim=1)
    dy_test = torch.cat((dq, dp), dim=1)

    dy_hat_test = model.time_derivative(y_test)
    test_dist = (dy_test - dy_hat_test)**2

    print('Final test loss {:.4e} +/- {:.4e}'
        .format(test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
    
    y_test_trajectory_list = np.split(y_test.detach().numpy(), 1125/45)
    dy_test_trajectory_list = np.split(dy_test.detach().numpy(), 1125/45)

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
    print("Coords MSE", test_coords_MSE)
    print("Energy MSE", energy_MSE)

if __name__ == "__main__":
    for m in getModels():
        model, label = m
        print("Evaluating ", label)
        evaluateModel(model)
        print("-----------------------------\n")