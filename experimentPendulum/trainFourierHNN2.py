import experimentPendulum.generateDataSets as generateDataSets

import torch
from torch import nn
import numpy as np
import os

class MLP(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity=torch.tanh):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) 

    self.nonlinearity = nonlinearity

  def forward(self, y):
    h = self.nonlinearity( self.linear1(y) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)

class FourierHNN2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, B_Fourier, Learn_B_Fourier, Forward_Inputs, assume_canonical_coords=True, nonlinearity=torch.tanh):
        super(FourierHNN2, self).__init__()
        self.B_Fourier = torch.nn.Parameter(B_Fourier)   
        self.B_Fourier.requires_grad = Learn_B_Fourier
        self.forwardInputs = Forward_Inputs
        if Forward_Inputs:  
            mlp_input_dim = B_Fourier.shape[1]*2 + 2
        else:
            mlp_input_dim = B_Fourier.shape[1]*2 + 1
        self.differentiable_model = MLP(mlp_input_dim, hidden_dim, 1, nonlinearity)
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor

    def forward(self, y):
        if self.forwardInputs:
            z = torch.column_stack([y, torch.sin(y[:,0:1] @ self.B_Fourier), torch.cos(y[:,0:1] @ self.B_Fourier)])
        else:
            z = torch.column_stack([torch.sin(y[:,0:1] @ self.B_Fourier), torch.cos(y[:,0:1] @ self.B_Fourier), y[:,1:]])

        hamiltonian = self.differentiable_model(z)
        assert hamiltonian.dim() == 2 and hamiltonian.shape[1] == 1, "Output tensor should have shape [batch_size, 1]"
        return hamiltonian

    def time_derivative(self, y):
        hamiltonian = self.forward(y)
        dH_dx = torch.autograd.grad(hamiltonian.sum(), y, create_graph=True)[0]
        solenoidal_field = dH_dx @ self.M.t()
        return solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

def train(seed=0, hidden_dim=200, learn_rate=1e-3, total_steps=2000, print_every=200, B_Fourier=torch.eye(2), Forward_Inputs=True, Learn_B_Fourier=True, nonlinearity=torch.tanh, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FourierHNN2(2, hidden_dim, B_Fourier, Learn_B_Fourier, Forward_Inputs, nonlinearity=nonlinearity)
    
    optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-4)
    lossL2 = nn.MSELoss()

    trainDataSet, valDataSet, _ = generateDataSets.get_pendulum_dataset_with_cache()
    
    q = torch.tensor(trainDataSet["q"], dtype=torch.float32, requires_grad=True)
    p = torch.tensor(trainDataSet["p"], dtype=torch.float32, requires_grad=True)
    dq = torch.tensor(trainDataSet["dq"], dtype=torch.float32, requires_grad=True)
    dp = torch.tensor(trainDataSet["dp"], dtype=torch.float32, requires_grad=True)
    y_train = torch.cat((q, p), dim=1)
    dy_train = torch.cat((dq, dp), dim=1)
    
    q = torch.tensor(valDataSet["q"], dtype=torch.float32, requires_grad=True)
    p = torch.tensor(valDataSet["p"], dtype=torch.float32, requires_grad=True)
    dq = torch.tensor(valDataSet["dq"], dtype=torch.float32, requires_grad=True)
    dp = torch.tensor(valDataSet["dp"], dtype=torch.float32, requires_grad=True)
    y_val  = torch.cat((q, p), dim=1)
    dy_val = torch.cat((dq, dp), dim=1)
    
    if torch.cuda.is_available():
        print("Cuda Available")
        model.to("cuda")
        model.M = model.M.to("cuda")
        y_train = y_train.to("cuda")
        dy_train = dy_train.to("cuda")
        y_val = y_val.to("cuda")
        dy_val = dy_val.to("cuda")

    stats = {'train_loss': [], 'test_loss': []}
    for step in range(total_steps+1):
        
        # train step
        dy_hat_train = model.time_derivative(y_train)
        loss = lossL2(dy_train, dy_hat_train)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        dy_hat_val_hat = model.time_derivative(y_val)
        test_loss = lossL2(dy_val, dy_hat_val_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if verbose and step % print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    dy_hat_train = model.time_derivative(y_train)
    train_dist = (dy_train - dy_hat_train)**2
    dy_hat_val = model.time_derivative(y_val)
    val_dist = (dy_val - dy_hat_val)**2

    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
        .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
                val_dist.mean().item(), val_dist.std().item()/np.sqrt(val_dist.shape[0])))

    return model, stats

if __name__ == "__main__":
    
    config_list = []
    for forward_inputs in (True, False):
        for Learn_B in (True, False):
            learnString = "Learn" if Learn_B else "DontLearn"
            forwardString = "ForwardInputs" if forward_inputs else "DontForwardInputs"
            
            # Basic fourier
            B_basic = torch.eye(1)
            config_list.append({"B": B_basic, "Learn_B": Learn_B, "Forward_Inputs": forward_inputs, "label": f"Basic_{learnString}_{forwardString}"})
            
            # Gaussian Fourier
            basicGaussianScale = 1
            B_out_dim = 10
            B_gaussian = torch.randn(1, B_out_dim) * basicGaussianScale
            config_list.append({"B": B_gaussian, "Learn_B": Learn_B, "Forward_Inputs": forward_inputs, "label": f"Gaussian_{B_out_dim}_{basicGaussianScale}__{learnString}_{forwardString}"})

            # Positional
            B_Positional = torch.cat([torch.eye(1), 2*torch.eye(1), 3*torch.eye(1)], dim=1)
            config_list.append({"B": B_Positional, "Learn_B": Learn_B, "Forward_Inputs": forward_inputs, "label": f"Positional_{learnString}_{forwardString}"})
    
    for config in config_list:
        print(config)
        model, stats = train(B_Fourier=config["B"], Learn_B_Fourier=config["Learn_B"], Forward_Inputs=config["Forward_Inputs"])
    
        scriptPath = os.path.dirname(os.path.abspath(__file__))
        dataSetFolder = os.path.join(scriptPath, "Models")

        label = f'-FourierHNN2-{config["label"]}'
        path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
        torch.save(model.state_dict(), path)