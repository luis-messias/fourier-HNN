import experimentDoublePendulum.generateDataSets as generateDataSets

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

class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor

    def forward(self, y):
        hamiltonian = self.differentiable_model(y)
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

def train(seed=0, hidden_dim=200, learn_rate=1e-3, total_steps=2000, print_every=200, nonlinearity=torch.tanh, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    nn_model = MLP(4, hidden_dim, 1, nonlinearity)
    model = HNN(4, differentiable_model=nn_model)
    
    optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-4)
    lossL2 = nn.MSELoss()

    trainDataSet, valDataSet, _ = generateDataSets.get_pendulum_dataset_with_cache()
    
    q1 = torch.tensor(trainDataSet["q1"], dtype=torch.float32, requires_grad=True)
    q2 = torch.tensor(trainDataSet["q2"], dtype=torch.float32, requires_grad=True)
    p1 = torch.tensor(trainDataSet["p1"], dtype=torch.float32, requires_grad=True)
    p2 = torch.tensor(trainDataSet["p2"], dtype=torch.float32, requires_grad=True)
    dq1 = torch.tensor(trainDataSet["dq1"], dtype=torch.float32, requires_grad=True)
    dq2 = torch.tensor(trainDataSet["dq2"], dtype=torch.float32, requires_grad=True)
    dp1 = torch.tensor(trainDataSet["dp1"], dtype=torch.float32, requires_grad=True)
    dp2 = torch.tensor(trainDataSet["dp2"], dtype=torch.float32, requires_grad=True)
    y_train = torch.cat((q1, q2, p1, p2), dim=1)
    dy_train = torch.cat((dq1, dq2, dp1, dp2), dim=1)
    
    val_q1 = torch.tensor(valDataSet["q1"], dtype=torch.float32, requires_grad=True)
    val_q2 = torch.tensor(valDataSet["q2"], dtype=torch.float32, requires_grad=True)
    val_p1 = torch.tensor(valDataSet["p1"], dtype=torch.float32, requires_grad=True)
    val_p2 = torch.tensor(valDataSet["p2"], dtype=torch.float32, requires_grad=True)
    val_dq1 = torch.tensor(valDataSet["dq1"], dtype=torch.float32, requires_grad=True)
    val_dq2 = torch.tensor(valDataSet["dq2"], dtype=torch.float32, requires_grad=True)
    val_dp1 = torch.tensor(valDataSet["dp1"], dtype=torch.float32, requires_grad=True)
    val_dp2 = torch.tensor(valDataSet["dp2"], dtype=torch.float32, requires_grad=True)
    y_val = torch.cat((val_q1, val_q2, val_p1, val_p2), dim=1)
    dy_val = torch.cat((val_dq1, val_dq2, val_dp1, val_dp2), dim=1)
    
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
    model, stats = train()
    
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    dataSetFolder = os.path.join(scriptPath, "Models")

    label = '-hnn'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    torch.save(model.state_dict(), path)