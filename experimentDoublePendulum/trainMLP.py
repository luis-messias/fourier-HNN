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
  
  def time_derivative(self, y):
    h = self.nonlinearity( self.linear1(y) )
    h = self.nonlinearity( self.linear2(h) )
    return self.forward(y)

def train(seed=0, hidden_dim=200, learn_rate=1e-3, total_steps=2000, print_every=200, nonlinearity=torch.tanh, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(4, hidden_dim, 4, nonlinearity)
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
        y_train = y_train.to("cuda")
        dy_train = dy_train.to("cuda")
        y_val = y_val.to("cuda")
        dy_val = dy_val.to("cuda")
        device = "cuda"
    else:
       device = "cpu"
    
    stats = {'train_loss': [], 'test_loss': []}    
    for step in range(total_steps+1):
        
        # train step
        dy_hat_train = model.forward(y_train.to(device))
        loss = lossL2(dy_train, dy_hat_train)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        dy_hat_val_hat = model.forward(y_val)
        test_loss = lossL2(dy_val, dy_hat_val_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if verbose and step % print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    dy_hat_train = model.forward(y_train)
    train_dist = (dy_train - dy_hat_train)**2
    dy_hat_val = model.forward(y_val)
    val_dist = (dy_val - dy_hat_val)**2

    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
        .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
                val_dist.mean().item(), val_dist.std().item()/np.sqrt(val_dist.shape[0])))

    return model, stats

if __name__ == "__main__":
    model, stats = train()
    
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    dataSetFolder = os.path.join(scriptPath, "Models")

    label = '-mlp'
    path = '{}/{}{}.tar'.format(dataSetFolder, "double_pendulum", label)
    torch.save(model.state_dict(), path)