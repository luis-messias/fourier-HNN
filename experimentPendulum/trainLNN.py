import generateDataSets

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

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

class LNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model):
        super(LNN, self).__init__()
        self.differentiable_model = differentiable_model

    def forward(self, y):
        lagrangian = self.differentiable_model(y)
        assert lagrangian.dim() == 2 and lagrangian.shape[1] == 1, "Output tensor should have shape [batch_size, 1]"
        return lagrangian

    def time_derivative(self, y):
        q, q_t = y[:, [0]], y[:, [1]]
        
        def l(q, q_t):
            y = torch.cat((q, q_t), dim=1)
            return self.forward(y).sum()

        hessian = torch.func.hessian(l, 1)(q, q_t)
        hessian = torch.diagonal(hessian, offset=0, dim1=0, dim2=2).permute((2,0,1))
        pinv = torch.linalg.pinv(hessian)
        grad_q = torch.func.grad(l, 0)(q, q_t)
        grad_q = torch.reshape(grad_q, [grad_q.shape[0],grad_q.shape[1],1])
        lastTerm = torch.func.jacrev(torch.func.jacrev(l, 1), 0)(q, q_t)
        lastTerm = torch.diagonal(lastTerm, offset=0, dim1=0, dim2=2).permute((2,0,1))
        q_t_reshaped = torch.reshape(q_t, [q_t.shape[0],q_t.shape[1],1])
        d2q_dt = torch.bmm(pinv, (grad_q - torch.bmm(lastTerm, q_t_reshaped)))
        d2q_dt = torch.reshape(d2q_dt, [d2q_dt.shape[0],d2q_dt.shape[1]])
        return d2q_dt
    
class LagrangianDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, type="Train"):
        trainDataSet, valDataSet, testDataSet = generateDataSets.get_pendulum_dataset_with_cache()
        
        if type == "Train":
            self.dataSet = trainDataSet
        elif type == "Val":
            self.dataSet = valDataSet
        elif type == "Test":
            self.dataSet = testDataSet
            
        self.q = torch.tensor(self.dataSet["q"], dtype=torch.float32, requires_grad=True)
        self.dq = torch.tensor(self.dataSet["dq"], dtype=torch.float32, requires_grad=True)
        self.d2q = torch.tensor(self.dataSet["d2q"], dtype=torch.float32, requires_grad=True)
        self.y = torch.cat((self.q, self.dq), dim=1)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.y[idx], self.d2q[idx]

def train(seed=0, hidden_dim=200, learn_rate=1e-3, total_steps=2000, print_every=200, nonlinearity=torch.tanh, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    nn_model = MLP(2, hidden_dim, 1, nonlinearity)
    model = LNN(2, differentiable_model=nn_model)
    
    optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-4)
    lossL2 = nn.MSELoss()
    
    TrainDataloader = DataLoader(LagrangianDataSet(type='Train'), batch_size=800, shuffle=True)
    ValDataloader = DataLoader(LagrangianDataSet(type='Val'), batch_size=800, shuffle=True)
    
    if torch.cuda.is_available():
        print("Cuda Available")
        model.to("cuda")
    
    stats = {'train_loss': [], 'val_loss': []}
    for step in range(total_steps+1):
        start_time = time.time()
        loss = 0
        for i_batch, sample_batched in enumerate(TrainDataloader):
            y_train, d2q_train = sample_batched
            if torch.cuda.is_available():
                y_train = y_train.to('cuda')
                d2q_train = d2q_train.to('cuda')
                
            d2q_hat_train = model.time_derivative(y_train)
            loss_batch = lossL2(d2q_train, d2q_hat_train)
            loss_batch.backward() ; optim.step() ; optim.zero_grad()
            loss += loss_batch.item()
        
        val_loss = loss
        for i_batch, sample_batched in enumerate(ValDataloader):
            y_val, d2q_val = sample_batched
            if torch.cuda.is_available():
                y_val = y_val.to('cuda')
                d2q_val = d2q_val.to('cuda')
                
            d2q_hat_val = model.time_derivative(y_val)
            val_loss = lossL2(d2q_val, d2q_hat_val).item()


        # logging
        stats['train_loss'].append(loss)
        stats['val_loss'].append(val_loss)
        if verbose and step % print_every == 0:
            end_time = time.time()
            print("step {}, train_loss {:.4e}, val_loss {:.4e}, time: {}".format(step, loss, val_loss, end_time - start_time))

    # d2q_hat_train = model.time_derivative(y_train)
    # train_dist = (d2q_train - d2q_hat_train)**2
    # d2q_hat_val_hat = model.time_derivative(y_val)
    # val_dist = (d2q_val - d2q_hat_val_hat)**2

    # print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    #     .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
    #             val_dist.mean().item(), val_dist.std().item()/np.sqrt(val_dist.shape[0])))

    return model, stats

if __name__ == "__main__":
    model, stats = train(print_every=50)
    
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    dataSetFolder = os.path.join(scriptPath, "Models")

    label = '-lnn'
    path = '{}/{}{}.tar'.format(dataSetFolder, "pendulum", label)
    torch.save(model.state_dict(), path)