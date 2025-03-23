from DynamicSystemIntegrator import ClassicHamiltonian, SystemsIntegrator
import autograd.numpy as np
import os

dataSetVariables = ["q1", "q2", "p1", "p2", "dq1", "dq2", "dp1", "dp2"]

def hamiltonian_fn(coords):
    q1, q2, p1, p2 = np.split(coords,4)
    t1_num = p1**2 + 2*(p2**2) - 2*p1*p2*np.cos(q1 - q2)
    t1_denom = 2*(1 + np.sin(q1 - q2)**2)
    t2 = -2*3*np.cos(q1) - 3*np.cos(q2)
    H = t1_num/t1_denom + t2
    return H

def get_trajectory(hamiltonian, t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(4)*2.-1
        if radius is None:
            radius = np.random.rand() + 1.3 # sample a range of radii
        y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    t, y, dy = SystemsIntegrator.integrateHamiltonian(hamiltonian, y0, t_eval)
    # add noise
    y += np.random.randn(*y.shape)*noise_std

    q1, q2, p1, p2 = np.split(y,4)
    dq1, dq2, dp1, dp2 = np.split(dy,4)

    return q1, q2, p1, p2, dq1, dq2, dp1, dp2, t_eval

def get_datasets(hamiltonian, seed=0, samples=75, train_val_test_split=[1/3, 1/3, 1/3]):
    np.random.seed(seed)
    data = {}
    q1, q2, p1, p2, dq1, dq2, dp1, dp2 = [], [], [], [], [], [], [], []
    
    for s in range(samples):
        q1_, q2_, p1_, p2_, dq1_, dq2_, dp1_, dp2_, t_eval = get_trajectory(hamiltonian)
        q1.append(q1_.T)
        q2.append(q2_.T)
        dq1.append(dq1_.T)
        dq2.append(dq2_.T)

        p1.append(p1_.T)
        p2.append(p2_.T)
        dp1.append(dp1_.T)
        dp2.append(dp2_.T)
    
    data['q1'] = np.concatenate(q1)
    data['q2'] = np.concatenate(q2)
    data['dq1'] = np.concatenate(dq1)
    data['dq2'] = np.concatenate(dq2)
    data['p1'] = np.concatenate(p1)
    data['p2'] = np.concatenate(p2)
    data['dp1'] = np.concatenate(dp1)
    data['dp2'] = np.concatenate(dp2)

    data_set_size = len(data['q1'])
    val_i = int( data_set_size * train_val_test_split[0])
    test_i = val_i + int(data_set_size * train_val_test_split[1])
    data_train, data_val, data_test = {"label": "train", "system": "pendulum"}, {"label": "val", "system": "pendulum"}, {"label": "test", "system": "pendulum"}
    for k in dataSetVariables:
        data_train[k], data_val[k], data_test[k] = data[k][:val_i], data[k][val_i:test_i], data[k][test_i:]

    return data_train, data_val, data_test

def get_pendulum_dataset():
    h = ClassicHamiltonian.Hamiltonian(2, hamiltonian_fn)
    return get_datasets(h)

def get_pendulum_dataset_with_cache(forceNew=False):
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    dataSetFolder = os.path.join(scriptPath, "Data")
    
    dataSetIsAvailable = not forceNew
    dataSets = []
    for label in ["train", "val", "test"]:
        dataSet = {"label": label, "system": "double_pendulum"}
        for variable in dataSetVariables:
            path = os.path.join(dataSetFolder, f"{label}_double_pendulum_{variable}.npy")
            dataSetIsAvailable = dataSetIsAvailable and os.path.exists(path)
            if dataSetIsAvailable:
                dataSet[variable] = np.load(path)
        dataSets.append(dataSet)

    if dataSetIsAvailable:
        print("DataSet available, loading from file")
        return dataSets
    else:
        print("DataSet not available, creating a new one")
        dataSets = get_pendulum_dataset()
        for dataSet in dataSets:
            for variable in dataSetVariables:
                np.save(os.path.join(dataSetFolder, f"""{dataSet["label"]}_{dataSet["system"]}_{variable}.npy"""), dataSet[variable])
        return dataSets


if __name__ == '__main__':
    datasets = get_pendulum_dataset_with_cache(forceNew=False)
    for dataSet in get_pendulum_dataset_with_cache():
        print(f"""{dataSet["label"].upper()} dataset for {dataSet["system"]} simulation""")
        for variable in dataSetVariables:
            print(variable, dataSet[variable])
        print()
    
    # h = ClassicHamiltonian.Hamiltonian(2, hamiltonian_fn)
    # q1, q2, p1, p2, dq1, dq2, dp1, dp2, t_eval = (get_trajectory(h))
    
    # import matplotlib.pyplot as plt
    # plt.plot(t_eval, q1.T, label="q1")
    # plt.plot(t_eval, q2.T, label="q2")
    # plt.plot(t_eval, dq1.T, label="dq1")
    # plt.plot(t_eval, dq2.T, label="dq2")
    # plt.plot(t_eval, p1.T, label="p1")
    # plt.plot(t_eval, p2.T, label="p2")
    # plt.plot(t_eval, dp1.T, label="dp1")
    # plt.plot(t_eval, dp2.T, label="dp2")
    # plt.legend()
    # plt.show()