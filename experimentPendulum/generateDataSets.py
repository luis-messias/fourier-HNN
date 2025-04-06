from DynamicSystemIntegrator import ClassicHamiltonian, SystemsIntegrator
import autograd.numpy as np
import os

dataSetVariables = ['q', 'dq', 'd2q', 'p', 'dp']
def get_trajectory(hamiltonian, t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1, radiusMul=1):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
        if radius is None:
            radius = np.random.rand() + 1.3 # sample a range of radii
        y0 = y0 / np.sqrt((y0**2).sum()) * radius * radiusMul## set the appropriate radius
    
    t, y, dy = SystemsIntegrator.integrateHamiltonian(hamiltonian, y0, t_eval)
    # add noise
    y += np.random.randn(*y.shape)*noise_std

    q, p = np.split(y,2)
    dq, dp = np.split(dy,2)
    d2q = dp / 0.5

    return q, dq, d2q, p, dp, t_eval

def get_datasets(hamiltonian, seed=0, samples=75, train_val_test_split=[1/3, 1/3, 1/3], radiusMul=1):
    np.random.seed(seed)
    data = {}
    q, dq, d2q, p, dp = [], [], [], [], []
    
    for s in range(samples):
        q_, dq_, d2q_, p_, dp_, t_eval = get_trajectory(hamiltonian, radiusMul=radiusMul)
        q.append(q_.T)
        dq.append(dq_.T)
        d2q.append(d2q_.T)
        p.append(p_.T)
        dp.append(dp_.T)
    
    data['q'] = np.concatenate(q)
    data['dq'] = np.concatenate(dq)
    data['d2q'] = np.concatenate(d2q)
    data['p'] = np.concatenate(p)
    data['dp'] = np.concatenate(dp)

    data_set_size = len(data['q'])
    val_i = int( data_set_size * train_val_test_split[0])
    test_i = val_i + int(data_set_size * train_val_test_split[1])
    data_train, data_val, data_test = {"label": "train", "system": "pendulum"}, {"label": "val", "system": "pendulum"}, {"label": "test", "system": "pendulum"}
    for k in dataSetVariables:
        data_train[k], data_val[k], data_test[k] = data[k][:val_i], data[k][val_i:test_i], data[k][test_i:]

    return data_train, data_val, data_test

def get_pendulum_dataset(radiusMul=1):
    def hamiltonian_fn(coords):
        q, p = np.split(coords,2)
        H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H
    
    h = ClassicHamiltonian.Hamiltonian(1, hamiltonian_fn)
    
    return get_datasets(h, radiusMul=radiusMul)

def get_pendulum_dataset_with_cache(forceNew=False, radiusMul=1, customName=""):
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    dataSetFolder = os.path.join(scriptPath, "Data")
    
    dataSetIsAvailable = not forceNew
    dataSets = []
    for label in ["train", "val", "test"]:
        dataSet = {"label": label, "system": "pendulum"}
        for variable in dataSetVariables:
            path = os.path.join(dataSetFolder, f"{label}_pendulum_{variable}{customName}.npy")
            dataSetIsAvailable = dataSetIsAvailable and os.path.exists(path)
            if dataSetIsAvailable:
                dataSet[variable] = np.load(path)
        dataSets.append(dataSet)

    if dataSetIsAvailable:
        print("DataSet available, loading from file")
        return dataSets
    else:
        print("DataSet not available, creating a new one")
        dataSets = get_pendulum_dataset(radiusMul=radiusMul)
        for dataSet in dataSets:
            for variable in dataSetVariables:
                np.save(os.path.join(dataSetFolder, f"""{dataSet["label"]}_{dataSet["system"]}_{variable}{customName}.npy"""), dataSet[variable])
        return dataSets


if __name__ == '__main__':
    datasets = get_pendulum_dataset_with_cache(forceNew=False)
    for dataSet in get_pendulum_dataset_with_cache():
        print(f"""{dataSet["label"].upper()} dataset for {dataSet["system"]} simulation""")
        for variable in dataSetVariables:
            print(variable, dataSet[variable])
        print()
    
    # def hamiltonian_fn(coords):
    #     q, p = np.split(coords,2)
    #     H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
    #     return H
    
    # h = ClassicHamiltonian.Hamiltonian(1, hamiltonian_fn)
    # q, dq, d2q, p, dp, t_eval = (get_trajectory(h))
    
    # import matplotlib.pyplot as plt
    # plt.plot(t_eval, q.T, label="q")
    # plt.plot(t_eval, dq.T, label="dq")
    # plt.plot(t_eval, d2q.T, label="d2q")
    # plt.plot(t_eval, p.T, label="p")
    # plt.plot(t_eval, dp.T, label="dp")
    # plt.legend()
    # plt.show()