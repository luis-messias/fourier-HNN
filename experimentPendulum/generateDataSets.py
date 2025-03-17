from DynamicSystemIntegrator import ClassicHamiltonian, SystemsIntegrator
import autograd.numpy as np
import os

def get_trajectory(hamiltonian, t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius
    
    t, y, dy = SystemsIntegrator.integrateHamiltonian(hamiltonian, y0, t_eval)

    # add noise
    y += np.random.randn(*y.shape)*noise_std

    return y, dy, t_eval

def get_datasets(hamiltonian, seed=0, samples=75, train_val_test_split=[1/3, 1/3, 1/3]):
    np.random.seed(seed)
    data = {}
    hamiltonian_coords_ys, hamiltonian_coords_dys = [], []
    
    for s in range(samples):
        y, dy, t = get_trajectory(hamiltonian)
        hamiltonian_coords_ys.append(y.T)
        hamiltonian_coords_dys.append(dy.T)
        
    data['hamiltonian_coords_ys'] = np.concatenate(hamiltonian_coords_ys)
    data['hamiltonian_coords_dys'] = np.concatenate(hamiltonian_coords_dys).squeeze()

    data_set_size = len(data['hamiltonian_coords_ys'])
    val_i = int( data_set_size * train_val_test_split[0])
    test_i = val_i + int(data_set_size * train_val_test_split[1])
    data_train, data_val, data_test = {"label": "train", "system": "pendulum"}, {"label": "val", "system": "pendulum"}, {"label": "test", "system": "pendulum"}
    for k in ['hamiltonian_coords_ys', 'hamiltonian_coords_dys']:
        data_train[k], data_val[k], data_test[k] = data[k][:val_i], data[k][val_i:test_i], data[k][test_i:]

    return data_train, data_val, data_test

def get_pendulum_dataset():
    def hamiltonian_fn(coords):
        q, p = np.split(coords,2)
        H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H
    
    h = ClassicHamiltonian.Hamiltonian(1, hamiltonian_fn)
    
    return get_datasets(h)

def get_pendulum_dataset_with_cache(forceNew=False):
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    dataSetFolder = os.path.join(scriptPath, "Data")
    
    dataSetIsAvailable = not forceNew
    dataSets = []
    for label in ["train", "val", "test"]:
        dataSet = {"label": label, "system": "pendulum"}
        for variable in ["hamiltonian_coords_ys", "hamiltonian_coords_dys"]:
            path = os.path.join(dataSetFolder, f"{label}_pendulum_{variable}.npy")
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
            np.save(os.path.join(dataSetFolder, f"""{dataSet["label"]}_{dataSet["system"]}_hamiltonian_coords_ys.npy"""), dataSet["hamiltonian_coords_ys"])
            np.save(os.path.join(dataSetFolder, f"""{dataSet["label"]}_{dataSet["system"]}_hamiltonian_coords_dys.npy"""), dataSet["hamiltonian_coords_dys"])
        return dataSets


if __name__ == '__main__':
    for dataSet in get_pendulum_dataset_with_cache():
        print(f"""{dataSet["label"].upper()} dataset for {dataSet["system"]} simulation""")
        print(dataSet["hamiltonian_coords_ys"])
        print(dataSet["hamiltonian_coords_dys"])
        print()