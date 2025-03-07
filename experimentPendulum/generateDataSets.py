from DynamicSystemIntegrator import ClassicHamiltonian, SystemsIntegrator
import autograd.numpy as np

def get_trajectory(hamiltonian, t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius
    
    t, y, dy = SystemsIntegrator.integrateHamiltonian(hamiltonian, y0, t_eval)
    q, p = y[0], y[1]
    dq, dp = np.split(dy,2)

    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dq, dp, t_eval

def get_dataset(hamiltonian, seed=0, samples=50, test_split=0.5):
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(hamiltonian)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        input("")
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

if __name__ == '__main__':
    def hamiltonian_fn(coords):
        q, p = np.split(coords,2)
        H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H
    
    h = ClassicHamiltonian.Hamiltonian(1, hamiltonian_fn)
    
    data = get_dataset(h)
    print(data)
