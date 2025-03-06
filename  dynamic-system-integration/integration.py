from hamiltonian import Hamiltonian

import autograd.numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def integrateHamiltonian(h, initialCoords, timePoints):
    def dynamics_fn(t, coords):
        return h.getTimeDerivative(coords)
    return integrateSystem(dynamics_fn, initialCoords, timePoints)

def integrateSystem(dynamics_fn, initialCoords, timePoints):
    spring_ivp = scipy.integrate.solve_ivp(fun=dynamics_fn, t_span=[timePoints[0], timePoints[-1]], y0=initialCoords, t_eval=timePoints, rtol=1e-10)
    return timePoints, spring_ivp['y']

def plotData(t, y, title=None):
    dof = y.shape[0]
    if title is not None:
        plt.title(title)
    for i in range(dof):
        plt.subplot(1,dof,i + 1)
        plt.plot(t, y[i])
    plt.show()
    
if __name__ == '__main__':
    def pendulum(coords):
        q, p = np.split(coords,2)
        H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H
    
    h = Hamiltonian(1, pendulum)
    
    startPoint = np.array((np.pi/2, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, title=f"Pendulum starting at: {startPoint}")
    
    startPoint = np.array((0, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, title=f"Pendulum starting at: {startPoint}")
    
    startPoint = np.array((np.pi, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, title=f"Pendulum starting at: {startPoint}")
    
    startPoint = np.array((0.001, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, title=f"Pendulum starting at: {startPoint}")