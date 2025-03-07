import classicHamiltonian
import integrator

import autograd.numpy as np

def integrateHamiltonian(h, initialCoords, timePoints):
    def dynamics_fn(t, coords):
        return h.getTimeDerivative(coords)
    return integrator.integrateSystem(dynamics_fn, initialCoords, timePoints)

if __name__ == '__main__':
    def pendulum(coords):
        q, p = np.split(coords,2)
        H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H
    
    h = classicHamiltonian.Hamiltonian(1, pendulum)
    
    startPoint = np.array((np.pi/2, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    integrator.plotData(t, y, title=f"Pendulum starting at: {startPoint}")
    
    startPoint = np.array((0, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    integrator.plotData(t, y, title=f"Pendulum starting at: {startPoint}")
    
    startPoint = np.array((np.pi, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    integrator.plotData(t, y, title=f"Pendulum starting at: {startPoint}")
    
    startPoint = np.array((0.001, 0))
    t, y = integrateHamiltonian(h, startPoint, np.linspace(0, 10, 1000))
    integrator.plotData(t, y, title=f"Pendulum starting at: {startPoint}")