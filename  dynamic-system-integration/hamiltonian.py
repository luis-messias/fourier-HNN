import autograd
import autograd.numpy as np

class Hamiltonian:
    def __init__(self, degreesOfFreedom, hamiltonianFunc):
        self.dof = degreesOfFreedom
        self.hamiltonian = hamiltonianFunc
        
        # Input Validation
        try:
            coords = np.zeros((2*degreesOfFreedom))
            h = self.hamiltonian(coords)
        except:
            raise("Invalid Hamiltonian")

        assert h.shape[0] == 1

    def getEnergy(self, coords):
        return self.hamiltonian(coords)
    
    def getTimeDerivative(self, coords):
        dH_dCoords = autograd.grad(self.hamiltonian)(coords)
        dH_dQ, dH_dP = np.split(dH_dCoords, 2)
        dCoords_dt = np.concatenate([dH_dP, -dH_dQ], axis=-1)
        return dCoords_dt

if __name__ == '__main__':
    
    def pendulum(coords):
        q, p = np.split(coords,2)
        H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H
    
    h = Hamiltonian(1, pendulum)
    
    point = np.array([0.0,0.0])
    print(f"Point {point}")
    print(f"\tEnergy: {h.getEnergy(point)}")
    print(f"\tTime Derivative: {h.getTimeDerivative(point)}")
    
    point = np.array([np.pi, 0.0])
    print(f"Point {point}")
    print(f"\tEnergy: {h.getEnergy(point)}")
    print(f"\tTime Derivative: {h.getTimeDerivative(point)}")
    
    point = np.array([np.pi/2, 0.0])
    print(f"Point {point}")
    print(f"\tEnergy: {h.getEnergy(point)}")
    print(f"\tTime Derivative: {h.getTimeDerivative(point)}")
    
    point = np.array([-np.pi/2, 0.0])
    print(f"Point {point}")
    print(f"\tEnergy: {h.getEnergy(point)}")
    print(f"\tTime Derivative: {h.getTimeDerivative(point)}")