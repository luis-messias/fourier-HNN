import autograd.numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

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
    def simpleEDO(t, x):
        k = - 1
        dx = k * x
        return dx
    
    startPoint = np.array([np.pi/2])
    t, y = integrateSystem(simpleEDO, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, title=f"Simple EDO starting at: {startPoint}")