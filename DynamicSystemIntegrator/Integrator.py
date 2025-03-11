import autograd.numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def integrateSystem(dynamics_fn, initialCoords, timePoints):
    spring_ivp = scipy.integrate.solve_ivp(fun=dynamics_fn, t_span=[timePoints[0], timePoints[-1]], y0=initialCoords, t_eval=timePoints, rtol=1e-10)
    dy = np.stack([dynamics_fn(None, y) for y in spring_ivp['y'].T]).T
    return timePoints, spring_ivp['y'], dy

def plotData(t, y, dy=None, title=None):
    dof = y.shape[0]
    if title is not None:
        plt.title(title)
    for i in range(dof):
        plt.subplot(1,dof,i + 1)
        plt.plot(t, y[i], label=f'Y[{i}]')
        plt.subplot(1,dof,i + 1)
        if dy is not None:
            plt.plot(t, dy[i], label=f'dY[{i}]')
            plt.legend(fontsize=7)
    plt.show()
    
if __name__ == '__main__':
    def simpleEDO(t, y):
        k = - 2
        dy = k * y
        return dy
    
    startPoint = np.array([np.pi/2])
    t, y, dy = integrateSystem(simpleEDO, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, dy, title=f"Simple EDO starting at: {startPoint}")
    
    startPoint = np.array([np.pi/2, 1])
    t, y, dy = integrateSystem(simpleEDO, startPoint, np.linspace(0, 10, 1000))
    plotData(t, y, dy, title=f"Simple EDO starting at: {startPoint}")