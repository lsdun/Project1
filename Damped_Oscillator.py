"""
This file contains the function for the damped harmonic oscillator

The equation of motion for the damped harmonic oscillator is d^2(x)/dt^2 = (k/m) * x - gamma * dx/dt,
which introduces the damping force, gamma * dx/dt.

"""

def damped(X, t, m, gamma, k):
    """
   Perform one step of Euler's method for an ODE.

    Args:
        X (array_like): State vector.
        t (float): Current time.
        m (float): Mass.
        gamma (float): Damping coefficient.
        k (float): Spring constant.

    Returns:
        ndarray: Derivatives of the state vector.
    """
    x, v = X    # unpack variables
    dxdt = v
    dvdt = -(k/m) * x - (gamma/m) * v
    dXdt = [dxdt, dvdt]    # pack derivatives
    return dXdt

# set parameters
m = 1    # mass
gamma = 0.2 # damping coefficient 
k = 1 # spring coefficient 