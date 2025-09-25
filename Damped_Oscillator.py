"""
This file contains the function for the damped harmonic oscillator

The equation of motion for the damped harmonic oscillator is d^2(x)/dt^2 = (k/m) * x - gamma * dx/dt,
which introduces the damping force, gamma * dx/dt.

"""

def damped(X, t, m=1.0, gamma=0.2, k=1.0):
    """
   Equation of motion for the damped harmonic oscillator

    Args:
        X (array_like): State vector.
        t (float): Current time.
        m (float): Mass.
        gamma (float): Damping coefficient.
        k (float): Spring constant.

    Returns:
        list: Derivatives of the state vector.
    """
    x, v = X  # unpack state vector
    dxdt = v
    dvdt = -(k/m) * x - (gamma/m) * v
    return [dxdt, dvdt] # pack derivatives
