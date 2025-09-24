# Analytic_Solutions.py
import numpy as np

def analytic_damped(t, m, k, gamma, x0, v0):
    """
    Analytic solution of the damped harmonic oscillator
    for underdamped, critically damped, and overdamped cases.
    
    Args:
    t (array_like): Time values.
    m (float): Mass.
    k (float): Spring constant.
    gamma (float): Damping coefficient.
    x0 (float): Initial displacement.
    v0 (float): Initial velocity.
    
    Returns:
    x (ndarray): Displacement at times t.
    """
    t = np.array(t)
    omega0 = np.sqrt(k/m) # natural frequency
    alpha = gamma / (2*m) # damping factor
    disc = omega0**2 - alpha**2 # discriminant

    if disc > 0:  # underdamped
        omega_d = np.sqrt(disc)
        C1 = x0
        C2 = (v0 + alpha*x0) / omega_d
        x = np.exp(-alpha*t) * (C1*np.cos(omega_d*t) + C2*np.sin(omega_d*t))

    elif np.isclose(disc, 0):  # critically damped
        C1 = x0
        C2 = v0 + alpha*x0
        x = (C1 + C2*t) * np.exp(-alpha*t)

    else:  # overdamped
        r1 = -alpha + np.sqrt(-disc)
        r2 = -alpha - np.sqrt(-disc)
        A = (v0 - r2*x0) / (r1 - r2)
        B = (r1*x0 - v0) / (r1 - r2)
        x = A*np.exp(r1*t) + B*np.exp(r2*t)

    return x
