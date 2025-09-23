"""
This file contains a set of numerical integration methods. 
These functions are resusable for any ODE or integral.

ODE Solvers: Euler's method, 4th order Runge-Kutta 
Definite Integral Solvers: Riemann sum, the trapezoidal rule, Simpson's rule.

"""

import numpy as np

# Explicit Euler Method:

def euler_method(func, t, S, h):
    """
   Perform one step of Euler's method for an ODE.

    Args:
        func (callable): Function.
        t (float): Current time.
        S (ndarray): Current state vector.
        h (float): Time step. 

    Returns:
        ndarray: State vector at next step. 
    """
    return S + h * np.array(func(S, t))

def euler_loop(func, S0, t0, t1, h):
    """
    Using the Euler function defined above, loop over a given time interval 
    to perform the integration.

    Args:
        func (callable): Function.
        S0 (array_like): Initial condition vector.
        t0 (float): Initial time.
        t1 (float): Final time.
        h (float): Time step. 

    Returns:
        t_vals (ndarray): Time values.
        S_vals (ndarray): Solutions at each time step.
    """

    t_vals = np.arange(t0, t1 + h, h) # array of time values from initial to final time
    S_vals = np.zeros((len(t_vals), len(S0))) # array with len(t_vals) rows and len(S0) columns

    # Perform the integration for the chosen number of steps:
    S_vals[0] = S0 # initial value of the state vector
    for i in range(len(t_vals) - 1):
        S_vals[i+1] = euler_method(func, t_vals[i], S_vals[i], h)
    return t_vals, S_vals

# Fourth-Order Runge-Kutta Method:

def rk4_method(func, t, S, h):
    """
    Perform one step of the 4th-order Runge-Kutta method for an ODE.

    Args:
        func (callable): Function.
        t (float): Current time.
        S (ndarray): Current state vector.
        h (float): Time step. 

    Returns:
        ndarray: State vector at next step. 
    """
    k1 = np.array(func(S, t))
    k2 = np.array(func(S + 1/2 * k1 * h, t + h/2))
    k3 = np.array(func(S + 1/2 * k2 * h, t + h/2))
    k4 = np.array(func(S + k3 * h, t + h))

    return S + (h/6) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_loop(func, S0, t0, t1, h):
    """
    Using the Runge-Kutta function defined above, loop over a given time interval 
    to perform the integration.

    Args:
        func (callable): Function.
        S0 (array_like): Initial condition vector.
        t0 (float): Initial time.
        t1 (float): Final time.
        h (float): Time step. 

    Returns:
        t_vals (ndarray): Time values.
        S_vals (ndarray): Solutions at each time step.
    """

    t_vals = np.arange(t0, t1 + h, h) # array of time values from initial to final time
    S_vals = np.zeros((len(t_vals), len(S0))) # array with len(t_vals) rows and len(S0) columns

    # Perform the integration for the chosen number of steps:
    S_vals[0] = S0 # initial value of the state vector
    for i in range(len(t_vals) - 1):
        S_vals[i+1] = rk4_method(func, t_vals[i], S_vals[i], h)
    return t_vals, S_vals
