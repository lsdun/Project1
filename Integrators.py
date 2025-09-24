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
        # append the updated state value to the empty array
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

# Riemann Sum Integral:

def riemann_sum(func, a, b, n):
    """
    Approximate an integral using the midpoint Riemann sum.

    Args:
        func (callable): Function.
        a (float): Lower integration limit.
        b (float): Upper integration limit. 
        n (int): Number of subintervals.

    Returns:
        float: Approximation of the integral.
    """
    dx = (b - a) / n
    total = 0.0
    for i in range(n):
        xi = a + i * dx
        midpoint = xi + dx / 2
        total += func(midpoint)
    R = total * dx
    return R

# Trapezoid Rule Integral:

def trapezoidal_rule(func, a, b, n):
    """
    Approximate an integral using the trapezoidal rule.

    Args:
        func (callable): Function.
        a (float): Lower integration limit.
        b (float): Upper integration limit. 
        n (int): Number of subintervals.

    Returns:
    float: Approximation of the integral.
    """
    x = np.linspace(a, b, n+1) # n subintervals
    y = func(x)
    dx = (b - a) / n
    T = (dx / 2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
    return T

# Simpsons Integral:

def simpson_rule(func, a, b, n, *args):
    """
    Approximate an integral using Simpson's rule.

    Args:
        func (callable): Function.
        a (float): Lower integration limit.
        b (float): Upper integration limit. 
        n (int): Number of subintervals.

    Returns:
    float: Approximation of the integral.
    """
    if n % 2 != 0:
        raise ValueError("n must be even.")

    x = np.linspace(a, b, n+1)
    y = func(x)
    dx = (b - a) / n

    S = (dx/3) * (y[0] + 2*np.sum(y[2:-1:2]) + 4*np.sum(y[1::2]) + y[-1])
    return S
