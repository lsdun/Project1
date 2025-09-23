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
    return S + h * func(t, y)

# Fourth-Order Runge-Kutta Method:

def rk4_step(func, t, S, h):
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
    k1 = func(t, S)
    k2 = func(t + h/2, S + 1/2 * k1 * h)
    k3 = func(t + h/2, S + 1/2 * k2 * h)
    k4 = func(t + h, S + k3 * h)

    return S + (h/6) * (k1 + 2 * k2 + 2 * k3 + k4)
