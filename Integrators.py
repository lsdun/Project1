"""
This file contains a set of numerical integration methods. 
These functions are resusable for any ODE or integral.

ODE Solvers: Euler's method, 4th order Runge-Kutta 
Definite Integral Solvers: Riemann sum, the trapezoidal rule, Simpson's rule.

"""

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