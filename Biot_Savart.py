"""
This file contains the 1D Biot-Savart law function for a finite straight wire
along the z-axis. 

It returns the integrand function for the magnetic field, 
which can then be integrated using any method from Integrators.py.
"""

import numpy as np

mu0 = 4 * np.pi * 1e-7  # permeability of free space in H/m

def biotsavart_integrand(R):
    """
    Returns a function f(z) representing the Biot-Savart integrand for a
    straight wire along the z-axis at perpendicular distance R.

    Magnetic field is only in the y-direction; the integrand is:
        f(z) = R / (R^2 + z^2)^(3/2)

    Args:
        R (float): Perpendicular distance from wire to observation point (m).

    Returns:
        function: Integrand function for integration.
    """
    # define the integrand
    def f(z):
        return R / (R**2 + z**2)**1.5

    return f
