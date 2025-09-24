"""
This is the main script that produces the results of the integration techniques for 
the damped harmonic oscillator.

"""

import numpy as np
import matplotlib.pyplot as plt

# Import the ODE solvers
from Integrators import euler_loop, rk4_loop
from Damped_Oscillator import damped

# Define parameters to run the integrators
t0 = 0.0 # initial time
t1 = 20.0 # final time
h = 0.05 # time step
S0 = np.array([1.0, 0.0]) # initial state with position and velocity
m = 1.0 # mass
gamma = 0.2 # damping coefficient
k = 1.0 # spring constant
