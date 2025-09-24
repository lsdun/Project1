"""
This is the main script that produces the results of the integration techniques for 
the damped harmonic oscillator.

"""

# Import plotting packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# Perform the integration methods

"""
Running the Explicit Euler Method and 4th Order Runge Kutta integrations. 

The arguments for these loops are the function we are integrating (the damped oscillator),
the intial state vector, the initial and final time, and the time step.

The output is an array of time values from the initial to final time with 
a step size equal to the time step and an array of the updated state.
"""
t_vals, S_vals_euler = euler_loop(damped, S0, t0, t1, h)
t_vals, S_vals_rk4 = rk4_loop(damped, S0, t0, t1, h)

def energy (S, m, k):
    """
    Calculating the total mechanical energy:
    E = 1/2 * mv^2 + 1/2 * kx^2

    Args:
        S (ndarray): Current state vector.
        m (float): Mass.
        k (float): Spring Constant. 

    Returns:
        ndarray: State vector at next step. 

    """
    # need position and velocity values to calculate energy
    x = S[:, 0] # position values are the first column of the state array
    v = S[:, 1] # velocity values are the second column of the state array

    return 0.5 * m * v**2 + 0.5 * k * x**2

# calculate the energy for both ODE methods
E_euler = energy(S_vals_euler, m, k)
E_rk4 = energy(S_vals_rk4, m, k)

# plot energy
plt.figure(figsize=(10, 7))
plt.plot(t_vals, E_euler, label='Euler Energy', color="mediumvioletred")
plt.plot(t_vals, E_rk4, label='RK4 Energy', color="teal")
plt.xlabel("Time [s]", fontsize=15)
plt.ylabel('Total Energy E(t) [J]', fontsize=15)
plt.title('Damped Harmonic Oscillator Energy Decay', fontsize=20)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()