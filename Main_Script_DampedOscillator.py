"""
This is the main script that produces the results of the integration techniques for 
the damped harmonic oscillator.

Methods compared:
- Hand-written explicit Euler Method and 4th Order Runge Kutta integrations
- SciPy ODE solver
- Analytic solution
"""

# Import plotting packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp

# Import the ODE solvers
from Integrators import euler_loop, rk4_loop
from Damped_Oscillator import damped
from Analytic_Solutions import analytic_damped


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

# try different damping values to show under, critical, and overdamping
gamma_values = [0.2, 2.0, 5.0]  # under, critical, over
labels = ["Underdamped", "Critically Damped", "Overdamped"]
colors = ["mediumvioletred", "teal", "mediumseagreen"]

plt.figure(figsize=(10, 7))
for gamma, label, color in zip(gamma_values, labels, colors):
    # redefine damped function with this gamma
    def damped_gamma(X, t, m=1.0, gamma=gamma, k=1.0):
        """
        Rerun the ODE with different damping values gamma and plot displacement vs. time
        """
        x, v = X
        dxdt = v
        dvdt = -(k/m) * x - (gamma/m) * v
        return [dxdt, dvdt]
    
    t_vals, S_vals = rk4_loop(damped_gamma, S0, t0, t1, h)
    plt.plot(t_vals, S_vals[:, 0], label=label, lw =2, color=color)

plt.xlabel("Time [s]", fontsize=15)
plt.ylabel("Displacement x(t)", fontsize=15)
plt.title("Damped Harmonic Oscillator in Different Regimes", fontsize=20)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Regimes.png")
plt.show()

# calculate the analytic solution
x_exact = analytic_damped(t_vals, m, k, gamma, x0=1, v0=0) # position
v_exact = np.gradient(x_exact, h) # derivative of position
t_vals = np.arange(t0, t1 + h, h)

# calculate the energy for the exact solution: 
S_exact = np.column_stack((x_exact, v_exact))
E_exact = energy(S_exact, m, k)

# call SciPy's ODE solver
# solving an initial value problem for an ODE system
sol = solve_ivp(
    lambda t, S: damped(S, t, m=m, gamma=gamma, k=k), # RHS of the ODE
    [t0, t1], # time interval
    S0, # initial condition
    t_eval = t_vals, # times to return solution
    method = 'RK45', #explicit rk method 4th order
)

# extract position and velocity
x_scipy = sol.y[0] # position is the first component
v_scipy = sol.y[1] # velocity is the second component
S_scipy = np.column_stack((x_scipy, v_scipy))
E_scipy = energy(S_scipy, m, k)

# Compare analytic vs numerical solutions by plotting displacement

plt.figure(figsize=(10,6))
plt.plot(t_vals, x_exact, 'k--', lw=3, label='Analytic')
plt.plot(t_vals, S_vals_euler[:, 0], color="mediumvioletred", lw=2, label='Euler')
plt.plot(t_vals, S_vals_rk4[:, 0], color="teal", lw=2, label='RK4')
plt.plot(t_vals, x_scipy, color="skyblue", lw=2, label='SciPy RK45')
plt.xlabel('Time t [s]', fontsize=14)
plt.ylabel('Displacement x(t)', fontsize=14)
plt.title('Damped Harmonic Oscillator: Analytic vs Numeric', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the energy for all 4 methods

plt.figure(figsize=(10,6))
plt.plot(t_vals, E_exact, 'k--', lw=3, label="Analytic")
plt.plot(t_vals, E_euler, color="mediumvioletred", lw=2, label="Euler")
plt.plot(t_vals, E_rk4, color="teal", lw=2, label="RK4")
plt.plot(t_vals, E_scipy, color="skyblue", lw=2, label="SciPy RK45")
plt.xlabel("Time [s]", fontsize=15)
plt.ylabel('Total Energy E(t) [J]', fontsize=15)
plt.title('Damped Harmonic Oscillator Total Mechanical Energy', fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig("TotalEnergy.png")
plt.show()
