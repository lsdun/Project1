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
import os

# create a directory for the figures
os.makedirs("figures", exist_ok=True)

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

t_vals = np.arange(t0, t1 + h, h)

"""
Running the Explicit Euler Method and 4th Order Runge Kutta integrations. 

The arguments for these loops are the function we are integrating (the damped oscillator),
the intial state vector, the initial and final time, and the time step.

The output is an array of time values from the initial to final time with 
a step size equal to the time step and an array of the updated state.
"""
t_vals, S_vals_euler = euler_loop(damped, S0, t0, t1, h)
t_vals, S_vals_rk4   = rk4_loop(damped, S0, t0, t1, h)

def energy (S, m, k, gamma=0.0):
    """
    Calculating the total mechanical energy:
    E = 1/2 * mv^2 + 1/2 * kx^2
    
    For damped oscillators, energy decays due to damping.
    """
    x = S[:, 0] # position
    v = S[:, 1] # velocity
    return 0.5 * m * v**2 + 0.5 * k * x**2

# calculate the energy for both ODE methods
E_euler = energy(S_vals_euler, m, k, gamma)
E_rk4 = energy(S_vals_rk4, m, k, gamma)

# try different damping values to show under, critical, and overdamping
gamma_values = [0.2, 2.0, 5.0]  # under, critical, over
labels = ["Underdamped", "Critically Damped", "Overdamped"]
colors = ["mediumvioletred", "teal", "mediumseagreen"]

plt.figure(figsize=(7, 5))
for gamma_loop, label, color in zip(gamma_values, labels, colors):
    # redefine damped function with this gamma
    def damped_gamma(X, t, m=1.0, gamma=gamma_loop, k=1.0):
        x, v = X
        dxdt = v
        dvdt = -(k/m) * x - (gamma/m) * v
        return [dxdt, dvdt]
    
    t_vals, S_vals = rk4_loop(damped_gamma, S0, t0, t1, h)
    plt.plot(t_vals, S_vals[:, 0], label=label, lw =2, color=color)

plt.xlabel("Time [s]", fontsize=10)
plt.ylabel("Displacement x(t)", fontsize=10)
plt.title("Damped Harmonic Oscillator in Different Regimes", fontsize=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/oscillator_differentregimes.png")
plt.show()

# calculate the analytic solution
x_exact = analytic_damped(t_vals, m, k, gamma, x0=1, v0=0) # position
v_exact = np.gradient(x_exact, h) # approximate derivative
S_exact = np.column_stack((x_exact, v_exact))
E_exact = energy(S_exact, m, k, gamma)

# call SciPy's ODE solver
# solving an initial value problem for an ODE system
sol = solve_ivp(
    lambda t, S: damped(S, t, m=m, gamma=gamma, k=k),
    [t0, t1],
    S0,
    t_eval = t_vals,
    method = 'RK45',
)

# extract position and velocity
x_scipy = sol.y[0]
v_scipy = sol.y[1]
S_scipy = np.column_stack((x_scipy, v_scipy))
E_scipy = energy(S_scipy, m, k, gamma)

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
plt.savefig("figures/oscillator_displacement.png")
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

"""
The next part of this script evaluates the local and global errors for the ODEs.

The local error is calculates by err = |x_n - x(t_n)|,
where x_n is the numerical solution at step n and x(t_n) is the exact solution at time t_n.

The global error accounts for the accumulated error over many steps.
It is found by taking the maximum value of the local error, 
since it is the largest deviation of the numerical solution from the exact solution over all steps.

The slope of a log-log plot of the global error versus the step size h for Euler's method should be 1,
since Euler's method is a first-order method, meaning its global error is proportional to h.

The slope of a Runge-Kutta 4th-order (RK4) error plot should be ~ 4, due to the method's fourth-order accuracy.
"""

def compute_errors(h):
    # Euler and RK4
    t_vals, S_vals_euler = euler_loop(damped, S0, t0, t1, h)
    t_vals, S_vals_rk4 = rk4_loop(damped, S0, t0, t1, h)

    # analytic solution for Analytic_Solutions.py
    x_exact = analytic_damped(t_vals, m, k, gamma, x0=1, v0=0)

    # local errors
    err_euler = np.abs(S_vals_euler[:, 0] - x_exact)
    err_rk4   = np.abs(S_vals_rk4[:, 0]   - x_exact)

    # global errors
    global_err_euler = np.max(err_euler)
    global_err_rk4   = np.max(err_rk4)

    return t_vals, err_euler, err_rk4, global_err_euler, global_err_rk4

# local error for a single timestep
h = 0.05
t_vals, err_euler, err_rk4, ge_euler, ge_rk4 = compute_errors(h)

plt.figure(figsize=(7,5))
plt.semilogy(t_vals, err_euler, label="Euler", color="mediumvioletred",)
plt.semilogy(t_vals, err_rk4, label="RK4", color="teal",)
plt.xlabel("t [s]")
plt.ylabel("Local error")
plt.title(f"Local Error vs Time (h = {h})")
plt.legend()
plt.grid(True)
plt.savefig("figures/oscillator_localerror.png")
plt.show()

print(f"Global error (h={h}): Euler = {ge_euler:.3e}, RK4 = {ge_rk4:.3e}")

# global error scaling 
hs = [3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
global_euler = []
global_rk4 = []

for h in hs:
    t_vals, localerr_euler, localerr_rk4, ge_euler, ge_rk4 = compute_errors(h)
    global_euler.append(ge_euler)
    global_rk4.append(ge_rk4)

plt.figure(figsize=(8,6))
plt.loglog(hs, global_euler, "o-", label="Euler", color="mediumvioletred",)
plt.loglog(hs, global_rk4, "s-", label="RK4", color="teal",)
# creating line with slope=1
plt.loglog(hs, hs, ls="--", label="O(h)", color="mediumvioletred",)
# creating line with slope=4:
plt.loglog(hs, [h**4 for h in hs], ls=":", label="O(h^4)", color="teal",) 
plt.xlabel("Step size h", fontsize=10)
plt.ylabel("Global error (max)", fontsize=10)
plt.title("Global Error Scaling", fontsize=15)
plt.legend()
plt.grid(True, which="both")
plt.savefig("figures/oscillator_globalerrorscaling.png")
plt.show()
