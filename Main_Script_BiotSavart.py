"""
This is the main script for comparing numerical integration techniques
to compute the magnetic field from a finite current-carrying wire using the Biot–Savart law.

Methods compared:
- Hand-written Riemann, Trapezoidal, Simpsons rule
- SciPy trapezoid and simpson
- Analytic solution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Import custom modules
from Biot_Savart import biotsavart_integrand
from Analytic_Solutions import biotsavart_analytic
from Integrators import riemann_sum, trapezoidal_rule, simpson_rule

# constants
mu0 = 4 * np.pi * 1e-7 # permittivity of free space (H/m)
I = 1.0 # current (A)
R = .1 # distance from wire (m)
n = 1000  # number of integration points

L_values = np.linspace(.1, 100, 50)  # wire lengths to evaluate

# computing B-field
B_riemann, B_trap, B_simp = [], [], []
B_trap_scipy, B_simp_scipy = [], []

for L in L_values:
    f = biotsavart_integrand(R)

    # getting hand-written integrators
    B_riemann.append((mu0*I/(4*np.pi)) * riemann_sum(f, -L/2, L/2, n))
    B_trap.append((mu0*I/(4*np.pi)) * trapezoidal_rule(f, -L/2, L/2, n))
    B_simp.append((mu0*I/(4*np.pi)) * simpson_rule(f, -L/2, L/2, n))

    # SciPy integrators
    xvals = np.linspace(-L/2, L/2, n)
    yvals = f(xvals)
    B_trap_scipy.append((mu0*I/(4*np.pi)) * integrate.trapezoid(yvals, xvals)) # traps
    B_simp_scipy.append((mu0*I/(4*np.pi)) * integrate.simpson(yvals, xvals)) # simps

# convert lists to arrays to calculate error
B_riemann = np.array(B_riemann)
B_trap = np.array(B_trap)
B_simp = np.array(B_simp)
B_trap_scipy = np.array(B_trap_scipy)
B_simp_scipy = np.array(B_simp_scipy)

# compute analytic solution
B_analytic = biotsavart_analytic(I, R, L_values)

# plotting B vs. L
plt.figure(figsize=(8, 6))
plt.plot(L_values, B_riemann, color="olive", label="Riemann", linestyle='dashed')
plt.plot(L_values, B_trap, color="darkslateblue", label="Trapezoid", linestyle='dashed')
plt.plot(L_values, B_simp, color="teal", label="Simpson", linestyle='dashed')
plt.plot(L_values, B_analytic, color="goldenrod", label="Analytic", linestyle='dashed')
plt.plot(L_values, B_trap_scipy, color="mediumslateblue", label="SciPy Trapezoid", linestyle='dashed')
plt.plot(L_values, B_simp_scipy, color="darkslategrey", label="SciPy Simpson", linestyle='dashed')
plt.xlabel("Wire length L (m)")
plt.ylabel("Magnetic field B_y (T)")
plt.title(f"Magnetic field vs wire length at R={R} m")
plt.legend()
plt.grid(True)
plt.show()

# plotting error vs. analytic solution

"""
To compare the integration method against the analytic result,
the absolute error is calculated by subtracting the analytic solution
from each numerical method and taking the absolute value.
"""
plt.figure(figsize=(8, 6))
plt.plot(L_values, np.abs(B_riemann - B_analytic), color="olive", label="Riemann error", linestyle='dashed')
plt.plot(L_values, np.abs(B_trap - B_analytic), color="darkslateblue", label="Trapezoid error", linestyle='dashed')
plt.plot(L_values, np.abs(B_simp - B_analytic), color="teal", label="Simpson error", linestyle='dashed')
plt.plot(L_values, np.abs(B_trap_scipy - B_analytic), color="mediumslateblue", label="Trapezoid (SciPy) error", linestyle='dashed')
plt.plot(L_values, np.abs(B_simp_scipy - B_analytic), color="darkslategrey", label="Simpson (SciPy) error", linestyle='dashed')
plt.yscale("log")
plt.xlabel("Wire length L (m)")
plt.ylabel("Absolute error (T)")
plt.title("Error vs Wire Length for Numerical Integration")
plt.legend()
plt.grid(True)
plt.show()

# verifying the limiting cases where L is very small and L goes to infinity
def compute_B(Ls):
    vals = []
    for L in Ls:
        f = biotsavart_integrand(R)
        vals.append((mu0*I/(4*np.pi)) * simpson_rule(f, -L/2, L/2, n))
    return np.array(vals)

L_small = np.linspace(1, 0.01, 50)   # limit as L goes to 0
L_large = np.linspace(0.1, 50, 50)   # limit as L goes to infinity

B_small = compute_B(L_small)
B_large = compute_B(L_large)

B_inf = mu0 * I / (2*np.pi*R)  # known formula for B-field of an infinite sraight wire

plt.figure(figsize=(12, 5))

# very small L behavior
plt.subplot(1, 2, 1)
plt.plot(L_small, B_small, label="Numeric Biot–Savart", color="teal", linestyle='dashed')
plt.axhline(0, color="red", linestyle='dashed', label="Zero length limit")
plt.gca().invert_xaxis()
plt.xlabel("Wire length L (m)")
plt.ylabel("Magnetic field B (T)")
plt.title("Limit $L \\to 0$")
plt.legend()
plt.grid(True)

# very large L behavior
plt.subplot(1, 2, 2)
plt.plot(L_large, B_large, label="Numeric Biot–Savart", color="teal", linestyle='dashed')
plt.axhline(B_inf, color="red", linestyle='dashed', label="Infinite wire formula")
plt.xlabel("Wire length L (m)")
plt.ylabel("Magnetic field B (T)")
plt.title("Limit $L \\to \\infty$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
