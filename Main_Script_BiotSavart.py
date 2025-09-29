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
import matplotlib.colors as mcolors
import os

# create a directory for the figures
os.makedirs("figures", exist_ok=True)

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
plt.figure(figsize=(7, 5))
plt.plot(L_values, B_riemann, color="olive", label="Riemann", linestyle='dashed')
plt.plot(L_values, B_trap, color="darkslateblue", label="Trapezoid", linestyle='dashed')
plt.plot(L_values, B_simp, color="teal", label="Simpson", linestyle='dashed')
plt.plot(L_values, B_analytic, color="goldenrod", label="Analytic", linestyle='dashed')
plt.plot(L_values, B_trap_scipy, color="mediumslateblue", label="SciPy Trapezoid", linestyle='dashed')
plt.plot(L_values, B_simp_scipy, color="darkslategrey", label="SciPy Simpson", linestyle='dashed')
plt.xlabel("Wire length L (m)", fontsize=14)
plt.ylabel("Magnetic field B_y (T)", fontsize=14)
plt.title(f"Magnetic field vs wire length at R={R} m", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("figures/biot_vs_length.png")
plt.show()

# verifying the limiting cases where L is very small and L goes to infinity
def compute_B(Ls):
    vals = []
    for L in Ls:
        f = biotsavart_integrand(R)
        vals.append((mu0*I/(4*np.pi)) * simpson_rule(f, -L/2, L/2, n))
    return np.array(vals)

L_small = np.linspace(1, 0.0001, 100)   # limit as L goes to 0
L_large = np.linspace(0.1, 10, 100)   # limit as L goes to infinity

B_small = compute_B(L_small)
B_large = compute_B(L_large)

B_inf = mu0 * I / (2*np.pi*R)  # known formula for B-field of an infinite sraight wire

plt.figure(figsize=(12, 5))

# very small L behavior
plt.subplot(1, 2, 1)
plt.plot(L_small, B_small, label="Numeric Biot–Savart", color="teal", linestyle='dashed')
plt.axhline(0, color="red", linestyle='dashed', label="Zero length limit")
plt.gca().invert_xaxis()
plt.xlabel("Wire length L (m)", fontsize=14)
plt.ylabel("Magnetic field B_y (T)", fontsize=14)
plt.title("Limit $L \\to 0$", fontsize=16)
plt.legend()
plt.grid(True)

# very large L behavior
plt.subplot(1, 2, 2)
plt.plot(L_large, B_large, label="Numeric Biot–Savart", color="teal", linestyle='dashed')
plt.axhline(B_inf, color="red", linestyle='dashed', label="Infinite wire formula")
plt.xlabel("Wire length L (m)", fontsize=14)
plt.ylabel("Magnetic field B_y (T)", fontsize=14)
plt.title("Limit $L \\to \\infty$", fontsize=16)
plt.legend()
plt.grid(True)

plt.savefig("figures/biot_length_limits.png")
plt.tight_layout()
plt.show()

"""
The next part of this script is a numerical error analysis of the definite integration methods
applied to the Biot-Savart law for a finite wire segment.
"""

L_fixed = 1.0 # fixed length of wire (m) for error analysis
a, b = -L_fixed/2, L_fixed/2 # bounds of integration
f_fixed = biotsavart_integrand(R)

# compute B at this point:
B_analytic_fixed = biotsavart_analytic(I, R, [L_fixed])[0]

ns = np.array([20, 40, 80, 160, 320, 640, 1280]) # values of n to test convergence at

errors_trap = []
errors_simp = []
errors_riem = []

constant = mu0 * I / (4*np.pi)

# compute numerical integrals
for n_val in ns:
    I_riem = riemann_sum(f_fixed, a, b, n_val)
    I_trap = trapezoidal_rule(f_fixed, a, b, n_val)
    I_simp = simpson_rule(f_fixed, a, b, n_val) if n_val % 2 == 0 else np.nan

    errors_riem.append(abs(constant * I_riem - B_analytic_fixed))
    errors_trap.append(abs(constant * I_trap - B_analytic_fixed))
    errors_simp.append(abs(constant * I_simp - B_analytic_fixed) if not np.isnan(I_simp) else np.nan) # ignore NaNs
    
# fit slopes using log-log
valid = ~np.isnan(errors_simp) # simpson's rule only works if n is even 
slope_riem = np.polyfit(np.log(ns), np.log(errors_riem), 1)[0]
slope_trap = np.polyfit(np.log(ns), np.log(errors_trap), 1)[0]
slope_simp = np.polyfit(np.log(ns[valid]), np.log(np.array(errors_simp)[valid]), 1)[0] # ignore NaNs

# add reference slopes
ref_n = np.array([ns[0], ns[-1]], dtype=float)

# plot errors
plt.figure()
plt.loglog(ns, errors_riem, "o-", label=f"Riemann error (slope ~ {slope_riem:.2f})", color="darkgreen")
plt.loglog(ns, errors_trap, "o-", label=f"Trapezoid error (slope ~ {slope_trap:.2f})", color="darkseagreen")
plt.loglog(ns[valid], np.array(errors_simp)[valid], "o-", label=f"Simpson error (slope ~ {slope_simp:.2f})", color="olivedrab")

# reference slopes
plt.loglog(ref_n, (ref_n.astype(float))**-1 * errors_riem[0], ls= '--', label="k = -1", color="darkgreen")
plt.loglog(ref_n, (ref_n.astype(float))**-2 * errors_trap[0], ls= '-.', label="k = -2", color="darkseagreen")
plt.loglog(ref_n, (ref_n.astype(float))**-4 * errors_simp[2], ls= ':', label="k = -4", color="olivedrab")

plt.xlabel("n")
plt.ylabel("Absolute error (B)")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.title("Integration Error Convergence (Biot–Savart)")
plt.savefig("figures/convergence_biot.png")
plt.show()