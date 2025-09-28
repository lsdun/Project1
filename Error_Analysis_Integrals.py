"""
This script is a numerical error analysis of the definite integration methods
applied to the Biot-Savart law for a finite wire segment.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Integrators import riemann_sum, trapezoidal_rule, simpson_rule
from Analytic_Solutions import biotsavart_analytic
from Biot_Savart import biotsavart_integrand

if __name__ == "__main__": # execute only when the script is run directly

    # parameters:
    I = 1.0 # current  (A)
    R = 0.1 # distance from wire (m)
    # fix L for convergence test
    L = 1.0 # length of wire (m)
    a, b = -L/2, L/2 # bounds of integration
    f = biotsavart_integrand(R)

    # compute B at this point:
    B_analytic = biotsavart_analytic(I, R, [L])[0] 

    ns = np.array([20, 40, 80, 160, 320, 640, 1280]) # values of n to test convergence at

    errors_trap = []
    errors_simp = []
    errors_riem = []

    # compute numerical integrals
    for n in ns:
        I_riem = riemann_sum(f, a, b, n)
        I_trap = trapezoidal_rule(f, a, b, n)
        I_simp = simpson_rule(f, a, b, n) if n % 2 == 0 else np.nan

        mu0 = 4*np.pi*1e-7
        constant = mu0 * I / (4*np.pi)
        errors_riem.append(abs(constant * I_riem - B_analytic))
        errors_trap.append(abs(constant * I_trap - B_analytic))
        errors_simp.append(abs(constant * I_simp - B_analytic) if not np.isnan(I_simp) else np.nan)

    # fit slopes using log-log
    valid = ~np.isnan(errors_simp) # simpson's rule only works if n is even 
    slope_simp = np.polyfit(np.log(ns[valid]), np.log(np.array(errors_simp)[valid]), 1)[0] # ignore NaNs
    slope_trap = np.polyfit(np.log(ns), np.log(errors_trap), 1)[0]
    slope_riem = np.polyfit(np.log(ns), np.log(errors_riem), 1)[0]

    # print slope values and compare them to expected values
    print("Trapezoid slope:", slope_trap, "(expect ~ -2)")
    print("Simpson slope:", slope_simp, "(expect ~ -4)")
    print("Riemann slope:", slope_riem, "(expect ~ -1)")

    # plot errors
    plt.figure()
    plt.loglog(ns, errors_riem, "o-", label="Riemann error", color="darkgreen")
    plt.loglog(ns, errors_trap, "o-", label="Trapezoid error", color="darkseagreen")
    plt.loglog(ns[~np.isnan(errors_simp)], np.array(errors_simp)[~np.isnan(errors_simp)], "o-", label="Simpson error", color="olivedrab")
    plt.xlabel("n")
    plt.ylabel("Absolute error (B)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.title("Integration Error Convergence (Biot–Savart)")
    plt.savefig("convergence_biot.png")
    plt.show()
