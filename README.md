# Numerical Methods for Physics: Damped Harmonic Oscillator & Biot–Savart Law

This project compares numerical integration techniques for two physics problems with known analytic solutions:

## 1. Damped Harmonic Oscillator ODE problem

- Numerical methods: Explicit Euler, Classical RK4, SciPy RK45.
- Validates numerical solutions against analytic displacement and energy decay.
  
## 2. Magnetic Field of a Finite Wire via Biot–Savart Law (Definite integral problem)

- Numerical methods: Hand-written Riemann, Trapezoid, Simpson; SciPy trapezoid and simpson.
- Validates numerical solutions against analytic field and limiting cases (wire length -> 0 or infinity)

## Repository Structure

Integrators.py                    # Reusable numerical integrators (Euler, RK4, Riemann, Trapezoid, Simpson)
Analytic_Solutions.py             # Solutions for both problems
Damped_Oscillator.py              # Equation for damped oscillator ODE
Biot_Savart.py                    # Functions for magnetic field of a finite wire
Main_Script_DampedOscillator.py   # Runs ODE solvers, plots results, computes error
Main Script_BiotSavart.py         # Runs integration methods, plots results, computes error
figures/                          # Generated figures
report/                           # LaTeX report

## Requirements

- Python 3.9+
- Dependencies:
  - numpy
  - scipy
  - matplotlib

Install with:
`pip install numpy scipy matplotlib`

## Usage

### Damped Harmonic Oscillator

Run:

`python Main_Script_DampedOscillator.py`

This generates:

- Displacement vs. time for numeric and analytic methods.
- Displacement vs. time for overdamped, underdamped, and critically damped
- Energy decay comparison.
- Local and global error plots.
- Error convergence slope plot.

### Biot–Savart Finite Wire

Run:

`python Main_Script_BiotSavart.py`

This generates:

- Magnetic field vs. wire length (numeric vs. analytic).
- Error vs. number of subdivisions n.
- Convergence test log-log plot with slope to show scaling

Figures are saved to the `figures/` directory.

## Validation

### Damped Oscillator:

- Euler diverges unless time step is small.
- RK4 tracks analytic solution closely with larger steps.
- SciPy RK45 adapts step size to maintain error tolerance.
  
### Biot–Savart:

- Riemann: first-order convergence.
- Trapezoid: second-order convergence.
- Simpson: fourth-order convergence.
- Limiting cases correctly reproduce expected values.

## Report
in `report/` directory



