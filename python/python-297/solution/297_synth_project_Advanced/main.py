"""
main.py
========
Unified entry point for the dusty plasma crystal simulation.

Scientific problem:
    Simulation of 2D dusty plasma crystal structure using high-order
    finite difference methods and eigenvalue-based stability analysis.

    In a low-temperature RF discharge plasma, micron-sized dust grains
    acquire large negative charges (~10^3-10^4 electrons) and self-organize
    into crystal structures due to the screened Coulomb (Yukawa) interaction.
    The coupling parameter Gamma determines whether the system forms a
    crystal (Gamma > 137) or liquid (Gamma < 137).

    This simulation package implements:
    1. High-order finite difference operators (2nd-8th order)
    2. Sparse matrix I/O (Harwell-Boeing and Matrix Market formats)
    3. Vandermonde-based charge interpolation (Bjorck-Pereyra algorithm)
    4. Symmetric quadrature rules for Yukawa screening integrals
    5. Hermite polynomial spectral decomposition of grain oscillation modes
    6. Hexagonal lattice geometry with Wigner-Seitz cell analysis
    7. Block Toeplitz dynamical matrix construction and solution
    8. Lagrange potential reconstruction from grain measurements
    9. Dust lattice wave (DLW) propagation with spectral methods
    10. KdV soliton exact solutions for nonlinear benchmarking
    11. Monte Carlo thermodynamics for crystal melting transitions
    12. Eigenvalue-based stability analysis (Born criteria, pseudospectra)

Usage:
    python main.py

No arguments required. The simulation runs with default parameters
appropriate for a laboratory RF discharge dusty plasma experiment.

Output:
    Prints a comprehensive report of all simulation results including:
    - Plasma regime parameters
    - Crystal stability assessment
    - Wave dispersion relations
    - Thermodynamic properties
    - Numerical accuracy benchmarks

Authors: Synthesized from 15 scientific computing projects.
"""

import sys
import os

# Ensure the project directory is on the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


def main():
    """
    Entry point: run the complete dusty plasma crystal simulation.

    This function:
    1. Initializes the physical regime (plasma parameters)
    2. Builds the hexagonal crystal lattice
    3. Validates high-order finite difference stencils
    4. Computes screening integrals and Madelung constants
    5. Performs Vandermonde charge interpolation
    6. Reconstructs the sheath potential via Lagrange basis
    7. Analyzes Hermite spectral modes
    8. Constructs and solves the block Toeplitz dynamical matrix
    9. Performs eigenvalue stability analysis
    10. Simulates dust lattice wave propagation
    11. Benchmarks with KdV soliton exact solutions
    12. Runs Monte Carlo thermodynamics
    13. Tests sparse matrix I/O
    14. Computes trajectory arc lengths

    All results are printed to stdout.
    """
    from dusty_plasma_simulator import DustyPlasmaSimulation

    sim = DustyPlasmaSimulation(output_dir=project_dir)
    results = sim.run(verbose=True)

    return results


if __name__ == "__main__":
    main()
