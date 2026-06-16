# -*- coding: utf-8 -*-
"""
PROJECT 289 -- 1D slab gyrokinetic turbulence simulator
=======================================================

A self-contained, zero-argument,博士级 numerical laboratory for the study
of ion-temperature-gradient (ITG) turbulence in a magnetised plasma,
using high-order finite differences, implicit time integration and
stability analysis.

The project synthesises the core algorithms of the 15 seed projects
into a coherent physics pipeline:

    physics_constants.py    : physical constants, equilibrium profiles
    velocity_space.py       : Gauss-Patterson quadrature, CVT grids, J_n
    radial_mesh.py          : P1 FEM mesh, triangulations, phase-space I/O
    finite_difference.py    : high-order FD operators, banded R8GB LU
    gyroaverage.py          : FLR operators (Gamma_0, J_0 application)
    collision_operator.py   : Lorentz pitch-angle scattering, Toeplitz
    time_integrator.py      : implicit / IMEX time-advancers
    linear_eigenvalue.py    : slab ITG eigenvalue problem, threshold
    zonal_flow.py           : Rosenbluth-Hinton residual
    turbulence_diagnostics.py : SG filter, PDF, synthetic diagnostics

Run
---
    python main.py

Output
------
A text report is written to stdout and a JSON summary to
``results_summary.json`` summarising:
    - linear stability: top eigenvalues
    - ITG threshold in R/L_Ti
    - zonal-flow residual level
    - turbulent heat-flux statistics
    - stretched-exponential intermittency exponent
"""

from .physics_constants import *           # noqa: F401,F403
from .velocity_space import *              # noqa: F401,F403
from .radial_mesh import *                 # noqa: F401,F403
from .finite_difference import *           # noqa: F401,F403
from .gyroaverage import *                 # noqa: F401,F403
from .collision_operator import *          # noqa: F401,F403
from .time_integrator import *             # noqa: F401,F403
from .linear_eigenvalue import *           # noqa: F401,F403
from .zonal_flow import *                  # noqa: F401,F403
from .turbulence_diagnostics import *      # noqa: F401,F403
