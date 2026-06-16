"""
hamiltonian_builder.py  --  Legacy entry (redirects to sparse_hamiltonian)
=========================================================================
"""
from sparse_hamiltonian import (  # noqa: F401
    build_shell_model_hamiltonian,
    enumerate_seniority_zero_basis,
    single_particle_energies,
)
