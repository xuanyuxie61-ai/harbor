"""
shell_potential.py  --  Legacy entry point (redirects to woods_saxon_potential)
==============================================================================
Kept for backward compatibility. New code should import from woods_saxon_potential.
"""
from woods_saxon_potential import (  # noqa: F401
    central_potential,
    spin_orbit_potential,
    coulomb_potential,
    total_single_particle_potential,
    centrifugal_barrier,
)
