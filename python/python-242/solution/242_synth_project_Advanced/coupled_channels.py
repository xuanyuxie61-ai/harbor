"""
coupled_channels.py  --  Coupled-channels radial equations (thin wrapper)
========================================================================
For the small-scale shell-model calculation of this project the radial
equations decouple by (l, j); we expose a minimal API that invokes the
single-channel solver from radial_fd_solver.
"""
from radial_fd_solver import solve_radial_levels  # noqa: F401


def solve_coupled_channels(orbitals: list, A: int, Z_core: int) -> list:
    """Solve each orbital independently; in a fully coupled calculation the
    channels would be mixed by the residual interaction.
    """
    results = []
    for (n_rad, l, j, is_proton) in orbitals:
        res = solve_radial_levels(A=A, Z_core=Z_core, l_q=l, j_q=j,
                                  is_proton=is_proton, n_levels=n_rad + 1)
        results.append((l, j, res))
    return results
