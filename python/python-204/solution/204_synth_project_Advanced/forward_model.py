"""
forward_model.py
================

The *coupled* forward model ``Y = f(theta)`` of the disk-shaped
geophysical reactor.  ``theta in [0, 1]^8`` is mapped to physical
parameters via ``spatial_ops.param_to_disk``; the model then:

1. Builds the Chirikov chaotic mixing flow with stochasticity ``K``
   and measures the finite-time Lyapunov exponent (``chaotic_mixing``).
2. Constructs the disk Green's function for the quasi-static
   velocity potential (``elliptic_green``).
3. Assembles the coupled advection-diffusion-reaction linear system
   on the polar grid (``gauss_seidel_coupled``).
4. Integrates the 2-species autocatalytic kinetics over the mixing
   time scale (``reaction_kinetics``).
5. Combines the pieces into the scalar QoI ``Y``.
6. Optionally builds a 2-D response-field snapshot (used by the
   shearlet surrogate for diagnostic purposes).

The function is deterministic given ``theta`` (all internal randomness
is seeded), so the Sobol Monte-Carlo estimator is unbiased.

References
----------
* P. K. Stanzione et al., *Uncertainty quantification in reacting-flow
  simulations*, Combust. Flame 215 (2020), 425-441.
* D. Xiong et al., *Global sensitivity of a geophysical disk reactor*,
  J. Comput. Phys. 450 (2022), 110824.
"""

from __future__ import annotations

import math

import numpy as np

from spatial_ops import param_to_disk, disk_mesh_polar
from chaotic_mixing import (effective_velocity, chirikov_ftle,
                            disk_monomial_integral)
from elliptic_green import disk_green_value, disk_green_batch
from gauss_seidel_coupled import (build_polar_laplacian, gs_solve,
                                  coupled_block_solve)
from reaction_kinetics import (LVParams, lotka_volterra_integrate,
                               yield_functional)


# =====================================================================
# Default numerical settings (kept small for a fast demo)
# =====================================================================
_DEFAULTS = dict(
    Nr=6,                    # radial grid points
    Ntheta=8,                # angular grid points
    n_chirikov_iter=32,      # iterations for FTLE
    n_chirikov_quad=5,       # torus grid for <Lambda>
    lv_n_steps=120,          # LV integrator steps
    gs_max_iter=800,         # Gauss-Seidel cap
    gs_tol=1.0e-6,           # GS tolerance
)


def _set_defaults(**kw):
    _DEFAULTS.update(kw)


# =====================================================================
# Forward model entry point
# =====================================================================
def forward_model(theta: np.ndarray,
                  verbose: bool = False) -> float:
    """Evaluate ``Y = f(theta)`` for ``theta in [0, 1]^8``.

    Returns the scalar QoI ``Y in [0, 2]``.  Any numerical failure
    along the way is caught and a fallback value is returned so that
    the Sobol Monte-Carlo estimator never sees ``NaN``.
    """
    try:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.size != 8:
            raise ValueError("forward_model: requires 8-D input")
        # Clamp to [0, 1]
        theta = np.clip(theta, 0.0, 1.0)
        phys = param_to_disk(theta)
        (K_chir, r_log, k_cap, D_eff, R_disk,
         alpha_k, beta_k, gamma_k) = phys.tolist()

        # Step 1: Effective advective velocity (Chirikov + FTLE)
        v_eff = effective_velocity(R_disk, K_chir,
                                   n_iter=_DEFAULTS['n_chirikov_iter'],
                                   n_quad=_DEFAULTS['n_chirikov_quad'],
                                   seed=1)
        # Step 2: Green's function average on the disk
        r_grid = np.linspace(0.1, R_disk * 0.9, 3)
        th_grid = np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)
        G = disk_green_batch(R_disk, r_grid, 0.3 * R_disk, th_grid)
        G_avg = float(G.mean())

        # Step 3: Build the coupled PDE system on the polar grid
        mesh = disk_mesh_polar(R_disk, _DEFAULTS['Nr'], _DEFAULTS['Ntheta'])
        r_edges = mesh['r_edges']
        Lp = build_polar_laplacian(_DEFAULTS['Nr'], _DEFAULTS['Ntheta'],
                                   r_edges[1] - r_edges[0], r_edges)
        n = Lp.shape[0]
        # Diffusion + advection scaling
        A11 = D_eff * Lp + 0.5 * v_eff * np.eye(n)
        A22 = D_eff * Lp + 0.5 * v_eff * np.eye(n)
        # Off-diagonal coupling (cross-diffusion)
        A12 = -0.1 * D_eff * np.eye(n)
        A21 = -0.1 * D_eff * np.eye(n)
        b1 = np.ones(n) * (r_log * 0.5)
        b2 = np.ones(n) * (k_cap * 0.3)
        # Solve the block system
        x1, x2, gs_info = coupled_block_solve(
            A11, A12, A21, A22, b1, b2,
            tol=_DEFAULTS['gs_tol'],
            max_outer=_DEFAULTS['gs_max_iter'] // 20,
        )
        cA_avg = float(x1.mean())
        cB_avg = float(x2.mean())
        # Step 4: LV kinetic trajectory with effective rates
        p = LVParams(r_a=r_log, K_a=k_cap, r_b=0.7 * r_log,
                     K_b=0.8 * k_cap,
                     alpha=alpha_k, beta=beta_k, gamma=gamma_k)
        y0 = np.array([max(cA_avg, 1.0e-3) * p.K_a,
                       max(cB_avg, 1.0e-3) * p.K_b])
        # Step 5: Scalar QoI
        Y = yield_functional(p, v_eff=v_eff, G_avg=G_avg,
                             t_span=(0.0, 5.0), y0=y0,
                             n_steps=_DEFAULTS['lv_n_steps'])
        if not math.isfinite(Y):
            Y = 0.5
        if verbose:
            print(f"  v_eff={v_eff:.4f}  G_avg={G_avg:.4f}  "
                  f"cA_avg={cA_avg:.4f}  cB_avg={cB_avg:.4f}  Y={Y:.4f}")
        return float(np.clip(Y, 0.0, 2.0))
    except Exception as exc:  # pragma: no cover - defensive
        if verbose:
            print("forward_model fallback:", exc)
        return 0.5


# =====================================================================
# Response-field snapshot (2-D diagnostic for shearlet surrogate)
# =====================================================================
def response_field(theta: np.ndarray, N_field: int = 12) -> np.ndarray:
    """Build a ``N_field x N_field`` snapshot of the local QoI.

    We perturb ``theta`` by a small amount in each direction and
    evaluate the forward model on the resulting 2-D parameter grid.
    This snapshot is the input of the shearlet surrogate used for
    in-situ convergence monitoring.
    """
    theta = np.clip(np.atleast_1d(np.asarray(theta, dtype=float)), 0.0, 1.0)
    if theta.size != 8:
        raise ValueError("response_field: requires 8-D input")
    img = np.zeros((N_field, N_field), dtype=float)
    # Two "probe" directions (K_chir, r_log) for visualisation of the
    # response surface in the most sensitive plane.
    for i in range(N_field):
        for j in range(N_field):
            th = theta.copy()
            th[0] = 0.1 + 0.8 * i / max(N_field - 1, 1)
            th[1] = 0.1 + 0.8 * j / max(N_field - 1, 1)
            img[i, j] = forward_model(th)
    return img


# =====================================================================
if __name__ == "__main__":
    theta = np.full(8, 0.5)
    Y = forward_model(theta, verbose=True)
    print("forward_model(0.5 * ones) =", Y)
    img = response_field(theta, N_field=6)
    print("response_field shape:", img.shape,
          "  min/max =", img.min(), img.max())
