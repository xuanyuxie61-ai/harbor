"""
turbulent_spectrum.py
=====================

Generate and characterise turbulent velocity fields in a protogalactic
gas cloud.  The turbulent energy spectrum is integrated via Gauss-Laguerre
quadrature, directly lifting the exactness-testing routines of the
laguerre_exactness seed project (639) into an astrophysical context.

Key physics
-----------
The turbulent velocity power spectrum in the interstellar medium is
commonly modelled as a broken power law

    E(k) = A k^{-alpha} exp(-(k / k_diss)^2)

where

    alpha = 5/3   (Kolmogorov 1941, subsonic incompressible)
    alpha = 2     (Burgers 1974, highly compressible / shock-dominated)

and k_diss is the dissipation wavenumber set by viscosity or numerical
diffusion.  The velocity dispersion is

    sigma^2 = int_0^infty E(k) dk.

Since the domain of integration is [0, infty) and the integrand
contains the Gaussian factor exp(-(k/k_diss)^2), Gauss-Laguerre
quadrature (designed for integrals with exp(-x) weight) is a natural
choice.  After the substitution  x = (k/k_diss)^2  the integral
becomes

    sigma^2 = (k_diss / 2) int_0^infty x^{-1/2} E(k_diss sqrt(x))
                                  exp(-x) dx

which is of the exact form for which Gauss-Laguerre is designed.

Exactness testing (laguerre_exactness seed)
-------------------------------------------
An N-point Gauss-Laguerre rule is exact for polynomials of degree
up to 2N - 1 times the exp(-x) weight.  We run an exactness test
analogous to laguerre_exactness: integrating monomials x^m exp(-x)
for m = 0, 1, ..., 2N and comparing the quadrature result with the
exact value Gamma(m + 1) = m!.  Deviations from exactness indicate
either the presence of dissipation-scale modes that are under-resolved
or a turbulent spectrum steeper than the rule can integrate.

References
----------
- Kolmogorov, A. N. 1941, Dokl. Akad. Nauk SSSR 30, 299
- Burgers, J. M. 1974, The Nonlinear Diffusion Equation (D. Reidel)
- McKee, C. F., & Ostriker, E. C. 2007, ARA&A 45, 565 (ISM turbulence)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional

from astro_constants import KILOPARSEC_CGS, DOMAIN_SIZE_KPC


# =====================================================================
#               GAUSS-LAGUERRE NODES AND WEIGHTS
# =====================================================================

def gauss_laguerre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the nodes x_i and weights w_i of the N-point Gauss-Laguerre
    quadrature rule for the integral  int_0^infty f(x) exp(-x) dx.

    Uses the Golub-Welsch algorithm (eigenvalues of the symmetric
    tridiagonal Jacobi matrix).

    For monic Laguerre polynomials the Jacobi matrix has
        diag_j     = 2 j + 1     (j = 0, 1, ..., n-1)
        off_diag_j = j           (j = 1, 2, ..., n-1)

    The weights are w_i = (v_i[0])^2 where v_i is the normalised
    eigenvector, multiplied by Gamma(1) = 1.

    For N-point GL, the rule integrates exactly any polynomial of
    degree <= 2N - 1 against the exp(-x) weight.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    # Build symmetric tridiagonal Jacobi matrix
    diag = 2.0 * np.arange(n) + 1.0
    # sub-diagonal: sqrt(beta_j) with beta_j = j^2  ->  j
    off = np.arange(1, n, dtype=float)
    J = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    # nodes = eigenvalues, weights = (first component of normalised eigvec)^2
    nodes = eigenvalues
    weights = eigenvectors[0, :] ** 2
    return nodes, weights


# =====================================================================
#                 LAGUERRE EXACTNESS TEST (from 639)
# =====================================================================

def laguerre_monomial_exact(n_quad: int, degree_max: int = 30
                            ) -> List[Tuple[int, float, float, float]]:
    """
    Test the exactness of the N-point Gauss-Laguerre rule on monomials
    x^m for m = 0, 1, ..., degree_max.

    The exact integral is

        int_0^infty x^m exp(-x) dx = m! = Gamma(m + 1)

    Returns a list of (degree, quadrature, exact, absolute_error).
    This is the direct astrophysical analogue of the laguerre_exactness
    script: here we use it to verify that the turbulent-spectrum
    integrals are resolved to full precision.
    """
    nodes, weights = gauss_laguerre_nodes_weights(n_quad)
    results = []
    for m in range(degree_max + 1):
        quad_val = float(np.sum(weights * nodes ** m))
        exact = math.gamma(m + 1.0)
        err = abs(quad_val - exact)
        results.append((m, quad_val, exact, err))
    return results


def laguerre_exactness_report(n_quad: int, degree_max: int = 20) -> str:
    """
    Produce a human-readable exactness report.  The rule should be
    exact (to machine precision) for m <= 2n - 1 and show growing
    errors for m >= 2n.
    """
    results = laguerre_monomial_exact(n_quad, degree_max)
    lines = [
        f"Laguerre exactness test: n_quad = {n_quad}, degree_max = {degree_max}",
        f"  (exactness expected up to degree 2 n - 1 = {2 * n_quad - 1})",
        "",
        "  m    quadrature         exact              |error|",
        "  ---  ----------------   ----------------   ----------------",
    ]
    for m, quad, exact, err in results:
        lines.append(f"  {m:3d}  {quad:16.8e}   {exact:16.8e}   {err:16.3e}")
    return "\n".join(lines)


# =====================================================================
#                     TURBULENT POWER SPECTRUM
# =====================================================================

class TurbulentSpectrum:
    """
    Analytic turbulent energy spectrum model

        E(k) = A k^{-alpha} exp(-(k / k_diss)^2)  (k >= k_drive)
        E(k) = 0                                   (k < k_drive)

    with:
        alpha   -- spectral slope (5/3 Kolmogorov, 2 Burgers)
        k_drive -- driving wavenumber (injection scale)
        k_diss  -- dissipation wavenumber (viscous cutoff)
        A       -- normalisation set by sigma^2 = int E(k) dk.
    """

    def __init__(
        self,
        alpha: float = 5.0 / 3.0,
        k_drive_kpc_inv: float = 2.0 * math.pi / 2.0,   # injection at 2 kpc
        k_diss_kpc_inv: float = 2.0 * math.pi / 0.01,   # dissipation at 10 pc
        sigma_kms: float = 10.0,
        box_kpc: float = DOMAIN_SIZE_KPC,
    ) -> None:
        self.alpha = alpha
        self.k_drive = k_drive_kpc_inv
        self.k_diss = k_diss_kpc_inv
        self.sigma_kms = sigma_kms
        self.box_kpc = box_kpc
        self._km_s_to_cgs = 1.0e5
        self.normalisation = self._compute_normalisation()

    def _compute_normalisation(self) -> float:
        """
        Set A so that

            sigma^2 = int_{k_drive}^{infty} E(k) dk

        with sigma = sigma_kms converted to CGS.
        """
        sigma_cgs = self.sigma_kms * self._km_s_to_cgs
        sigma2 = sigma_cgs * sigma_cgs
        # Gauss-Laguerre quadrature for the integral
        n_quad = 64
        nodes, weights = gauss_laguerre_nodes_weights(n_quad)
        # substitution x = (k/k_diss)^2,  k = k_diss sqrt(x), dk = k_diss / (2 sqrt(x)) dx
        # E(k) exp(-x) integrand (without A)
        def integrand_no_A(x):
            x_safe = np.maximum(x, 1.0e-30)
            k = self.k_diss * np.sqrt(x_safe)
            Ek_no_A = np.where(
                k >= self.k_drive,
                k ** (-self.alpha),
                0.0,
            )
            return Ek_no_A * self.k_diss / (2.0 * np.sqrt(x_safe))
        integral = float(np.sum(weights * integrand_no_A(nodes)))
        if integral < 1.0e-60:
            integral = 1.0e-60
        return sigma2 / integral

    # -----------------------------------------------------------------
    #  spectrum evaluation
    # -----------------------------------------------------------------
    def E(self, k_kpc_inv: np.ndarray) -> np.ndarray:
        """Evaluate E(k) in CGS (cm^3 s^{-2} per cm^{-1})."""
        k = np.maximum(k_kpc_inv, 1.0e-60)
        # convert k from kpc^{-1} to cm^{-1}
        k_cgs = k / KILOPARSEC_CGS
        k_diss_cgs = self.k_diss / KILOPARSEC_CGS
        k_drive_cgs = self.k_drive / KILOPARSEC_CGS
        Ek = np.where(
            k_cgs >= k_drive_cgs,
            self.normalisation * (k_cgs ** (-self.alpha))
            * np.exp(-(k_cgs / k_diss_cgs) ** 2),
            0.0,
        )
        return Ek

    def velocity_dispersion_cgs(self) -> float:
        """Return sigma in CGS via Gauss-Laguerre quadrature."""
        n_quad = 64
        nodes, weights = gauss_laguerre_nodes_weights(n_quad)
        x = nodes
        x_safe = np.maximum(x, 1.0e-30)
        k_kpc = self.k_diss * np.sqrt(x_safe)
        Ek = self.E(k_kpc)
        dk_dx = self.k_diss / (2.0 * np.sqrt(x_safe)) / KILOPARSEC_CGS
        integral = float(np.sum(weights * Ek * dk_dx * np.exp(x)))
        return math.sqrt(max(integral, 0.0))

    # -----------------------------------------------------------------
    #  Gaussian random velocity field generation
    # -----------------------------------------------------------------
    def generate_velocity_field_1d(self, n_cells: int, seed: int = 42
                                    ) -> np.ndarray:
        """
        Generate a 1-D turbulent velocity field u(x) with the target
        power spectrum on a periodic domain of length box_kpc.

        Construction:
            u_hat(k) = sqrt(2 E(k) dk) * (xi_1 + i xi_2) / sqrt(2)
            u(x) = IFFT(u_hat)
        """
        rng = np.random.default_rng(seed)
        dx_kpc = self.box_kpc / n_cells
        k_kpc = np.fft.fftfreq(n_cells, d=dx_kpc) * 2.0 * math.pi
        Ek = self.E(np.abs(k_kpc))
        dk = 2.0 * math.pi / self.box_kpc
        amp = np.sqrt(Ek * dk)
        xi1 = rng.standard_normal(n_cells)
        xi2 = rng.standard_normal(n_cells)
        u_hat = amp * (xi1 + 1j * xi2) / math.sqrt(2.0)
        u_hat[0] = 0.0
        u = np.fft.ifft(u_hat).real
        u -= np.mean(u)
        # convert from kpc^{-1} domain to CGS velocity
        u *= self._km_s_to_cgs  # rough scaling; exact is via sigma
        measured_sigma = np.std(u)
        if measured_sigma > 1.0e-30:
            u = u * (self.sigma_kms * self._km_s_to_cgs) / measured_sigma
        return u


# =====================================================================
#                       TURBULENCE DIAGNOSTICS
# =====================================================================

def power_spectrum_estimate(u: np.ndarray, dx_cgs: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the power spectrum P(k) of a 1-D velocity field u via
    the periodogram  P(k) = |FFT(u)|^2 / N.
    """
    n = u.size
    u_hat = np.fft.fft(u)
    Pk = np.abs(u_hat) ** 2 / n
    k = np.fft.fftfreq(n, d=dx_cgs) * 2.0 * math.pi
    # keep only positive k
    pos = k > 0
    return k[pos], Pk[pos]


def spectral_slope_loglog(k: np.ndarray, Pk: np.ndarray,
                            kmin: Optional[float] = None,
                            kmax: Optional[float] = None
                            ) -> float:
    """
    Estimate the spectral slope -alpha from a log-log fit of P(k) ~ k^{-alpha}.
    """
    if kmin is not None:
        mask = k >= kmin
        k = k[mask]
        Pk = Pk[mask]
    if kmax is not None:
        mask = k <= kmax
        k = k[mask]
        Pk = Pk[mask]
    pos = Pk > 0
    if np.sum(pos) < 2:
        return 0.0
    logk = np.log(k[pos])
    logP = np.log(Pk[pos])
    A = np.vstack([np.ones_like(logk), logk]).T
    sol, _res, _rank, _sv = np.linalg.lstsq(A, logP, rcond=None)
    return -sol[1]
