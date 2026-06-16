"""
plasma_profiles.py — Poloidal pressure p(ψ) and diamagnetic function F(ψ) profiles
for the Grad-Shafranov equilibrium, together with Green's-function special
functions (cosine integral Ci, sine integral Si, incomplete gamma Γ(a,x)).

Scientific background
---------------------
In a toroidally symmetric tokamak the MHD equilibrium satisfies

    ∂²ψ      1 ∂ψ    ∂²ψ
   ------ - ----- + ------ = -μ₀ R² p'(ψ) - F(ψ) F'(ψ)        (GS)
    ∂R²      R ∂R    ∂Z²

with poloidal flux ψ(R,Z), toroidal field function F = R B_φ, and kinetic
pressure p(ψ). The profiles p(ψ) and F(ψ) close the system. We adopt the
standard Cerfon-Freidberg parametrization (six coefficients, normalized to
ψ ∈ [0,1]):

    p'(ψ)  = c₁ [ (1+α) ψ^α - α ψ ]                        (1)
    F F'   = d₁ [ (1+β ) ψ^β  - β  ψ ] + d₂ ψ             (2)

The Green's function for the cylindrical operator L = Δ* in free space
involves complete elliptic integrals; we also supply the related cosine
integral Ci(x) = -∫_x^∞ cos(t)/t dt and sine integral Si(x) = ∫_0^x sin(t)/t dt
that appear when the plasma response is evaluated through the Biot-Savart
kernel in the small-elongation expansion (Shafranov 1966).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Special functions: cosine integral Ci(x), sine integral Si(x),
# incomplete gamma Gamma(a,x) (series/asymptotic hybrid).
# These enter the axisymmetric Green's function of ∇² in toroidal geometry.
# ---------------------------------------------------------------------------

def cosine_integral(x: float) -> float:
    """Ci(x) = γ + ln x + Σ_{k=1}^∞ (-1)^k x^{2k} / [(2k) (2k)!]
    with Euler-Mascheroni γ ≈ 0.5772156649."""
    if x <= 0.0:
        return -math.inf
    gamma_e = 0.5772156649015329
    # For x < 4: power series, otherwise asymptotic expansion
    if x < 4.0:
        s = 0.0
        term = 1.0
        x2 = x * x
        for k in range(1, 60):
            term *= -x2 / ((2 * k - 1) * (2 * k))
            s += term / (2 * k)
            if abs(term / (2 * k)) < 1e-16 * abs(s):
                break
        return gamma_e + math.log(x) + s
    else:
        # Asymptotic: Ci(x) ~ sin(x)/x P(x) - cos(x)/x Q(x)
        f, g = 0.0, 0.0
        tf, tg = 1.0 / x, 1.0 / x
        x2 = x * x
        for k in range(1, 30):
            tf *= -(2 * k - 1) * (2 * k) / x2
            tg *= -(2 * k) * (2 * k + 1) / x2
            f += tf
            g += tg
            if abs(tf) + abs(tg) < 1e-15:
                break
        return math.sin(x) / x * f - math.cos(x) / x * g


def sine_integral(x: float) -> float:
    """Si(x) = Σ_{k=0}^∞ (-1)^k x^{2k+1} / [(2k+1)(2k+1)!]."""
    if x < 0.0:
        return -sine_integral(-x)
    if x < 6.0:
        # Build series: term_k = (-1)^k x^{2k+1} / (2k+1)!
        # Then Si(x) = Σ term_k / (2k+1)
        # Recurrence: term_{k+1} = term_k * (-x²) / ((2k+2)(2k+3))
        s = 0.0
        term = x
        x2 = x * x
        for k in range(80):
            s += term / (2 * k + 1)
            term *= -x2 / ((2 * k + 2) * (2 * k + 3))
            if abs(term / (2 * k + 3)) < 1e-16 * abs(s):
                break
        return s
    # Asymptotic
    f, g = 0.0, 0.0
    tf, tg = 1.0 / x, 1.0 / x
    x2 = x * x
    for k in range(1, 30):
        tf *= -(2 * k - 1) * (2 * k) / x2
        tg *= -(2 * k) * (2 * k + 1) / x2
        f += tf
        g += tg
        if abs(tf) + abs(tg) < 1e-15:
            break
    return math.pi / 2.0 - math.cos(x) / x * f - math.sin(x) / x * g


def incomplete_gamma_upper(a: float, x: float) -> float:
    """Γ(a,x) = ∫_x^∞ t^{a-1} e^{-t} dt via series/continued-fraction hybrid.
    Appears when computing the neoclassical viscosity integrals for the
    bootstrap current j_bs ∝ <v_||> ~ Γ(5/2, ε) / Γ(5/2)."""
    if x < 0.0 or a <= 0.0:
        return math.nan
    if x == 0.0:
        return math.gamma(a)
    if x < a + 1.0:
        # Series: γ(a,x) = e^{-x} x^a Σ x^n / (a(a+1)...(a+n))
        term = 1.0 / a
        s = term
        for n in range(1, 300):
            term *= x / (a + n)
            s += term
            if abs(term) < 1e-15 * abs(s):
                break
        return math.gamma(a) - math.exp(-x) * (x ** a) * s
    # Lentz continued fraction for Γ(a,x)
    f = 1e-30
    C = 1e-30
    D = 1.0 / (x + 1.0 - a)
    f = D
    for n in range(1, 300):
        an = n * (a - n)
        bn = x + 2 * n + 1 - a
        D = bn + an * D
        if abs(D) < 1e-30:
            D = 1e-30
        C = bn + an / C
        if abs(C) < 1e-30:
            C = 1e-30
        D = 1.0 / D
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x) * (x ** a) * f


# ---------------------------------------------------------------------------
# Profile classes
# ---------------------------------------------------------------------------

@dataclass
class GSProfiles:
    """Cerfon-Freidberg-like pressure and FF' parametrization."""
    c1: float = 1.0              # pressure amplitude  [Pa]
    alpha: float = 1.5           # pressure peaking exponent
    d1: float = 2.0              # FF' amplitude  [T²]
    beta: float = 1.0            # FF' peaking exponent
    d2: float = 0.0              # vacuum toroidal field coefficient

    def pp(self, psi_n: float) -> float:
        """dp/dψ_n, eq. (1) in module docstring."""
        psi_n = max(0.0, min(1.0, psi_n))
        a = self.alpha
        if a <= 0.0:
            return self.c1 * (1.0 - 1.0) if psi_n > 0 else 0.0
        return self.c1 * ((1.0 + a) * (psi_n ** a) - a * psi_n)

    def ffp(self, psi_n: float) -> float:
        """F dF/dψ_n, eq. (2)."""
        psi_n = max(0.0, min(1.0, psi_n))
        b = self.beta
        term1 = 0.0
        if b > 0.0:
            term1 = (1.0 + b) * (psi_n ** b) - b * psi_n
        return self.d1 * term1 + self.d2 * psi_n

    def pressure(self, psi_n: float) -> float:
        """p(ψ_n) obtained by analytic integration of p'(ψ_n)."""
        psi_n = max(0.0, min(1.0, psi_n))
        a = self.alpha
        return self.c1 * ((1.0 + a) * psi_n ** (a + 1.0) / (a + 1.0)
                          - a * psi_n ** 2 / 2.0) - \
               self.c1 * ((1.0 + a) / (a + 1.0) - a / 2.0) * 0.0  # reference 0 at ψ=0

    def toroidal_field(self, psi_n: float, R0: float, B0: float) -> float:
        """F(ψ_n) = R B_φ, normalised so F(1) = R0 B0."""
        psi_n = max(0.0, min(1.0, psi_n))
        b = self.beta
        f_boundary = R0 * B0
        term1 = 0.0
        if b > 0.0:
            term1 = self.d1 * ((1.0 + b) * psi_n ** (b + 1.0) / (b + 1.0)
                               - b * psi_n ** 2 / 2.0)
        return math.sqrt(max(0.0, f_boundary ** 2 + 2.0 * (term1 + self.d2 * psi_n ** 2 / 2.0)))


# ---------------------------------------------------------------------------
# LIF-like spectroscopic diagnostic analog (mapped from wacl-york project)
# In laser-induced fluorescence the emission intensity ∝ n_e T_e σ(T_e);
# here we invert an observed Stark-broadened line width to local T_e.
# ---------------------------------------------------------------------------

@dataclass
class SpectroscopicDiagnostic:
    """Inversion of a synthetic Stark-broadened Hα signal to local T_e, n_e.
    Hα FWHM (Stark) ≈ 0.549 · (n_e / 10^16 cm^{-3})^{2/3} Å  (Griem 1974)."""
    wavelength_0: float = 6.5628e-7        # Hα rest wavelength [m]
    lambda_min: float = 6.562e-7
    lambda_max: float = 6.564e-7
    n_channels: int = 32

    def synthetic_spectrum(self, T_e: float, n_e: float) -> list[float]:
        """Build a Gaussian (thermal+instrumental) + Stark Lorentzian Voigt-ish
        profile sampled on a fixed wavelength grid."""
        lam = [self.lambda_min + i * (self.lambda_max - self.lambda_min) / (self.n_channels - 1)
               for i in range(self.n_channels)]
        # Thermal Doppler width (Gaussian)
        m_H = 1.67262192e-27
        k_B = 1.380649e-23
        sigma_D = self.wavelength_0 * math.sqrt(2.0 * k_B * max(T_e, 1.0) / m_H) / 2.998e8
        sigma_D = max(sigma_D, 1e-14)
        # Stark width (Lorentzian HWHM)
        n_e_16 = max(n_e, 1e14) / 1e22     # m^-3 -> 10^16 cm^-3 units (1e22 m^-3)
        gamma_S = 0.5 * 0.549e-10 * (n_e_16 ** (2.0 / 3.0))
        # Convolution-free pseudo-Voigt sum
        spec = []
        for L in lam:
            dL = L - self.wavelength_0
            gauss = math.exp(-0.5 * (dL / sigma_D) ** 2) / (sigma_D * math.sqrt(2.0 * math.pi))
            lorentz = (gamma_S / math.pi) / (dL * dL + gamma_S * gamma_S + 1e-40)
            spec.append(0.5 * gauss + 0.5 * lorentz)
        return spec

    def invert(self, spectrum: list[float]) -> tuple[float, float]:
        """Second-moment width -> T_e, n_e using Griem relation."""
        lam = [self.lambda_min + i * (self.lambda_max - self.lambda_min) / (self.n_channels - 1)
               for i in range(self.n_channels)]
        s0 = sum(spectrum) + 1e-40
        mean = sum(L * s for L, s in zip(lam, spectrum)) / s0
        var = sum(s * (L - mean) ** 2 for L, s in zip(lam, spectrum)) / s0
        sigma = math.sqrt(max(var, 1e-30))
        # Doppler -> T_e (Gaussian component approx half the width)
        k_B = 1.380649e-23
        m_H = 1.67262192e-27
        T_e = 0.5 * (sigma * 2.998e8 / self.wavelength_0) ** 2 * m_H / k_B
        # Stark -> n_e: FWHM ≈ 2·σ·√(2 ln 2) with Lorentz contribution ~50%
        fwhm = 2.0 * sigma * math.sqrt(2.0 * math.log(2.0))
        n_e_16 = (fwhm / 0.549e-10) ** 1.5 if fwhm > 1e-20 else 1e-4
        n_e = n_e_16 * 1e22
        return T_e, n_e
