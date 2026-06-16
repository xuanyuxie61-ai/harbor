"""
microstructure_field.py — Material heterogeneity and microstructure generation.

Seeds: 1075_carbon_export (empirical parameterization with spatial maps),
       1011_DiminishedRhythmsPathology (multi-scale band-pass filtering).

Core idea
=========
Real materials have spatially varying microstructure (grain orientations,
pore distributions, phase compositions) that profoundly affects damage
initiation and propagation. We model this by generating random fields
for:

  1. Local tensile strength f_t(x) — Weibull-distributed with spatial correlation
  2. Local fracture energy G_f(x) — correlated random field
  3. Local characteristic length ℓ_c(x) — varies with grain size

The random fields are generated via:
  - Spectral method (FFT-based): generate white noise, filter in Fourier space
    using a prescribed power spectral density S(k) ∝ (k² + k_0²)^{-H-d/2}
    where H is the Hurst exponent and d is the spatial dimension.
  - The correlation length L_c = 1/k_0 controls the spatial scale.

Multi-scale filtering (inspired by seed 1011):
  The microstructure has features at multiple scales:
    - Grain scale: L_grain ~ 1mm
    - Aggregate scale: L_agg ~ 10mm
    - Specimen scale: L_spec ~ 100mm
  We decompose the random field into bands using Butterworth-style filters:
    f_t(x) = f_t,mean * (1 + Σ_j band_j(x))
  where each band captures variability at a different scale.

Spatial parameterization (inspired by seed 1075):
  Following empirical parameterizations for spatially varying properties
  (like Schmidt number, solubility in ocean carbon flux), we use:
    f_t(x,y) = f_t,base * g(x,y) * h(T(x,y))
  where g is a spatial heterogeneity field and h is a temperature correction.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional
from config import SimulationConfig


# ===================================================================
# Random field generation via spectral method
# ===================================================================

def generate_gaussian_random_field(nx: int, ny: int,
                                   dx: float, dy: float,
                                   correlation_length: float,
                                   hurst_exponent: float = 0.5,
                                   seed: int = 42,
                                   variance: float = 1.0) -> np.ndarray:
    """Generate a 2D Gaussian random field with prescribed correlation.

    Uses the spectral (FFT) method:
      1. Generate white noise in physical space
      2. Transform to Fourier space
      3. Multiply by sqrt of power spectral density (filter)
      4. Transform back to physical space

    Power spectral density (Matérn-type):
        S(k) = σ² * (2π)^d * Γ(H+d/2) / (Γ(H) * π^{d/2})
               * k_0^{2H} / (k² + k_0²)^{H+d/2}

    where k_0 = 1/L_c is the corner frequency and H is the Hurst exponent.

    Simplified: S(k) ∝ (k² + k_0²)^{-(H+1)} for 2D.
    """
    rng = np.random.RandomState(seed)

    # Wave numbers
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * math.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * math.pi
    KX, KY = np.meshgrid(kx, ky)
    K_sq = KX ** 2 + KY ** 2

    # Corner wave number
    k0 = 1.0 / max(correlation_length, 1.0e-10)

    # Power spectral density: S(k) ∝ (k² + k_0²)^{-(H+1)}
    exponent = hurst_exponent + 1.0
    S_k = (K_sq + k0 ** 2) ** (-exponent)

    # Avoid DC singularity
    S_k[0, 0] = 0.0

    # Filter in Fourier space
    sqrt_S = np.sqrt(S_k)

    # Generate white noise and transform
    white_noise = rng.randn(ny, nx)
    noise_fft = np.fft.fft2(white_noise)

    # Apply filter
    filtered_fft = noise_fft * sqrt_S

    # Transform back
    field = np.real(np.fft.ifft2(filtered_fft))

    # Normalise to desired variance
    current_std = field.std()
    if current_std > 1.0e-15:
        field = field * math.sqrt(variance) / current_std

    # Remove mean (zero-mean field)
    field -= field.mean()

    return field


# ===================================================================
# Multi-scale band decomposition  (seed 1011)
# ===================================================================

def butterworth_band_filter(field: np.ndarray,
                            dx: float, dy: float,
                            k_low: float, k_high: float,
                            order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 2D field.

    The filter transfer function is:
        H(k) = 1 / (1 + (k/k_c)^{2n})

    For band-pass: H_bp(k) = H_lp(k_high) * (1 - H_lp(k_low))

    This separates the field into different spatial frequency bands,
    analogous to extracting biological rhythms from EEG signals
    (seed 1011 uses Butterworth for band-pass filtering of bandpower).
    """
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * math.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * math.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1.0e-15  # avoid division by zero

    # Low-pass component at k_high
    if k_high > 0:
        H_high = 1.0 / (1.0 + (K / k_high) ** (2 * order))
    else:
        H_high = np.ones_like(K)

    # Low-pass component at k_low
    if k_low > 0:
        H_low = 1.0 / (1.0 + (K / k_low) ** (2 * order))
    else:
        H_low = np.zeros_like(K)

    # Band-pass = H_high - H_low (pass frequencies between k_low and k_high)
    H_bp = H_high - H_low
    H_bp = np.clip(H_bp, 0.0, 1.0)

    # Apply filter
    field_fft = np.fft.fft2(field)
    filtered_fft = field_fft * H_bp
    filtered = np.real(np.fft.ifft2(filtered_fft))

    return filtered


def multi_scale_decomposition(field: np.ndarray,
                              dx: float, dy: float,
                              n_bands: int = 3) -> Dict[str, np.ndarray]:
    """Decompose a field into multiple spatial frequency bands.

    Band boundaries are logarithmically spaced between the domain
    scale and the grid scale:
        k_min = 2π / L_domain
        k_max = π / h_grid
        k_j = k_min * (k_max/k_min)^{j/n_bands}
    """
    ny, nx = field.shape
    Lx = nx * dx
    Ly = ny * dy
    L_domain = max(Lx, Ly)
    h_grid = min(dx, dy)

    k_min = 2.0 * math.pi / L_domain
    k_max = math.pi / h_grid

    bands = {}
    for j in range(n_bands):
        k_lo = k_min * (k_max / k_min) ** (j / n_bands)
        k_hi = k_min * (k_max / k_min) ** ((j + 1) / n_bands)
        band_name = f"band_{j}"
        bands[band_name] = butterworth_band_filter(field, dx, dy, k_lo, k_hi)

    return bands


# ===================================================================
# Weibull-distributed material properties
# ===================================================================

def weibull_random_field(base_field: np.ndarray,
                         modulus: float,
                         scale_value: float) -> np.ndarray:
    """Transform a Gaussian random field to Weibull-distributed values.

    The Weibull distribution is commonly used for material strength:
        P(X ≤ x) = 1 - exp(-(x/λ)^m)

    where m is the Weibull modulus (shape) and λ is the scale parameter.

    We use the inverse CDF transform:
        x = λ * (-ln(1 - Φ(g)))^{1/m}

    where Φ is the standard normal CDF and g is the Gaussian field.
    This preserves spatial correlation while giving Weibull marginal.
    """
    # Standard normal CDF
    from scipy.special import erf as _erf
    # Φ(g) = 0.5 * (1 + erf(g / √2))
    # Normalise base_field to unit variance
    g_std = base_field.std()
    if g_std > 1.0e-15:
        g_norm = base_field / g_std
    else:
        g_norm = np.zeros_like(base_field)

    phi = 0.5 * (1.0 + _erf(g_norm / math.sqrt(2.0)))

    # Clamp to avoid log(0) and log(1)
    phi = np.clip(phi, 1.0e-10, 1.0 - 1.0e-10)

    # Inverse Weibull CDF
    weibull_values = scale_value * (-np.log(1.0 - phi)) ** (1.0 / modulus)

    return weibull_values


# ===================================================================
# Spatially-varying material property fields
# ===================================================================

def generate_microstructure_fields(cfg: SimulationConfig) -> Dict[str, np.ndarray]:
    """Generate all spatially-varying material property fields.

    Returns dict with:
      tensile_strength: f_t(x,y) [Pa]
      fracture_energy: G_f(x,y) [J/m²]
      characteristic_length: l_c(x,y) [m]
      young_modulus: E(x,y) [Pa]
      damage_threshold: kappa_0(x,y)
    """
    nx, ny = cfg.numerical.nx, cfg.numerical.ny
    dx, dy = cfg.dx(), cfg.dy()
    mat = cfg.material

    # Base correlation length (grain size)
    L_grain = mat.characteristic_length * 0.5

    # Generate Gaussian random field for heterogeneity
    heterogeneity = generate_gaussian_random_field(
        nx, ny, dx, dy,
        correlation_length=L_grain,
        hurst_exponent=0.7,  # moderately smooth
        seed=42,
        variance=0.1,
    )

    # Multi-scale decomposition
    bands = multi_scale_decomposition(heterogeneity, dx, dy, n_bands=3)

    # Combine bands with different weights (grain, aggregate, structural)
    weights = [0.5, 0.3, 0.2]
    combined = np.zeros((ny, nx))
    for j, (key, band_field) in enumerate(sorted(bands.items())):
        if j < len(weights):
            combined += weights[j] * band_field

    # Exponentiate to get multiplicative heterogeneity
    spatial_factor = np.exp(combined)

    # Generate Weibull-distributed tensile strength
    ft_field = weibull_random_field(
        heterogeneity, mat.weibull_modulus, mat.tensile_strength
    )
    # Apply spatial modulation
    ft_field *= spatial_factor
    ft_field = np.maximum(ft_field, mat.tensile_strength * 0.1)  # floor

    # Fracture energy: correlated with tensile strength
    # G_f ∝ f_t * l_c  (characteristic energy)
    gf_field = mat.fracture_energy * (ft_field / mat.tensile_strength) ** 0.5

    # Characteristic length: varies with local microstructure
    lc_field = mat.characteristic_length * spatial_factor
    lc_field = np.clip(lc_field,
                       mat.characteristic_length * 0.3,
                       mat.characteristic_length * 3.0)

    # Young's modulus: slight spatial variation
    E_field = mat.young_modulus * (1.0 + 0.05 * combined)
    E_field = np.maximum(E_field, mat.young_modulus * 0.8)

    # Damage threshold: inversely related to strength
    kappa_0_field = ft_field / E_field

    return {
        "tensile_strength": ft_field,
        "fracture_energy": gf_field,
        "characteristic_length": lc_field,
        "young_modulus": E_field,
        "damage_threshold": kappa_0_field,
        "heterogeneity_base": heterogeneity,
        "spatial_factor": spatial_factor,
        "bands": bands,
    }


# ===================================================================
# Empirical parameterization  (seed 1075)
# ===================================================================

def temperature_correction_factor(temperature: np.ndarray,
                                  T_ref: float = 293.15,
                                  activation_energy: float = 50.0e3,
                                  ) -> np.ndarray:
    """Temperature correction for material properties.

    Following Arrhenius-type dependence (similar to solubility/gas-transfer
    parameterizations in seed 1075):
        h(T) = exp( -E_a / R * (1/T - 1/T_ref) )

    where E_a is the activation energy and R is the gas constant.
    """
    R_gas = 8.314  # J/(mol·K)
    correction = np.exp(-activation_energy / R_gas * (1.0 / temperature - 1.0 / T_ref))
    return correction


def Schmidt_number_analog(temperature: np.ndarray,
                          reference_value: float = 660.0) -> np.ndarray:
    """Compute a Schmidt-number analog for damage diffusion.

    In oceanography (seed 1075), the Schmidt number Sc = ν/D relates
    momentum diffusivity to mass diffusivity. For the damage problem,
    we define an analog:
        Sc_dam = μ / D_dam

    where D_dam is the damage diffusion coefficient.

    Empirical polynomial fit (Wanninkhof 2014 style):
        Sc(T) = a₀ + a₁T + a₂T² + a₃T³ + a₄T⁴
    """
    T_celsius = temperature - 273.15
    # Polynomial coefficients (example for CO2 in seawater, adapted)
    a0, a1, a2, a3, a4 = 2116.8, -136.25, 4.7353, -0.092307, 0.00075559
    Sc = a0 + a1 * T_celsius + a2 * T_celsius ** 2 + a3 * T_celsius ** 3 + a4 * T_celsius ** 4
    return np.maximum(Sc, 100.0)  # floor to avoid unphysical values
