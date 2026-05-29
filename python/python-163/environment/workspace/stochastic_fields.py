r"""
stochastic_fields.py
====================
Stochastic permeability and thermal diffusivity fields for heterogeneous
geothermal reservoirs, including random sampling for uncertainty quantification.

Incorporates algorithms from:
  - 1169_stochastic_diffusion: 2D stochastic diffusivity function
  - 837_opt_sample: random sampling for parameter estimation

Mathematical formulation:
The stochastic permeability field is modeled as a log-normal random field:

  \ln k(\mathbf{x}, \boldsymbol{\omega}) = \ln k_0
  + \sum_{m=1}^{M} \sqrt{\lambda_m} \, \phi_m(\mathbf{x}) \, \xi_m(\boldsymbol{\omega})

where {\lambda_m, \phi_m} are eigenpairs of the covariance kernel,
and \xi_m are independent standard normal random variables.

For the stochastic diffusivity (thermal conductivity analogy):

  D(\mathbf{x}; \boldsymbol{\omega}) = D_0 + \exp\left[\sum_{j=1}^{4}
  \omega_j \psi_j(\mathbf{x})\right]

where \psi_j are deterministic basis functions and \omega_j are
stochastic parameters (Karhunen-Loève expansion truncated at M=4).

The covariance kernel (exponential/squared-exponential):
  C(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|}{L_c}\right)

where \sigma^2 is variance and L_c is correlation length.
"""

import numpy as np


class StochasticDiffusivity2D:
    """
    2D stochastic diffusivity / thermal conductivity field.
    Based on the BNT (Babuska-Nobile-Tempone) stochastic diffusion model.
    """

    def __init__(self, D0=10.0, sigma=0.5):
        """
        Parameters
        ----------
        D0 : float
            Constant term in diffusivity expansion.
        sigma : float
            Standard deviation of stochastic fluctuation.
        """
        self.D0 = float(D0)
        self.sigma = float(sigma)

    def evaluate(self, omega, x, y):
        """
        Evaluate the stochastic diffusivity field.

        Parameters
        ----------
        omega : np.ndarray, shape (4,)
            Stochastic parameters (i.i.d., mean 0, variance 1).
        x : np.ndarray
            x-coordinates.
        y : np.ndarray
            y-coordinates.

        Returns
        -------
        D : np.ndarray
            Diffusivity values.
        """
        omega = np.asarray(omega, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if omega.size != 4:
            raise ValueError("omega must have exactly 4 elements.")

        # Basis functions for stochastic representation
        arg = (omega[0] * np.cos(np.pi * x)
               + omega[1] * np.sin(np.pi * x)
               + omega[2] * np.cos(np.pi * y)
               + omega[3] * np.sin(np.pi * y))

        arg = np.exp(-0.125) * arg
        D = self.D0 + np.exp(arg)
        D = np.clip(D, 0.1, 1.0e4)
        return D


class LogNormalPermeabilityField:
    """
    Log-normal stochastic permeability field using Karhunen-Loève expansion.
    """

    def __init__(self, k_mean=1.0e-14, sigma_ln_k=1.0, L_c=50.0):
        """
        Parameters
        ----------
        k_mean : float
            Geometric mean permeability (m^2).
        sigma_ln_k : float
            Standard deviation of ln(k).
        L_c : float
            Correlation length (m).
        """
        self.k_mean = float(k_mean)
        self.sigma_ln_k = float(sigma_ln_k)
        self.L_c = float(L_c)

    def _eigenfunctions_1d(self, x, M=6):
        """
        Approximate eigenfunctions for exponential covariance on [0, L].
        For the exponential kernel C(x,x') = exp(-|x-x'|/L_c),
        the eigenfunctions are approximately sinusoidal.
        """
        x = np.asarray(x, dtype=np.float64)
        L = np.max(x) - np.min(x)
        if L <= 0:
            raise ValueError("Domain length must be positive.")
        phi = []
        for m in range(1, M + 1):
            # Approximate eigenfunctions
            if m % 2 == 1:
                pm = np.cos((m * np.pi * (x - np.min(x))) / (2.0 * L))
            else:
                pm = np.sin((m * np.pi * (x - np.min(x))) / (2.0 * L))
            phi.append(pm)
        return np.stack(phi, axis=0)

    def evaluate_1d(self, x, xi=None, M=6):
        """
        Evaluate 1D log-normal permeability field.

        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates.
        xi : np.ndarray, shape (M,)
            Standard normal random coefficients. If None, sample randomly.
        M : int
            Number of KL terms.

        Returns
        -------
        k : np.ndarray
            Permeability values (m^2).
        """
        x = np.asarray(x, dtype=np.float64)
        if xi is None:
            xi = np.random.randn(M)
        xi = np.asarray(xi, dtype=np.float64)
        if xi.size != M:
            raise ValueError(f"xi must have {M} elements.")

        phi = self._eigenfunctions_1d(x, M)
        # Eigenvalues decay as lambda_m ~ 1/m^2 for exponential kernel
        lambda_m = self.sigma_ln_k ** 2 * (self.L_c ** 2) / (1.0 + (np.arange(1, M + 1) * self.L_c / (np.max(x) - np.min(x))) ** 2)
        lambda_m = np.clip(lambda_m, 1.0e-20, None)

        ln_k = np.log(self.k_mean)
        for m in range(M):
            ln_k += np.sqrt(lambda_m[m]) * phi[m, :] * xi[m]

        k = np.exp(ln_k)
        k = np.clip(k, 1.0e-20, 1.0e-8)  # physical bounds for permeability
        return k

    def evaluate_2d(self, x, y, xi=None, M=8):
        """
        Evaluate 2D log-normal permeability field using separable KL expansion.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if xi is None:
            xi = np.random.randn(M)
        xi = np.asarray(xi, dtype=np.float64)

        # Use tensor product of 1D eigenfunctions
        Lx = np.max(x) - np.min(x)
        Ly = np.max(y) - np.min(y)
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain dimensions must be positive.")

        nx = x.size
        ny = y.size
        X, Y = np.meshgrid(x, y, indexing='ij')
        ln_k = np.full((nx, ny), np.log(self.k_mean), dtype=np.float64)

        # Add stochastic fluctuations with decaying eigenvalues
        m_idx = 0
        for i in range(1, 4):
            for j in range(1, 4):
                if m_idx >= M:
                    break
                phi_x = np.sin(i * np.pi * (x - np.min(x)) / Lx)
                phi_y = np.sin(j * np.pi * (y - np.min(y)) / Ly)
                lam = (self.sigma_ln_k ** 2
                       / (1.0 + (i * self.L_c / Lx) ** 2)
                       / (1.0 + (j * self.L_c / Ly) ** 2))
                Phi = np.outer(phi_x, phi_y)
                ln_k += np.sqrt(max(lam, 0.0)) * Phi * xi[m_idx]
                m_idx += 1

        k = np.exp(ln_k)
        k = np.clip(k, 1.0e-20, 1.0e-8)
        return k


class RandomSampler:
    """
    Random sampling for parameter estimation and uncertainty quantification.
    Based on opt_sample algorithm.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(int(seed))

    def sample_min_max(self, func, a, b, n=10001):
        """
        Estimate minimum and maximum of a scalar function by random sampling.

        Parameters
        ----------
        func : callable
            Function f(x) to sample.
        a : float
            Lower bound.
        b : float
            Upper bound.
        n : int
            Number of sample points.

        Returns
        -------
        xmin, fmin, xmax, fmax : float
        """
        if a >= b:
            raise ValueError("Lower bound a must be less than upper bound b.")
        if n <= 0:
            raise ValueError("Number of samples must be positive.")

        x = np.random.rand(n)
        x = a * (1.0 - x) + b * x
        fx = np.asarray([func(val) for val in x])

        fmin = np.min(fx)
        xmin = x[np.argmin(fx)]
        fmax = np.max(fx)
        xmax = x[np.argmax(fx)]
        return float(xmin), float(fmin), float(xmax), float(fmax)

    def latin_hypercube(self, dim, n, bounds):
        """
        Generate Latin Hypercube Samples for d-dimensional parameter space.

        Parameters
        ----------
        dim : int
            Dimensionality.
        n : int
            Number of samples.
        bounds : list of (float, float)
            [(a1, b1), (a2, b2), ...] for each dimension.

        Returns
        -------
        samples : np.ndarray, shape (n, dim)
        """
        if len(bounds) != dim:
            raise ValueError("bounds must have length equal to dim.")
        samples = np.zeros((n, dim))
        for d in range(dim):
            a, b = bounds[d]
            # Generate LHS bins
            cut = np.linspace(0, 1, n + 1)
            u = np.random.rand(n)
            a_points = cut[:-1]
            b_points = cut[1:]
            points = a_points + u * (b_points - a_points)
            np.random.shuffle(points)
            samples[:, d] = a + points * (b - a)
        return samples

    def monte_carlo_expectation(self, func, sampler, n=10000):
        """
        Estimate E[func(X)] by Monte Carlo sampling.
        """
        samples = sampler(n)
        values = np.asarray([func(s) for s in samples])
        return np.mean(values), np.std(values) / np.sqrt(n)


def generate_stochastic_permeability_realization(params, n_realizations=1):
    """
    Generate stochastic permeability realizations for the reservoir grid.

    Parameters
    ----------
    params : THMParameters
    n_realizations : int

    Returns
    -------
    realizations : list of np.ndarray
    """
    from thm_model import THMParameters
    if not isinstance(params, THMParameters):
        raise TypeError("params must be a THMParameters instance.")

    nx, nz, ny = params.grid_shape()
    Lx = params.reservoir_length
    Ly = params.reservoir_width
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    field_gen = LogNormalPermeabilityField(
        k_mean=params.matrix_permeability,
        sigma_ln_k=0.5,
        L_c=min(Lx, Ly) / 4.0
    )

    realizations = []
    for _ in range(n_realizations):
        k_2d = field_gen.evaluate_2d(x, y, M=8)
        # Extend to 3D by replication along z
        k_3d = np.tile(k_2d[:, :, np.newaxis], (1, 1, nz))
        realizations.append(k_3d)

    return realizations
