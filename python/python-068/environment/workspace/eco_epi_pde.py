"""
eco_epi_pde.py
Core coupled eco-epidemiological PDE-ODE system definitions.

This module defines the full spatially-explicit reaction-diffusion-advection
system for two competing species with cross-species disease transmission.
"""

import numpy as np
from scipy.fft import fft2, ifft2


class EcoEpidemicPDE:
    """
    Coupled eco-epidemiological reaction-diffusion-advection system.

    Spatial domain: periodic [0, Lx] x [0, Ly]
    State variables: S1, I1, R1, S2, I2, R2 (6 fields)

    Equations:
    ∂S1/∂t = D_s1*∇²S1 - v·∇S1 + r(x,y)*S1*(1 - (N1+α12*N2)/K(x,y)) - β11*S1*I1 - β12*S1*I2
    ∂I1/∂t = D_i1*∇²I1 - v·∇I1 + β11*S1*I1 + β12*S1*I2 - (γ1+μ1)*I1
    ∂R1/∂t = D_r1*∇²R1 - v·∇R1 + γ1*I1
    (and similarly for species 2)
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0 * np.pi,
        Ly: float = 2.0 * np.pi,
        params: dict = None
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.params = params if params is not None else self.default_params()

        # Grid coordinates
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Habitat maps
        from habitat_surface import create_habitat_carrying_capacity, create_growth_rate_map
        self.K = create_habitat_carrying_capacity(nx, ny,
                                                   K_base=self.params['K_base'],
                                                   K_peak=self.params['K_peak'])
        self.r = create_growth_rate_map(nx, ny,
                                         r_base=self.params['r_base'],
                                         r_peak=self.params['r_peak'])

        # Diffusion coefficients for each field
        self.D = [
            self.params['D_s1'],
            self.params['D_i1'],
            self.params['D_r1'],
            self.params['D_s2'],
            self.params['D_i2'],
            self.params['D_r2'],
        ]

        # Advection velocity
        self.vx = self.params['vx']
        self.vy = self.params['vy']

    @staticmethod
    def default_params() -> dict:
        return {
            'D_s1': 0.01, 'D_i1': 0.005, 'D_r1': 0.01,
            'D_s2': 0.008, 'D_i2': 0.004, 'D_r2': 0.008,
            'vx': 0.1, 'vy': 0.05,
            'beta11': 0.3, 'beta12': 0.08,
            'beta21': 0.06, 'beta22': 0.25,
            'gamma1': 0.1, 'gamma2': 0.08,
            'mu1': 0.02, 'mu2': 0.015,
            'alpha12': 0.6, 'alpha21': 0.5,
            'K_base': 80.0, 'K_peak': 150.0,
            'r_base': 0.4, 'r_peak': 1.2,
            'selkov_threshold': 0.1,
        }

    def nonlinear_terms(self, u: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear reaction terms in physical space.

        Parameters
        ----------
        u : ndarray, shape (6, nx, ny)

        Returns
        -------
        rhs : ndarray, shape (6, nx, ny)
        """
        from reaction_kinetics import compute_reaction_terms
        return compute_reaction_terms(u, self.K, self.r, self.params)

    def compute_total_populations(self, u: np.ndarray) -> dict:
        """
        Compute spatially integrated total populations.
        """
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dA = dx * dy

        totals = {}
        names = ['S1', 'I1', 'R1', 'S2', 'I2', 'R2']
        for i, name in enumerate(names):
            totals[name] = float(np.sum(u[i]) * dA)
        totals['N1'] = totals['S1'] + totals['I1'] + totals['R1']
        totals['N2'] = totals['S2'] + totals['I2'] + totals['R2']
        return totals

    def compute_reproduction_numbers(self, u: np.ndarray) -> dict:
        """
        Compute spatially varying basic reproduction numbers.
        R0_1(x,y) = (β11*S1 + β12*S2) / (γ1 + μ1)  [for species 1 infection]
        R0_2(x,y) = (β21*S1 + β22*S2) / (γ2 + μ2)  [for species 2 infection]
        """
        S1 = u[0]
        S2 = u[3]
        gamma1 = self.params['gamma1']
        gamma2 = self.params['gamma2']
        mu1 = self.params['mu1']
        mu2 = self.params['mu2']
        beta11 = self.params['beta11']
        beta12 = self.params['beta12']
        beta21 = self.params['beta21']
        beta22 = self.params['beta22']

        # === HOLE 2 START ===
        # 修复要求：根据基本繁殖数（Basic Reproduction Number）的定义计算 R0_1 和 R0_2 的空间分布及统计量。
        # 科学知识要点：
        #   R0_1(x,y) = (beta11 * S1 + beta12 * S2) / (gamma1 + mu1)  # 物种1感染的基本繁殖数
        #   R0_2(x,y) = (beta21 * S1 + beta22 * S2) / (gamma2 + mu2)  # 物种2感染的基本繁殖数
        # 需返回 dict，包含空间平均值和最大值：R0_1_mean, R0_1_max, R0_2_mean, R0_2_max
        raise NotImplementedError("HOLE 2: 请实现基本繁殖数 R0_1 和 R0_2 的计算")
        # === HOLE 2 END ===

    def initialize_state(self, seed: int = 42) -> np.ndarray:
        """
        Initialize state with two species populations and localized infection.
        """
        rng = np.random.default_rng(seed)
        u = np.zeros((6, self.nx, self.ny))

        # Species 1: higher population, centered left
        cx1, cy1 = self.Lx * 0.3, self.Ly * 0.5
        u[0] = 30.0 * np.exp(-((self.X - cx1) ** 2 + (self.Y - cy1) ** 2) / 3.0) + rng.normal(0, 0.5, (self.nx, self.ny))
        u[0] = np.maximum(u[0], 0.0)

        # Localized infection for species 1
        u[1] = 2.0 * np.exp(-((self.X - cx1 - 1.0) ** 2 + (self.Y - cy1) ** 2) / 0.5)

        # Species 2: lower population, centered right
        cx2, cy2 = self.Lx * 0.7, self.Ly * 0.5
        u[3] = 20.0 * np.exp(-((self.X - cx2) ** 2 + (self.Y - cy2) ** 2) / 4.0) + rng.normal(0, 0.3, (self.nx, self.ny))
        u[3] = np.maximum(u[3], 0.0)

        # Small infection for species 2
        u[4] = 1.0 * np.exp(-((self.X - cx2 + 1.0) ** 2 + (self.Y - cy2) ** 2) / 0.5)

        return u
